#include "metalgb/engine.hpp"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace metalgb {
void Learner::clear_linear_parameters() {
  // Configure() recreates the reference linear updater from only the current
  // argument batch. Updater parameters absent from that batch use defaults.
  for (auto key : {"learning_rate", "reg_alpha", "reg_lambda", "eta", "alpha", "lambda", "feature_selector", "top_k"}) params.erase(key);
  linear_configuration_dirty_ = true;
}
void Learner::configure_linear() {
  if (!linear_configuration_dirty_) return;
  auto updater = str("updater", "shotgun");
  require(updater != "gpu_coord_descent", "gpu_coord_descent is deprecated. Use device='metal', updater='coord_descent'.");
  require(updater == "shotgun" || updater == "coord_descent", "Unknown linear updater: " + updater);
  auto selector = str("feature_selector", "cyclic");
  require(selector == "cyclic" || selector == "shuffle" || selector == "random" || selector == "greedy" || selector == "thrifty",
          "Unknown linear feature_selector: " + selector);
  require(updater != "shotgun" || selector == "cyclic" || selector == "shuffle",
          "Unsupported feature selector for shotgun updater. Supported options are: {cyclic, shuffle}");
  for (auto key : {"learning_rate", "reg_alpha", "reg_lambda", "tolerance"})
    require(number(key, 0) >= 0, "Parameter must be nonnegative: " + std::string(key));
  if (updater == "coord_descent") require(integer("top_k", 0) <= INT32_MAX, "top_k exceeds the integer range.");
  auto max_rows = str("max_row_perbatch", "18446744073709551615");
  std::size_t consumed;
  require(!max_rows.empty() && max_rows.front() != '-', "max_row_perbatch must be nonnegative.");
  (void)std::stoull(max_rows, &consumed);
  require(consumed == max_rows.size(), "Invalid max_row_perbatch.");
  // Like the reference updater, reconfiguration recreates the feature selector.
  linear_order_.clear();
  linear_configuration_dirty_ = false;
}
void Learner::upload_linear() {
  auto count = checked_product(static_cast<std::size_t>(num_feature) + 1, groups);
  if (linear_weights.empty()) linear_weights.resize(count, 0);
  require(linear_weights.size() == count, "Invalid linear model weight shape.");
  if (!linear_model_) linear_model_ = Buffer(linear_weights.data(), count * sizeof(float));
  if (!base_buffer_) base_buffer_ = Buffer(base_margin.data(), base_margin.size() * sizeof(float));
}
Buffer Learner::linear_buffer(std::string key, std::size_t bytes) {
  auto& buffer = linear_workspace_[std::move(key)];
  if (!buffer || buffer.size() < bytes) buffer = Buffer(bytes);
  return buffer.prefix(bytes);
}
void Learner::update_linear(Matrix const& data, Buffer const& gradient) {
  upload_linear();
  bool fused_enabled = integer("metal_linear_fused", 1) != 0;
  u32 tile_rows = (fused_enabled && data.rows > 1024)
                      ? std::max<u32>(512u, static_cast<u32>((div_up(data.rows, 256) + 255) & ~std::size_t(255)))
                      : kLinearRowsPerTile;
  u32 has_w_packed = (static_cast<u32>(!data.get_info("weight").empty()) & 1u) | (tile_rows << 1);
  LinearParams p{data.rows, num_feature, groups, 0, num_feature, 0,
                 static_cast<u32>(std::max<std::size_t>(1, div_up(data.rows, tile_rows))),
                 has_w_packed,
                 number("learning_rate", 0.5f), number("reg_alpha", 0), number("reg_lambda", 0), number("tolerance", 0)};
  if (p.tolerance > 0) {
    if (linear_converged_) return;
    if (!linear_previous_) linear_previous_ = Buffer(linear_weights.data(), linear_model_.size());
    else {
      auto converged = linear_buffer("converged", sizeof(u32));
      Command check("linear convergence", true);
      check.dispatch("linear_convergence", {linear_model_, linear_previous_, converged}, p, {1, 1, 1});
      record("linear_convergence", check.wait());
      linear_converged_ = *converged.as<u32>() != 0;
      if (linear_converged_) return;
    }
  }
  auto selector = str("feature_selector", "cyclic");
  bool ranked = selector == "greedy" || selector == "thrifty";
  auto partial = linear_buffer("partial", checked_product(checked_product(p.tiles, ranked ? num_feature : 1), kSurvivalSumBytes));
  auto count = checked_product(num_feature, groups);
  auto deltas = linear_buffer("deltas", ranked ? checked_product(count, kLinearRealBytes) : 0);
  auto scores = linear_buffer("scores", ranked ? checked_product(count, sizeof(float)) : 0);
  auto update = linear_buffer("update", sizeof(LinearUpdate));
  auto order_buf = linear_buffer("order", checked_product(std::max<std::size_t>(count, 1), sizeof(u32)));
  bool sparse_csc = data.is_sparse() && data.cols == num_feature;
  u32 max_col_nnz = 0;
  if (sparse_csc) {
    data.ensure_csc();
    auto const* cp = data.sparse_->csc_col_ptr.as<u32>();
    for (u32 c = 0; c < num_feature; ++c) max_col_nnz = std::max(max_col_nnz, cp[c + 1] - cp[c]);
  }
  auto command = std::make_unique<Command>("linear coordinate descent", true);
  Buffer columns;
  if (!sparse_csc) {
    columns = linear_buffer("columns", checked_product(checked_product(data.rows, data.cols), sizeof(float)));
    if (linear_matrix_.lock().get() != &data || linear_generation_ != data.generation) {
      command->dispatch("linear_columns", {data.dense_data(), columns}, p, {div_up(data.rows, 32), div_up(data.cols, 32), 1}, {32, 8, 1});
      linear_matrix_ = data.weak_from_this(); linear_generation_ = data.generation;
    }
  }
  if (!linear_mass_) {
    // Instance weight normalization belongs to the booster, not to a DMatrix
    // cache: the pinned implementation freezes it on the first boosting call.
    linear_mass_ = Buffer(kLinearRealBytes);
    command->dispatch("linear_weight_partial", {data.weights, partial}, p, {p.tiles, 1, 1});
    command->dispatch("linear_weight_total", {partial, linear_mass_}, p, {1, 1, 1});
  }
  auto sums = [&] {
    auto features = (p.all_features & 1u) ? num_feature : 1;
    if (sparse_csc) {
      LinearParams ps = p;
      if (p.all_features & 3u) ps.tiles = std::max<u32>(1, div_up(max_col_nnz, tile_rows));
      else if (p.feature < num_feature) {
        auto const* cp = data.sparse_->csc_col_ptr.as<u32>();
        ps.tiles = std::max<u32>(1, div_up(cp[p.feature + 1] - cp[p.feature], tile_rows));
      }
      command->dispatch("linear_partial_csc",
                        {data.sparse_->csc_col_ptr, data.sparse_->csc_row_idx, data.sparse_->csc_val, gradient, partial, order_buf},
                        ps, {ps.tiles, features, 1});
      command->dispatch("linear_reduce", {partial, linear_model_, linear_mass_, deltas, scores, update, order_buf}, ps, {features, 1, 1});
    } else {
      command->dispatch("linear_partial", {columns, gradient, partial, order_buf}, p, {p.tiles, features, 1});
      command->dispatch("linear_reduce", {partial, linear_model_, linear_mass_, deltas, scores, update, order_buf}, p, {features, 1, 1});
    }
  };
  auto residual = [&] {
    if (sparse_csc) {
      std::size_t work = data.rows;
      if (p.all_features & 3u) work = max_col_nnz;
      else if (p.feature < num_feature) {
        auto const* cp = data.sparse_->csc_col_ptr.as<u32>();
        work = cp[p.feature + 1] - cp[p.feature];
      }
      if (work)
        command->dispatch("linear_residual_csc",
                          {data.sparse_->csc_col_ptr, data.sparse_->csc_row_idx, data.sparse_->csc_val, gradient, update},
                          p, {div_up(work, 256), 1, 1});
    } else {
      command->dispatch("linear_residual", {columns, gradient, update}, p, {div_up(data.rows, 256), 1, 1});
    }
  };
  for (p.group = 0; p.group < groups; ++p.group) { sums(); residual(); }
  if (selector == "shuffle") {
    if (linear_order_.empty()) { linear_order_.resize(num_feature); std::iota(linear_order_.begin(), linear_order_.end(), 0); }
    std::shuffle(linear_order_.begin(), linear_order_.end(), rng_);
  }
  // Greedy and thrifty stop before the last counter value in the pinned source.
  u32 steps = ranked ? std::min(num_feature ? num_feature - 1 : 0u, integer("top_k", 0) ? integer("top_k", 0) : UINT32_MAX) : num_feature;
  bool use_fused = selector != "greedy" && fused_enabled && data.cols == num_feature &&
                   (sparse_csc ? (max_col_nnz <= 1024u * groups) : (data.rows <= 1024u * groups));
  bool gpu_thrifty_sort = selector == "thrifty" && num_feature <= 2048 && integer("metal_linear_gpu_sort", 1);
  std::vector<std::size_t> ranked_order;
  if (selector == "thrifty") {
    p.all_features = 1;
    for (p.group = 0; p.group < groups; ++p.group) sums();
    if (gpu_thrifty_sort) {
      if (num_feature) command->dispatch("linear_thrifty_sort", {scores, order_buf}, p, {groups, 1, 1});
    } else {
      record("linear", command->wait());
      // Preserve libc++'s observable ordering of equal scores. Only feature IDs
      // are sorted on the host; reductions and candidate deltas are GPU results.
      ranked_order.resize(count); std::iota(ranked_order.begin(), ranked_order.end(), 0);
      for (u32 group = 0; group < groups; ++group) {
        auto first = ranked_order.begin() + static_cast<std::size_t>(group) * num_feature;
        std::sort(first, first + num_feature, [&](auto a, auto b) { return std::abs(scores.as<float>()[a]) > std::abs(scores.as<float>()[b]); });
      }
      command = std::make_unique<Command>("linear thrifty coordinates", true);
    }
  }
  if (use_fused) {
    if (steps && !gpu_thrifty_sort) {
      auto* dst = order_buf.as<u32>();
      for (u32 g = 0; g < groups; ++g) {
        for (u32 step = 0; step < steps; ++step) {
          dst[static_cast<std::size_t>(g) * num_feature + step] =
              selector == "shuffle" ? linear_order_[step] :
              selector == "random" ? static_cast<u32>(rng_() % num_feature) :
              selector == "thrifty" ? static_cast<u32>(ranked_order[static_cast<std::size_t>(g) * num_feature + step] % num_feature) : step;
        }
      }
    }
    if (steps) {
      LinearParams pf = p;
      pf.all_features = 0;
      pf.feature = steps;
      if (sparse_csc)
        command->dispatch("linear_coord_fused_csc",
                          {data.sparse_->csc_col_ptr, data.sparse_->csc_row_idx, data.sparse_->csc_val,
                           gradient, linear_model_, linear_mass_, order_buf},
                          pf, {groups, 1, 1});
      else
        command->dispatch("linear_coord_fused",
                          {columns, gradient, linear_model_, linear_mass_, order_buf},
                          pf, {groups, 1, 1});
    }
  } else {
    for (p.group = 0; p.group < groups; ++p.group) for (u32 step = 0; step < steps; ++step) {
      if (selector == "greedy") {
        p.all_features = 1;
        p.feature = step;
      } else if (gpu_thrifty_sort) {
        p.all_features = 2;
        p.feature = step;
      } else {
        p.all_features = 0;
        p.feature = selector == "shuffle" ? linear_order_[step] : selector == "random" ? rng_() % num_feature :
                    selector == "thrifty" ? ranked_order[static_cast<std::size_t>(p.group) * num_feature + step] % num_feature
                                          : step;
      }
      sums();
      if (p.all_features & 1u) command->dispatch("linear_pick", {deltas, scores, linear_model_, update}, p, {1, 1, 1});
      residual();
    }
  }
  record("linear", command->wait());
  std::copy_n(linear_model_.as<float>(), linear_weights.size(), linear_weights.begin());
  active_tuning["linear_coordinate_steps"] = steps;
  active_tuning["linear_rows_per_tile"] = kLinearRowsPerTile;
  active_tuning["linear_fused"] = use_fused;
}
Prediction Learner::predict_linear(Matrix const& data, Json const& config) {
  require(data.cols <= num_feature, "Number of columns does not match number of features in booster.");
  auto type = config.value("type", 0u);
  require(type <= 6, "Invalid prediction type.");
  require(type != 6, "gblinear does not support prediction of leaf index.");
  require(config.value("iteration_begin", 0u) == 0, "Linear booster does not support prediction range.");
  require(type >= 4 || !data.is_quantile, "gblinear does not support QuantileDMatrix.");
  // The reference's interaction implementation omits the bias dimension, and
  // its public shape check rejects nonempty output. Keep that API behavior.
  require(type < 2 || !data.rows || (type < 4 ? data.cols == num_feature : data.cols + 1 == num_feature),
          "gblinear contribution prediction shape is inconsistent with the feature and bias dimensions.");
  auto const& margins = data.get_info("base_margin");
  require(margins.empty() || margins.size() == checked_product(data.rows, groups), "base_margin shape must match rows and output groups.");
  upload_linear();
  PredictParams p{data.rows, data.cols, groups, num_feature, 0, 0, objective(), type,
                  static_cast<u32>(!margins.empty()), groups, 0, 0};
  bool strict = config.value("strict_shape", false);
  bool proba = type == 0 && config.value("metal_proba", false);
  bool binary_proba = proba && groups == 1;
  bool logitraw_multilabel_proba = proba && p.objective == 2 && groups > 1;
  std::size_t columns = static_cast<std::size_t>(data.cols) + 1;
  auto elements = checked_product(data.rows, groups);
  if (type >= 2) elements = checked_product(elements, columns);
  if (type >= 4) elements = checked_product(elements, columns);
  Prediction result{Buffer(checked_product(elements, sizeof(float))), {data.rows}};
  Command command("linear predict", true);
  if (type >= 4) command.zero(result.values);
  else if (data.is_sparse())
    command.dispatch("linear_predict_csr",
                     {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, data.sparse_->csr_val,
                      linear_model_, base_buffer_, data.margins, result.values},
                     p, {div_up(data.rows, 256), groups, 1});
  else command.dispatch("linear_predict", {data.data, linear_model_, base_buffer_, data.margins, result.values}, p, {div_up(data.rows, 256), groups, 1});
  if (binary_proba) {
    Buffer expanded(checked_product(data.rows, 2 * sizeof(float)));
    if (data.rows) command.dispatch("expand_binary_proba", {result.values, expanded}, p, {div_up(data.rows, 256), 1, 1});
    result.values = expanded;
    result.shape.push_back(2);
  } else {
    if (type == 0 && (!(objective() == 15 && config.value("eval", false)) || logitraw_multilabel_proba)) {
      PredictParams p_tx = p;
      if (logitraw_multilabel_proba) p_tx.objective = 1;
      command.dispatch("transform", {result.values}, p_tx, {div_up(data.rows, 256), 1, 1});
    }
    if (type == 0 && objective() == 4 && !config.value("eval", false) && !proba) {
      Buffer classes(data.rows * sizeof(float));
      command.dispatch("argmax", {result.values, classes}, p, {div_up(data.rows, 256), 1, 1});
      result.values = classes;
      if (strict) result.shape.push_back(1);
    } else if (strict || groups > 1 || proba) result.shape.push_back(groups);
  }
  if (type >= 2) result.shape.push_back(columns);
  if (type >= 4) result.shape.push_back(columns);
  if (!data.rows && type < 2 && !proba) result.shape = {0, 0};
  record(type >= 2 ? "shap" : "predict", command.wait());
  return result;
}
Json Learner::linear_config() const {
  auto text = [&](std::string_view name, float fallback) {
    std::ostringstream output; output << std::setprecision(9) << number(name, fallback); return output.str();
  };
  auto eta = text("learning_rate", 0.5f), alpha = text("reg_alpha", 0), lambda = text("reg_lambda", 0);
  Json updater{{"linear_train_param", {{"learning_rate", eta}, {"eta", eta}, {"reg_alpha", alpha}, {"alpha", alpha},
                                     {"reg_lambda", lambda}, {"lambda", lambda}, {"feature_selector", str("feature_selector", "cyclic")}}}};
  if (str("updater", "shotgun") == "coord_descent") updater["coordinate_param"] = {{"top_k", std::to_string(integer("top_k", 0))}};
  return {{"name", "gblinear"}, {"gblinear_train_param", {{"updater", str("updater", "shotgun")}, {"tolerance", text("tolerance", 0)},
           {"max_row_perbatch", str("max_row_perbatch", "18446744073709551615")}}}, {"updater", updater}};
}
std::vector<std::string> Learner::dump_linear(std::string_view format) {
  require(format == "text" || format == "json", "Dump format is not supported by the gblinear model.");
  require(!linear_weights.empty(), "Model is not initialized.");
  std::ostringstream out;
  auto bias = checked_product(num_feature, groups);
  if (format == "text") {
    out << "bias:\n";
    for (u32 group = 0; group < groups; ++group) out << linear_weights[bias + group] << '\n';
    out << "weight:\n";
    for (std::size_t i = 0; i < bias; ++i) out << linear_weights[i] << '\n';
  } else {
    out << "  { \"bias\": [\n";
    for (u32 group = 0; group < groups; ++group) { if (group) out << ",\n"; out << "      " << linear_weights[bias + group]; }
    out << "\n    ],\n    \"weight\": [\n";
    for (std::size_t i = 0; i < bias; ++i) { if (i) out << ",\n"; out << "      " << linear_weights[i]; }
    out << "\n    ]\n  }";
  }
  return {out.str()};
}
}  // namespace metalgb
