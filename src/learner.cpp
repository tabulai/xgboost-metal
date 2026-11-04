#include "metalgb/engine.hpp"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace metalgb {
namespace {
template<class T> Buffer upload(std::vector<T> const& v) { return Buffer(v.data(), v.size() * sizeof(T)); }

bool needs_prediction_transform(u32 objective, u32 groups, bool evaluation) {
  switch (objective) {
    case 1: case 3: case 4: case 5: case 7: case 8: case 10: case 16: return true;
    case 12: case 13: return groups > 1;
    case 15: return !evaluation;
    default: return false;
  }
}
}
Learner::Learner() { Runtime::get(); }
Learner::~Learner() { abandon_rounds(); }
Buffer Learner::pooled_buffer(std::vector<Buffer>& pool, std::size_t bytes) {
  // A pooled buffer is free once the pool holds its only host reference:
  // no cache, recipe, caller result or round in flight can still read it.
  Buffer* best = nullptr;
  for (auto& buffer : pool)
    if (buffer.use_count() == 1 && buffer.size() >= bytes && (!best || buffer.size() < best->size())) best = &buffer;
  if (best) return best->prefix(bytes);
  if (pool.size() >= 64) std::erase_if(pool, [](Buffer const& buffer) { return buffer.use_count() == 1; });
  return pool.emplace_back(bytes).prefix(bytes);
}
Buffer Learner::round_output(std::size_t bytes) { return pooled_buffer(round_output_pool_, bytes); }
void Learner::retire_round() {
  auto round = std::move(inflight_.front());
  inflight_.pop_front();
  try {
    record("gradient_tree", round.command->wait());
    u32 levels = 0;
    auto tree = replay_gpu_tree(round.tree, levels);
    // The next round's margins were computed from the emitted table, so it
    // must equal the host model byte for byte before the tree is published.
    auto const& meta = *round.tree.meta.as<TreeMeta>();
    require(meta.count == tree.nodes.size() &&
            !std::memcmp(round.tree.nodes.data(), tree.nodes.data(), tree.nodes.size() * sizeof(Node)),
            "GPU-emitted tree nodes differ from the host replay.");
    trees.push_back(std::move(tree));
    if (round.tree.round == async_tuning_round_) active_tuning["gpu_tree_levels"] = levels;
  } catch (...) {
    // Every later round derived its gradients from this round's margins.
    // Drop them and recompute training margins from the retained model.
    // Entries stay in place: an update interrupted by this retirement may
    // still reference its cache.
    abandon_rounds();
    rounds = round.tree.round;
    for (auto& cache : prediction_caches_) { cache.margins = {}; cache.rounds = 0; cache.deferred.reset(); }
    throw;
  }
}
void Learner::abandon_rounds() noexcept {
  for (auto& round : inflight_) {
    try { round.command->wait(); } catch (...) {}
  }
  inflight_.clear();
}
void Learner::drain() { while (!inflight_.empty()) retire_round(); }
std::string Learner::str(std::string_view key, std::string fallback) const {
  auto it = params.find(key); return it == params.end() ? fallback : it->second;
}
float Learner::number(std::string_view key, float fallback) const {
  auto it = params.find(key);
  if (it == params.end()) return fallback;
  std::size_t read;
  float value = std::stof(it->second, &read);
  require(read == it->second.size() && std::isfinite(value), "Invalid value for " + std::string(key));
  return value;
}
u32 Learner::integer(std::string_view key, u32 fallback) const {
  auto it = params.find(key);
  if (it == params.end()) return fallback;
  std::size_t read;
  double value = std::stod(it->second, &read);
  require(read == it->second.size() && value >= 0 && value <= UINT32_MAX && std::floor(value) == value, "Invalid nonnegative integer parameter: " + std::string(key));
  return static_cast<u32>(value);
}
bool Learner::boolean(std::string_view key, bool fallback) const {
  auto value = str(key, fallback ? "1" : "0");
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  require(value == "1" || value == "0" || value == "true" || value == "false", "Invalid boolean parameter: " + std::string(key));
  return value == "1" || value == "true";
}
void Learner::configure_rng() {
  auto seed = str("seed", "0");
  if (seed != rng_seed_) {
    std::size_t read;
    auto value = std::stoll(seed, &read);
    require(read == seed.size(), "Invalid random seed.");
    rng_.seed(static_cast<std::mt19937::result_type>(value));
    rng_seed_ = std::move(seed);
  }
}
u32 Learner::random_seed() const {
  std::size_t read;
  auto seed = str("seed", "0");
  auto value = std::stoll(seed, &read);
  require(read == seed.size(), "Invalid random seed.");
  return static_cast<u32>(value);
}
void Learner::set_param(std::string key, std::string value) {
  linear_configuration_dirty_ = true;
  if (key == "eta") key = "learning_rate";
  if (key == "lambda") key = "reg_lambda";
  if (key == "alpha") key = "reg_alpha";
  if (key == "gamma") key = "min_split_loss";
  if (key == "n_jobs") key = "nthread";
  if (key == "random_state") key = "seed";
  if (key == "eval_metric") {
    if (std::find(metrics.begin(), metrics.end(), value) == metrics.end()) {
      metrics.push_back(value);
      if (value.starts_with("ndcg") || value.starts_with("map") || value.starts_with("pre@") ||
          value == "pre" || value == "pre-")
        ranking_metric_configs_[value] = ranking_metric_config(value).at("lambdarank_param");
    }
    return;
  }
  if (key == "device") require(value == "metal" || value == "metal:0" || value == "gpu" || value == "gpu:0", "This build supports only device='metal' (Apple GPU). No CPU or CUDA backend is present.");
  if (initialized && (key == "base_score" || key == "objective" || key == "num_class" || key == "num_target" || key == "quantile_alpha" || key == "expectile_alpha")) {
    auto current = params.find(key);
    if (current == params.end() || current->second != value) {
      configuration_dirty_ = true;
      if (key == "base_score") base_score_changed_ = true;
    }
  }
  bool metric_gain = key == "ndcg_exp_gain";
  params[std::move(key)] = std::move(value);
  if (metric_gain) for (auto& [name, config] : ranking_metric_configs_)
    if (name.starts_with("ndcg")) config["ndcg_exp_gain"] = boolean("ndcg_exp_gain", true) ? "1" : "0";
}
void Learner::configure() {
  // Booster switches, update mode and output/base-score changes rewrite the
  // model below. Retire committed asynchronous rounds first.
  if (!inflight_.empty() && (str("booster", "gbtree") != "gbtree" || configuration_dirty_ ||
                             str("process_type", "default") != "default")) drain();
  if (str("booster", "gbtree") == "dart") {
    static std::once_flag warned;
    std::call_once(warned, [this] {
      if (integer("verbosity", 1)) warning("`booster=dart` is deprecated. Use the tree booster directly with dropout parameters like `rate_drop`, `skip_drop`, or `one_drop`.");
    });
    params["booster"] = "gbtree";
  }
  configure_rng();
  (void)boolean("seed_per_iteration", false);
  if ((is_linear() && (!trees.empty() || !trees_to_update_.empty())) || (!is_linear() && !linear_weights.empty())) {
    trees.clear(); tree_weights_.clear(); linear_weights.clear(); dropped_trees_.clear();
    trees_to_update_.clear(); update_original_weights_ = false;
    linear_mass_ = {}; linear_previous_ = {}; linear_order_.clear(); linear_converged_ = false;
    rounds = 0; reset(false);
  }
  if (is_linear()) configure_linear();
  else { (void)dropout_parameters(); configure_tree(); }
  if (!params.contains("objective") && integer("num_class", num_class) > 1)
    params["objective"] = "multi:softmax";
  if (ranking_objective_cache_ && ranking_objective_cache_->parameters.at("objective") != objective()) {
    ranking_objective_cache_.reset();
    ranking_bias_plus_ = {}; ranking_bias_minus_ = {}; ranking_bias_work_initialized_ = false;
  }
  if (!initialized || !configuration_dirty_) return;
  auto obj = objective();
  auto classes = integer("num_class", num_class), targets = integer("num_target", num_target);
  if (obj == 12 || obj == 13 || obj == 14) {
    auto count = obj == 14 ? 2u : static_cast<u32>(loss_alphas(obj == 12 ? "quantile" : "expectile").size());
    require(targets == 1 || targets == count, "num_target does not match the objective output count.");
    targets = count;
  }
  auto outputs = std::max({1u, classes, targets});
  require(linear_weights.empty() || groups == outputs, "The output count cannot change for an initialized linear model.");
  require(std::all_of(trees.begin(), trees.end(), [&](auto const& tree) { return tree.group < outputs; }),
          "The output count cannot exclude groups used by existing trees.");
  std::vector<float> scores = base_score;
  if (base_score_changed_) {
    auto value = parse_json(str("base_score"));
    scores = value.is_array() ? value.get<std::vector<float>>() : std::vector<float>{value.get<float>()};
  }
  if (scores.size() == 1) scores.resize(outputs, scores[0]);
  require(scores.size() == outputs, "base_score must have one entry per output group.");
  auto margins = score_margins(scores);
  num_class = classes; num_target = targets; groups = outputs;
  base_score = std::move(scores); base_margin = std::move(margins);
  configuration_dirty_ = false; base_score_changed_ = false;
  reset(false);
}
u32 Learner::objective() const {
  static std::map<std::string, u32, std::less<>> const objectives{
    {"reg:squarederror", 0}, {"reg:linear", 0}, {"binary:logistic", 1}, {"reg:logistic", 1},
    {"binary:logitraw", 2}, {"multi:softprob", 3}, {"multi:softmax", 4}, {"count:poisson", 5},
    {"reg:squaredlogerror", 6}, {"reg:gamma", 7}, {"reg:tweedie", 8}, {"reg:pseudohubererror", 9},
    {"binary:hinge", 10}, {"reg:absoluteerror", 11}, {"reg:quantileerror", 12},
    {"reg:expectileerror", 13}, {"reg:normal", 14}, {"survival:aft", 15}, {"survival:cox", 16},
    {"rank:ndcg", 17}, {"rank:map", 18}, {"rank:pairwise", 19}};
  auto name = str("objective", "reg:squarederror");
  auto it = objectives.find(name);
  require(it != objectives.end(), "Objective is not implemented by the Metal backend: " + name);
  return it->second;
}
void Learner::initialize(Matrix const& data, bool custom) {
  configure();
  auto obj = objective();
  if (obj >= 17) (void)ranking_parameters(data.rows);
  auto levels = obj == 12 || obj == 13 ? loss_alphas(obj == 12 ? "quantile" : "expectile") : std::vector<float>{};
  if (obj == 12) require(data.label_cols == 1, "Multi-target is not yet supported by the quantile loss.");
  if (obj == 13) require(data.label_cols == 1, "Multi-target is not yet supported by the expectile loss.");
  if (obj == 14) require(data.label_cols == 1, "Normal regression requires a single response column.");
  if (obj == 15 || obj == 16)
    require(data.label_cols == 1 && integer("num_class", 0) == 0, "Multioutput survival regression is not supported.");
  if (obj >= 17)
    require(data.label_cols == 1 && integer("num_class", 0) == 0, "Multioutput ranking is not supported.");
  if (obj == 15) {
    (void)aft_parameters(data.rows);
    require(custom || (data.get_info("label_lower_bound").size() == data.rows &&
                       data.get_info("label_upper_bound").size() == data.rows),
            "AFT requires one label_lower_bound and label_upper_bound per row.");
  }
  require(data.cols > 0, "Data must contain at least one feature.");
  require(data.rows > 0 || is_tree_update() || obj == 0 || obj == 15 || obj >= 17, "Training data must contain at least one row.");
  require(custom || obj == 15 || data.get_info("label").size() == static_cast<std::size_t>(data.rows) * data.label_cols, "Labels must be provided for training.");
  auto expected_weights = obj >= 17 ? (data.group_ptr.empty() ? 1 : data.group_ptr.size() - 1) : data.rows;
  require(data.get_info("weight").empty() || data.get_info("weight").size() == expected_weights,
          obj >= 17 ? "Ranking weights must have one entry per query group." : "Number of weights must equal the number of rows.");
  // These checks depend only on parameters. While the parameters equal the
  // last fully validated set (metal_parameter_snapshot), skip re-parsing them.
  bool snapshot = integer("metal_parameter_snapshot", 1) != 0;
  bool validate = !snapshot || params != validated_params_;
  if (validate && (obj == 1 || obj == 2)) require(number("scale_pos_weight", 1) >= 0, "scale_pos_weight must be nonnegative.");
  if (validate) require(is_linear() || str("booster", "gbtree") == "gbtree", "Booster is not implemented by the Metal backend: " + str("booster"));
  if (is_linear()) {
    require(!data.is_quantile, "gblinear does not support QuantileDMatrix.");
    require(std::none_of(data.feature_types.begin(), data.feature_types.end(), [](auto const& type) { return type == "c"; }),
            "Categorical features are not supported by gblinear.");
    require(obj < 17 || data.get_info("weight").empty() || data.get_info("weight").size() == data.rows,
            "gblinear cannot normalize query weights as per-row instance weights.");
  } else if (validate) {
  if (!params.contains("updater"))
    require(str("tree_method", "hist") == "hist" || str("tree_method", "hist") == "auto", "Metal tree training requires tree_method='hist' or 'auto'.");
  else require(str("tree_method", "hist") == "hist" || str("tree_method") == "auto" || str("tree_method") == "exact" || str("tree_method") == "approx", "Invalid tree_method.");
  require(str("grow_policy", "depthwise") == "depthwise" || str("grow_policy") == "lossguide", "grow_policy must be depthwise or lossguide.");
  require(str("multi_strategy", "one_output_per_tree") == "one_output_per_tree" || str("multi_strategy") == "multi_output_tree",
          "multi_strategy must be one_output_per_tree or multi_output_tree.");
  require(str("sampling_method", "uniform") == "uniform" || str("sampling_method") == "gradient_based",
          "sampling_method must be uniform or gradient_based.");
  require(integer("max_cat_to_onehot", 4) >= 1 && integer("max_cat_threshold", 64) >= 1,
          "Categorical split thresholds must be positive.");
  for (auto key : {"reg_lambda", "reg_alpha", "min_child_weight", "min_split_loss", "max_delta_step", "learning_rate", "scale_pos_weight"}) require(number(key, 0) >= 0, "Parameter must be nonnegative: " + std::string(key));
  for (auto key : {"subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode"}) {
    auto value = number(key, 1); require(value >= 0 && value <= 1, "Sampling fractions must be in [0, 1].");
  }
  require(is_tree_update() || integer("max_depth", 6) <= 30 || integer("max_leaves", 0), "max_depth must not exceed 30 unless max_leaves is constrained.");
  require(integer("max_leaves", 0) <= (static_cast<u32>(INT32_MAX) + 1) / 2, "max_leaves exceeds the tree node index range.");
  require(is_tree_update() || integer("max_depth", 6) || integer("max_leaves", 0), "max_depth and max_leaves cannot both be unconstrained.");
  require(integer("num_parallel_tree", 1) > 0, "num_parallel_tree must be positive.");
  }
  auto classes = integer("num_class", 0);
  if (!custom && (validated_matrix_.lock().get() != &data || validated_generation_ != data.generation ||
                  validated_objective_ != obj || validated_classes_ != classes)) {
    auto const& labels = data.get_info("label");
    if (obj == 1 || obj == 2 || obj == 10)
      require(std::all_of(labels.begin(), labels.end(), [](float y) { return y >= 0 && y <= 1; }), "label must be in [0,1] for logistic or hinge loss.");
    if (obj == 3 || obj == 4)
      require(std::all_of(labels.begin(), labels.end(), [=](float y) { return y >= 0 && y < classes && std::floor(y) == y; }), "Softmax labels must be integers in [0, num_class).");
    if (obj == 5 || obj == 8)
      require(std::all_of(labels.begin(), labels.end(), [](float y) { return y >= 0; }), "Labels must be nonnegative for Poisson and Tweedie losses.");
    if (obj == 7)
      require(std::all_of(labels.begin(), labels.end(), [](float y) { return y > 0; }), "label must be positive for gamma regression.");
    if (obj == 6)
      require(std::all_of(labels.begin(), labels.end(), [](float y) { return y > -1; }), "label must be greater than -1 for squared log error.");
    validated_matrix_ = data.weak_from_this(); validated_generation_ = data.generation;
    validated_objective_ = obj; validated_classes_ = classes;
  }
  if (validate) {
    if (obj == 8) require(number("tweedie_variance_power", 1.5f) >= 1 && number("tweedie_variance_power", 1.5f) < 2, "tweedie_variance_power must be in [1, 2).");
    if (obj == 9) require(number("huber_slope", 1) > 0, "huber_slope must be positive.");
    if (snapshot) validated_params_ = params;
  }
  if (initialized) {
    require(num_feature == data.cols, "Number of columns does not match the trained model.");
    require(custom || num_target == (obj == 14 ? 2 : !levels.empty() ? levels.size() : data.label_cols) || num_class > 0,
            "Label shape does not match the model.");
    return;
  }
  num_feature = data.cols; num_class = integer("num_class", 0);
  num_target = obj == 14 ? 2 : !levels.empty() ? levels.size() : custom ? integer("num_target", data.label_cols) : data.label_cols;
  if (obj == 3 || obj == 4) require(num_class >= 2 && data.label_cols == 1, "Multiclass requires num_class >= 2 and one label per row.");
  groups = std::max(num_class, num_target);
  auto requested_targets = integer("num_target", 1);
  require(requested_targets == 1 || requested_targets == num_target, "num_target does not match labels or objective outputs.");
  // The public adapter does not infer model feature metadata from empty data.
  if (data.rows) { feature_names = data.feature_names; feature_types = data.feature_types; }
  categories = data.categories;
  base_score.assign(groups, 0.5f); base_margin.assign(groups, 0.5f);
  bool boost_average = integer("boost_from_average", params.contains("base_score") ? 0 : 1) != 0;
  if (params.contains("base_score") && !boost_average) {
    auto text = str("base_score");
    if (!text.empty() && text[0] == '[') base_score = parse_json(text).get<std::vector<float>>();
    else base_score = {std::stof(text)};
    if (base_score.size() == 1) base_score.resize(groups, base_score[0]);
    require(base_score.size() == groups, "base_score must have one entry per output group.");
  } else if (boost_average && !custom && (obj == 11 || obj == 12)) {
    base_score = estimate_quantiles(data);
  } else if (boost_average && !custom && (obj == 13 || obj == 14)) {
    base_score = estimate_distribution(data);
  } else if (boost_average && !custom && (obj == 15 || obj == 16)) {
    base_score.assign(groups, obj == 16 ? 1.0f : 0.5f);
  } else if (boost_average && !custom && obj >= 17) {
    base_score.assign(groups, 0);
  } else if (boost_average && !custom) {
    GradientParams p{data.rows, groups, data.label_cols, obj, static_cast<u32>(!data.get_info("weight").empty()), 0, 0, 0,
                     number("scale_pos_weight", 1), 1, number("huber_slope", 1), 0};
    Buffer partial(checked_product(checked_product(div_up(data.rows, 256), groups), kSurvivalSumBytes));
    Buffer values(groups * sizeof(float));
    Buffer scores(groups * sizeof(float)), margins(groups * sizeof(float));
    Command command("initialize base score");
    command.dispatch("base_partial", {data.labels, data.weights, partial}, p, {div_up(data.rows, 256), groups, 1});
    command.dispatch("base_reduce", {partial, values}, p, {div_up(groups, 64), 1, 1}, {64, 1, 1});
    command.dispatch("base_finalize", {values, scores, margins}, p, {div_up(groups, 64), 1, 1}, {64, 1, 1});
    record("base_score", command.wait());
    std::copy_n(scores.as<float>(), groups, base_score.begin());
    std::copy_n(margins.as<float>(), groups, base_margin.begin());
    if (obj == 10) for (u32 k = 0; k < groups; ++k) base_margin[k] = base_score[k] = (base_score[k] > 0.5f ? 1 : base_score[k] < 0.5f ? -1 : 0);
  }
  // A saved model contains scores, not margins. Use the same scalar conversion
  // as the model reader so GPU transcendental rounding cannot change a reload.
  base_margin = score_margins(base_score);
  initialized = true;
  forest_size_ = std::numeric_limits<std::size_t>::max();
  shap_size_ = std::numeric_limits<std::size_t>::max();
}

void Learner::upload_model() {
  // A deferred prediction may retain this forest's buffers. Finish it before
  // any prefix relocation or upload can change shared storage.
  flush_prediction_caches();
  if (forest_size_ == trees.size()) return;
  // Published trees are immutable. Model replacement, reconfiguration and
  // update-mode transitions invalidate this prefix through reset().
  bool rebuild = forest_size_ > trees.size();
  auto first = rebuild ? 0 : forest_size_;
  auto nodes = rebuild ? 0 : forest_node_count_;
  auto categories = rebuild ? 0 : forest_category_count_;
  auto leaves = rebuild ? 0 : forest_leaf_count_;
  auto total_nodes = nodes, total_categories = categories, total_leaves = leaves;
  auto add = [](std::size_t& count, std::size_t extra) {
    require(extra <= std::numeric_limits<std::size_t>::max() - count, "Forest size overflow.");
    count += extra;
  };
  for (std::size_t t = first; t < trees.size(); ++t) {
    add(total_nodes, trees[t].nodes.size());
    add(total_categories, trees[t].categories.size());
    add(total_leaves, trees[t].leaf_weights.size());
  }
  require(trees.size() <= UINT32_MAX && total_nodes <= UINT32_MAX && total_categories <= UINT32_MAX,
          "Forest exceeds 32-bit tree, node or category offsets.");
  auto reserve = [&](Buffer& buffer, std::size_t count, std::size_t old_count, std::size_t width) {
    auto used = checked_product(old_count, width);
    if (buffer.reserve(checked_product(count, width), used)) {
      ++forest_uploads.buffer_allocations;
      forest_uploads.bytes_relocated += used;
    }
  };
  reserve(forest_nodes_, total_nodes, nodes, sizeof(Node));
  reserve(forest_trees_, trees.size(), first, sizeof(TreeMeta));
  reserve(forest_categories_, total_categories, categories, sizeof(u32));
  reserve(forest_leaf_weights_, total_leaves, leaves, sizeof(float));
  // Dropout can change weights on old trees. Keep this upload independent of
  // the immutable node/metadata prefix, including drop_trees()' same-size path.
  if (!tree_weights_.empty() || !forest_weights_) forest_weights_ = upload(tree_weights_);
  if (rebuild || !base_buffer_) base_buffer_ = upload(base_margin);
  auto* uploaded_trees = forest_trees_.as<TreeMeta>();
  auto* uploaded_nodes = forest_nodes_.as<Node>();
  auto* uploaded_categories = forest_categories_.as<u32>();
  auto* uploaded_leaves = forest_leaf_weights_.as<float>();
  for (std::size_t t = first; t < trees.size(); ++t) {
    auto const& tree = trees[t];
    uploaded_trees[t] = {static_cast<u32>(nodes), static_cast<u32>(tree.nodes.size()),
                         tree.group, tree.round, leaves, tree.outputs, 0};
    for (auto node : tree.nodes) {
      node.category_begin += categories;
      uploaded_nodes[nodes++] = node;
    }
    if (!tree.categories.empty())
      std::memcpy(uploaded_categories + categories, tree.categories.data(), tree.categories.size() * sizeof(u32));
    if (!tree.leaf_weights.empty())
      std::memcpy(uploaded_leaves + leaves, tree.leaf_weights.data(), tree.leaf_weights.size() * sizeof(float));
    categories += tree.categories.size();
    leaves += tree.leaf_weights.size();
  }
  forest_uploads.bytes_uploaded += (trees.size() - first) * sizeof(TreeMeta) +
      (nodes - (rebuild ? 0 : forest_node_count_)) * sizeof(Node) +
      (categories - (rebuild ? 0 : forest_category_count_)) * sizeof(u32) +
      (leaves - (rebuild ? 0 : forest_leaf_count_)) * sizeof(float);
  forest_uploads.trees_uploaded += trees.size() - first;
  if (rebuild) ++forest_uploads.full_uploads;
  else ++forest_uploads.append_uploads;
  // Publish logical lengths last, so allocation failures leave a valid prefix
  // for the next attempt. GPU work is complete before these shared-buffer writes.
  forest_node_count_ = nodes; forest_category_count_ = categories; forest_leaf_count_ = leaves;
  forest_size_ = trees.size();
}
Learner::PredictionCache& Learner::prediction_cache(Matrix const& data) {
  std::erase_if(prediction_caches_, [](auto const& entry) { return entry.matrix.expired(); });
  // Match the reference's per-thread, 64-entry FIFO and half-cache eviction.
  if (prediction_caches_.size() >= 64)
    prediction_caches_.erase(prediction_caches_.begin(), prediction_caches_.end() - 31);
  auto thread = std::this_thread::get_id();
  for (auto& entry : prediction_caches_)
    if (entry.thread == thread && entry.matrix.lock().get() == &data) return entry;
  return prediction_caches_.emplace_back(PredictionCache{data.weak_from_this(), thread, {}, 0, {}});
}
Buffer Learner::encode_deferred_prediction(PredictionCache& cache, Command& command) {
  auto& deferred = *cache.deferred;
  if (!deferred.output)
    deferred.output = Buffer(checked_product(checked_product(deferred.params.rows, deferred.params.groups), sizeof(float)));
  auto const& b = deferred.inputs;
  if (deferred.csr_col_idx)
    command.dispatch("predict_csr", {b[0], deferred.csr_col_idx, deferred.csr_val, b[1], b[2], b[3], b[4],
                                     deferred.output, b[5], b[6], b[7], b[8]},
                     deferred.params, {div_up(deferred.params.rows, 256), deferred.params.groups, 1});
  else
    command.dispatch("predict", {b[0], b[1], b[2], b[3], b[4], deferred.output, b[5], b[6], b[7], b[8]},
                     deferred.params, {div_up(deferred.params.rows, 256), deferred.params.groups, 1});
  return deferred.output;
}
void Learner::finish_deferred_prediction(PredictionCache& cache) {
  cache.margins = cache.deferred->output;
  cache.rounds = cache.deferred->rounds;
  cache.deferred.reset();
}
void Learner::flush_prediction_caches() {
  for (auto& cache : prediction_caches_) {
    if (!cache.deferred) continue;
    if (cache.matrix.expired()) { cache.deferred.reset(); continue; }
    try {
      Command command("update training predictions");
      encode_deferred_prediction(cache, command);
      record("prediction_cache", command.wait());
      finish_deferred_prediction(cache);
    } catch (...) {
      // Keep only the known-good inputs. A retry must allocate a fresh output,
      // never publish a partial result from a failed or unsubmitted command.
      cache.deferred->output = {};
      throw;
    }
  }
}
Prediction Learner::predict(Matrix const& data, Json const& config) {
  drain();
  configure();
  if (config.value("training", false)) initialize(data, true);
  require(initialized, "Model is not yet initialized (not fitted).");
  if (is_linear()) return predict_linear(data, config);
  require(num_feature == 0 || data.cols == num_feature, "Feature count does not match the model.");
  if (!num_feature) num_feature = data.cols;
  u32 type = config.value("type", 0u), begin = config.value("iteration_begin", 0u), end = config.value("iteration_end", 0u);
  if (!end) end = rounds;
  require(begin <= end && end <= rounds, "Invalid prediction iteration range.");
  require(type <= 6, "Invalid prediction type.");
  require(type < 2 || begin == 0 || std::none_of(trees.begin(), trees.end(),
              [begin](auto const& tree) { return tree.round < begin; }),
          "Leaf and contribution predictions support only iteration end; use model slicing for a nonzero start.");
  bool strict = config.value("strict_shape", false);
  bool vector_leaves = type == 6 && std::any_of(trees.begin(), trees.end(), [](auto const& tree) { return tree.outputs > 1; });
  require(!vector_leaves || type != 6 || !strict, "strict_shape with predict leaf is not supported when vector leaf trees are used.");
  bool selected_vector = (type == 3 || type == 5) &&
      std::any_of(trees.begin(), trees.end(), [end](auto const& tree) { return tree.outputs > 1 && tree.round < end; });
  require(!selected_vector || (type != 3 && type != 5), "Approximate predict contribution support for multi-target trees is not implemented by the reference.");
  require(type != 5 || str("multi_strategy", "one_output_per_tree") != "multi_output_tree",
          "Approximate interaction contribution support for multi-target trees is not implemented by the reference.");
  auto const& margins = data.get_info("base_margin");
  require(margins.empty() || margins.size() == static_cast<std::size_t>(data.rows) * groups, "base_margin shape must match rows and output groups.");
  upload_model();
  PredictParams p{data.rows, data.cols, groups, static_cast<u32>(trees.size()), begin, end, objective(), type,
                  static_cast<u32>(!margins.empty()), groups, 0, 0};
  auto masked = type < 2 && config.value("training", false) ? drop_trees() : std::vector<float>{};
  auto weights = masked.empty() ? forest_weights_ : upload(masked);
  p.has_tree_weights = !tree_weights_.empty();
  u32 prediction_threads = integer("metal_predict_threads", 64);
  require(prediction_threads == 32 || prediction_threads == 64 || prediction_threads == 128 || prediction_threads == 256,
          "metal_predict_threads must be 32, 64, 128 or 256.");
  auto cache_bytes = checked_product(checked_product(prediction_threads, data.cols | 1u), sizeof(float));
  bool shared_prediction = integer("metal_predict_shared", 0) && data.cols && cache_bytes <= 32 * 1024;
  Prediction result;
  if (type >= 2 && type <= 5) {
    auto input = data.data_for_categories(categories);
    bool use_csr = data.is_sparse() && !input;
    Command command("predict");
    upload_shap();
    auto columns = static_cast<std::size_t>(data.cols) + 1;
    auto interactions = type == 4 || type == 5;
    auto elements = checked_product(checked_product(data.rows, groups), columns);
    if (interactions) elements = checked_product(elements, columns);
    result.values = Buffer(checked_product(elements, sizeof(float)));
    if (type == 3 || type == 5) {
      if (use_csr)
        command.dispatch("approximate_contributions_csr",
                         {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, data.sparse_->csr_val,
                          forest_nodes_, forest_trees_, node_means_, base_buffer_, data.margins, result.values, forest_categories_, weights},
                         p, {div_up(data.rows, 64), groups, 1}, {64, 1, 1});
      else
        command.dispatch("approximate_contributions",
                         {input, forest_nodes_, forest_trees_, node_means_, base_buffer_, data.margins, result.values, forest_categories_, weights},
                         p, {div_up(data.rows, 64), groups, 1}, {64, 1, 1});
    } else {
      p.trees = shap_leaf_count_;
      bool shap_fast_enabled = integer("metal_shap_fast", 1) != 0;
      p.bin_partition = shap_fast_enabled ? 1u : 0u;
      bool fast_sram = use_csr && type == 2 && data.cols <= 32 && shap_fast_enabled;
      u32 shap_threads = fast_sram ? 32 : 64;
      if (use_csr)
        command.dispatch(fast_sram ? "shap_contributions_csr_fast" : "shap_contributions_csr",
                         {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, data.sparse_->csr_val,
                          shap_paths_, shap_leaves_, base_buffer_, data.margins, result.values, shap_categories_, weights},
                         p, {div_up(data.rows, shap_threads), groups, 1}, {shap_threads, 1, 1});
      else
        command.dispatch(fast_sram ? "shap_contributions_fast" : "shap_contributions",
                         {input, shap_paths_, shap_leaves_, base_buffer_, data.margins, result.values, shap_categories_, weights},
                         p, {div_up(data.rows, shap_threads), groups, 1}, {shap_threads, 1, 1});
    }
    result.shape = {data.rows};
    if (strict || groups > 1) result.shape.push_back(groups);
    result.shape.push_back(columns);
    if (interactions) result.shape.push_back(columns);
    record("shap", command.wait());
  } else if (type == 6) {
    auto input = data.data_for_categories(categories);
    bool use_csr = data.is_sparse() && !input;
    Command command("predict");
    std::vector<TreeMeta> selected;
    for (u32 t = 0; t < trees.size(); ++t) if (trees[t].round >= begin && trees[t].round < end) selected.push_back(forest_trees_.as<TreeMeta>()[t]);
    p.trees = selected.size(); Buffer tree_buffer = upload(selected);
    result.values = Buffer(static_cast<std::size_t>(data.rows) * p.trees * sizeof(float));
    if (use_csr)
      command.dispatch("predict_leaf_csr",
                       {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, data.sparse_->csr_val,
                        forest_nodes_, tree_buffer, result.values, forest_categories_},
                       p, {div_up(data.rows, 256), p.trees, 1});
    else if (shared_prediction)
      command.dispatch("predict_leaf_cached", {input, forest_nodes_, tree_buffer, result.values, forest_categories_},
                       p, {div_up(data.rows, prediction_threads), 1, 1}, {prediction_threads, 1, 1}, cache_bytes);
    else command.dispatch("predict_leaf", {input, forest_nodes_, tree_buffer, result.values, forest_categories_},
                          p, {div_up(data.rows, 256), p.trees, 1});
    if (strict) result.shape = {data.rows, end - begin, groups, integer("num_parallel_tree", 1)};
    else if (p.trees == 1) result.shape = {data.rows};
    else result.shape = {data.rows, p.trees};
    record("predict", command.wait());
  } else {
    auto& cached = prediction_cache(data);
    bool preserve = !begin && margins.empty() && tree_weights_.empty() && masked.empty();
    bool reuse = preserve && cached.rounds && cached.rounds <= end && cached.margins;
    auto input_margins = data.margins;
    if (reuse) {
      p.begin = cached.rounds; p.has_margin = 1; input_margins = cached.margins;
    }
    bool complete = reuse && cached.rounds == end;
    bool proba = type == 0 && config.value("metal_proba", false);
    bool binary_proba = proba && groups == 1;
    bool logitraw_multilabel_proba = proba && p.objective == 2 && groups > 1;
    bool transform = type == 0 &&
                     (needs_prediction_transform(p.objective, groups, config.value("eval", false)) || logitraw_multilabel_proba) &&
                     !binary_proba;
    result.values = complete ? cached.margins : Buffer(static_cast<std::size_t>(data.rows) * groups * sizeof(float));
    auto raw_prediction = result.values;
    bool classes = type == 0 && p.objective == 4 && !config.value("eval", false) && !proba;
    // A complete raw-margin or identity-transform result is already resident.
    // Keep it without submitting an empty command or copying it on the GPU.
    if (!complete || transform || binary_proba) {
      Command command("predict");
      if (!complete) {
        auto input = data.data_for_categories(categories);
        bool tree_cached = !shared_prediction && integer("metal_predict_tree_cached", 1) &&
                           data.rows >= 512 && (end - p.begin) >= 2;
        if (data.is_sparse() && !input) {
          command.dispatch(tree_cached ? "predict_csr_tree_cached" : "predict_csr",
                           {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, data.sparse_->csr_val,
                            forest_nodes_, forest_trees_, base_buffer_, input_margins, result.values,
                            forest_categories_, weights, forest_leaf_weights_, base_buffer_},
                           p, {div_up(data.rows, 256), groups, 1});
        } else {
          command.dispatch(shared_prediction ? "predict_cached" : (tree_cached ? "predict_tree_cached" : "predict"),
                           {input, forest_nodes_, forest_trees_, base_buffer_, input_margins, result.values, forest_categories_, weights, forest_leaf_weights_, base_buffer_}, p,
                           {div_up(data.rows, shared_prediction ? prediction_threads : 256), groups, 1},
                           {shared_prediction ? prediction_threads : 256, 1, 1}, shared_prediction ? cache_bytes : 0);
        }
      }
      if (binary_proba) {
        result.values = Buffer(checked_product(data.rows, 2 * sizeof(float)));
        if (data.rows) command.dispatch("expand_binary_proba", {raw_prediction, result.values}, p, {div_up(data.rows, 256), 1, 1});
      } else if (transform) {
        PredictParams p_tx = p;
        if (logitraw_multilabel_proba) p_tx.objective = 1;
        if (preserve) {
          // Objective transforms must not overwrite the reusable raw margins.
          result.values = Buffer(static_cast<std::size_t>(data.rows) * groups * sizeof(float));
          command.dispatch("transform_copy", {raw_prediction, result.values}, p_tx, {div_up(data.rows, 256), 1, 1});
        } else command.dispatch("transform", {result.values}, p_tx, {div_up(data.rows, 256), 1, 1});
      }
      if (classes) {
        Buffer output(data.rows * sizeof(float));
        command.dispatch("argmax", {result.values, output}, p, {div_up(data.rows, 256), 1, 1});
        result.values = output;
      }
      record("predict", command.wait());
    }
    if (preserve) { cached.margins = raw_prediction; cached.rounds = end; }
    else { cached.margins = {}; cached.rounds = 0; }
    if (binary_proba) {
      result.shape = {data.rows, 2};
    } else if (classes) {
      result.shape = strict ? std::vector<std::uint64_t>{data.rows, 1} : std::vector<std::uint64_t>{data.rows};
    } else result.shape = (strict || groups > 1 || proba) ? std::vector<std::uint64_t>{data.rows, groups} : std::vector<std::uint64_t>{data.rows};
  }
  if (type < 2 || type == 6) {
    if (!data.rows) {
      if (type == 6 && strict) result.shape = {0, end - begin, groups, 1};
      else if (type == 0 && config.value("metal_proba", false)) result.shape = {0, groups == 1 ? 2u : groups};
      else result.shape = {0, 0};
    }
    active_tuning["prediction_shared"] = shared_prediction;
    active_tuning["prediction_threads"] = shared_prediction ? prediction_threads : 256;
  }
  return result;
}
void Learner::update(Matrix& data, int iteration, Array const* custom_gradient, Array const* custom_hessian,
                     Array const* custom_value_gradient, Array const* custom_value_hessian) {
  initialize(data, custom_gradient != nullptr);
  bool vector = !is_linear() && groups > 1 && str("multi_strategy", "one_output_per_tree") == "multi_output_tree";
  auto gradient_groups = groups;
  if (custom_value_gradient) {
    require(!is_linear(), "Multi-target with reduced gradient is not implemented for the current booster.");
    require(str("multi_strategy", "one_output_per_tree") == "multi_output_tree",
            "Reduced gradient must be used with vector leaf trees.");
    require(custom_gradient && custom_hessian && custom_value_hessian,
            "Split and value gradients require matching Hessians.");
    auto valid_shape = [&](Array const& array) {
      return (array.shape.size() == 1 || array.shape.size() == 2) && array.shape[0] == data.rows;
    };
    require(valid_shape(*custom_gradient) && valid_shape(*custom_value_gradient) &&
            custom_gradient->shape == custom_hessian->shape && custom_value_gradient->shape == custom_value_hessian->shape,
            "Gradient and Hessian shapes must match and contain one row per training sample.");
    auto columns = custom_gradient->shape.size() == 1 ? 1 : custom_gradient->shape[1];
    require(columns > 0 && columns <= groups, "The split gradient must have between one and num_target output columns.");
    auto value_columns = custom_value_gradient->shape.size() == 1 ? 1 : custom_value_gradient->shape[1];
    require(value_columns == groups, "Value gradient should have the same number of targets as the overall model.");
    gradient_groups = columns;
    auto constraint_text = str("monotone_constraints", "()");
    if (!constraint_text.empty() && constraint_text != "()") {
      std::replace(constraint_text.begin(), constraint_text.end(), '(', '[');
      std::replace(constraint_text.begin(), constraint_text.end(), ')', ']');
      auto constraints = parse_json(constraint_text).get<std::vector<int>>();
      require(std::all_of(constraints.begin(), constraints.end(), [](int v) { return v == 0; }),
              "Monotonic constraints are not supported with reduced gradients.");
    }
  }
  if (!is_linear()) {
    validate_tree_updaters();
    require(!data.is_quantile || (!is_tree_update() && !has_refresh()),
            "Refreshing or updating existing trees is not implemented for QuantileDMatrix.");
  }
  if (boolean("seed_per_iteration", false)) rng_.seed(random_seed() * 127u + rounds);
  if (!is_linear() && !is_tree_update()) {
    auto quantize_time = data.quantize_seconds;
    auto quantize_count = data.quantize_commands;
    data.quantize(integer("max_bin", 256));
    gpu_seconds["quantiles"] += data.quantize_seconds - quantize_time;
    command_count += data.quantize_commands - quantize_count;
  }
  Buffer prediction;
  bool weighted = !is_linear() && (!tree_weights_.empty() || dropout_parameters().active());
  auto budget = is_linear() ? static_cast<std::uint64_t>(data.rows)
                            : static_cast<std::uint64_t>(static_cast<float>(data.rows) * number("subsample", 1));
  bool sampling = !is_tree_update() && budget < data.rows;
  bool fuse_gradient_root = !is_linear() && !custom_gradient && !is_tree_update() &&
                            !has_refresh() && !sampling && objective() < 16 &&
                            integer("metal_gradient_root_fusion", 1);
  // Ordinary DMatrix feature storage is immutable. QuantileDMatrix can decode
  // changed feature types into the same storage, so retain its eager boundary.
  bool defer_prediction = fuse_gradient_root && !data.is_quantile && !vector &&
                          integer("num_parallel_tree", 1) == 1 && objective() <= 10 &&
                          !weighted && data.get_info("base_margin").empty() &&
                          tree_updaters_.size() == 1 && integer("metal_defer_prediction", 1);
  auto async_depth = integer("metal_async_rounds", 0);
  require(async_depth <= 4, "metal_async_rounds must be in [0, 4].");
  // Asynchronous rounds extend deferred updates: the next margin update reads
  // a GPU-emitted Node table, so an eligible GPU-driven round returns before
  // its command completes. Other updates first retire committed rounds.
  bool async = defer_prediction && async_depth && groups == 1 &&
               number("colsample_bylevel", 1) == 1 && number("colsample_bynode", 1) == 1;
  if (!async) drain();
  // Retirement during this call must not report an older tree's depth.
  async_tuning_round_ = UINT32_MAX;
  if (!defer_prediction) flush_prediction_caches();
  active_tuning["prediction_cache_deferred"] = 0;
  active_tuning["prediction_cache_consumed"] = 0;
  bool consumed_prediction = false;
  struct PendingUse {
    PredictionCache* cache{};
    ~PendingUse() { if (cache && cache->deferred) cache->deferred->output = {}; }
    void publish() {
      if (!cache) return;
      cache->margins = cache->deferred->output;
      cache->rounds = cache->deferred->rounds;
      cache->deferred.reset();
      cache = nullptr;
    }
  } pending;
  // Custom objectives obtain their masked training margins through predict().
  // Do not draw another dropout mask while accepting their supplied gradients.
  if (!custom_gradient || !weighted) {
    auto& cached = prediction_cache(data);
    if (defer_prediction && cached.deferred && cached.deferred->rounds == rounds) {
      pending.cache = &cached;
    } else if (!weighted && data.get_info("base_margin").empty() && cached.rounds == rounds && cached.rounds && cached.margins)
      prediction = cached.margins;
    else prediction = predict(data, {{"type", 1}, {"training", custom_gradient == nullptr}}).values;
  }
  GradientParams p{data.rows, gradient_groups, data.label_cols, objective(), static_cast<u32>(!data.get_info("weight").empty()),
                   random_seed(), rounds, static_cast<u32>(custom_gradient != nullptr), number("scale_pos_weight", 1), 1, 0, 0};
  p.parameter = objective() == 8 ? number("tweedie_variance_power", 1.5f) : number("huber_slope", 1);
  auto custom_pairs = [&](Array const& input, Array const* hessian, u32 columns) {
    auto count = checked_product(data.rows, columns);
    require(hessian && hessian->size() == input.size() && input.size() == count,
            "Gradient and Hessian must have shape (n_samples, n_targets). ");
    Buffer pairs(checked_product(count, sizeof(Pair)));
    auto g = input.floats(), h = hessian->floats();
    bool signed_hessian = !is_linear();
    for (std::size_t i = 0; i < g.size(); ++i) {
      require(std::isfinite(g[i]) && std::isfinite(h[i]), "Custom gradients and Hessians must be finite.");
      pairs.as<Pair>()[i] = h[i] >= 0 || signed_hessian ? Pair{g[i], h[i]} : Pair{0, 0};
    }
    return pairs;
  };
  // Round scratch is fully overwritten by its producing kernels before any
  // read, and the host never touches it, so it can persist across rounds.
  bool round_buffers = integer("metal_round_buffers", 1) != 0;
  auto scratch = [&](Buffer& slot, std::size_t bytes) {
    if (!round_buffers) return Buffer(bytes);
    if (!slot || slot.size() < bytes) slot = Buffer(bytes);
    return slot.prefix(bytes);
  };
  auto gradient_bytes = checked_product(checked_product(data.rows, groups), sizeof(Pair));
  Buffer gradient = custom_gradient ? custom_pairs(*custom_gradient, custom_hessian, gradient_groups) :
                    !is_linear() && objective() <= 10 ? scratch(round_buffers_.gradient, gradient_bytes) :
                    Buffer(gradient_bytes);
  Buffer value_gradient;
  if (custom_value_gradient) value_gradient = custom_pairs(*custom_value_gradient, custom_value_hessian, groups);
  // Keep stateful Cox/ranking, caller gradients and updater/sampling paths on their
  // existing synchronization boundary. Ordinary gradient kernels only produce
  // scratch buffers, so the first root can consume them before one shared wait.
  bool coalesce_prologue = fuse_gradient_root && integer("metal_coalesce_encoders", 0);
  auto owned_command = std::make_unique<Command>(fuse_gradient_root ? "gradients and root split" : "gradients",
                                                 coalesce_prologue, fuse_gradient_root && integer("metal_compute_fill", 0));
  auto& command = *owned_command;
  if (pending.cache) {
    prediction = encode_deferred_prediction(*pending.cache, command);
    consumed_prediction = true;
  }
  if (!custom_gradient) {
    if (objective() == 11 || objective() == 12) smooth_gradients(command, data, prediction, gradient, p);
    else if (objective() == 13 || objective() == 14) distribution_gradients(command, data, prediction, gradient, p);
    else if (objective() == 15 || objective() == 16) survival_gradients(command, data, prediction, gradient, p);
    else if (objective() >= 17) ranking_gradients(command, data, prediction, gradient, p, iteration);
    else command.dispatch("gradients", {prediction, data.labels, data.weights, gradient}, p, {div_up(data.rows, 256), groups, 1});
  }
  if (is_linear()) {
    record("gradients", command.wait());
    update_linear(data, gradient);
    ++rounds;
    return;
  }
  Buffer scales, quantized, value_scales, quantized_values;
  if (!is_tree_update() && !sampling) {
    auto quantize = [&](Buffer const& source, u32 columns, Buffer& scale, Buffer& output, bool values) {
      Buffer partial = scratch(values ? round_buffers_.value_partial : round_buffers_.partial,
                               checked_product(checked_product(div_up(data.rows, kGradientRowsPerTile), columns),
                                               kGradientPartialWords * sizeof(u32)));
      scale = scratch(values ? round_buffers_.value_scale : round_buffers_.scale,
                      checked_product(columns, sizeof(Pair) + sizeof(GradCert)));
      output = scratch(values ? round_buffers_.value_quantized : round_buffers_.quantized,
                       checked_product(checked_product(data.rows, columns), 2 * sizeof(std::int64_t)));
      auto qp = p; qp.groups = columns;
      command.dispatch("gradient_partial_sums", {source, partial}, qp, {div_up(data.rows, kGradientRowsPerTile), columns, 1});
      command.dispatch("gradient_scales", {partial, scale}, qp, {2, columns, 1});
      command.dispatch("quantize_gradients", {source, scale, output}, qp, {div_up(data.rows, 256), columns, 1});
    };
    quantize(gradient, gradient_groups, scales, quantized, false);
    if (value_gradient && vector) quantize(value_gradient, groups, value_scales, quantized_values, true);
  }
  if (!fuse_gradient_root) record("gradients", command.wait());
  // Start the prologue while the host builds and encodes the tree. One wait
  // still covers both command buffers. Explicitly coalesced prologues keep
  // their single encoder; asynchronous rounds already overlap host work with
  // the previous round, so a second command buffer would only add overhead.
  bool commit_early = fuse_gradient_root && !coalesce_prologue && !async &&
                      integer("metal_commit_early", 1) != 0;
  if (commit_early) command.flush();
  auto parallel = integer("num_parallel_tree", 1);
  auto eta = number("learning_rate", 0.3f) / parallel;
  auto tree_groups = vector ? 1u : groups;
  auto total_trees = static_cast<std::size_t>(tree_groups) * parallel;
  bool batch_trees = total_trees > 1 && !sampling && !is_tree_update() &&
                     !(value_gradient && vector) && tree_updaters_.size() == 1 && !async &&
                     number("colsample_bylevel", 1) == 1 && number("colsample_bynode", 1) == 1 &&
                     integer("metal_batch_trees", 1) != 0;
  std::optional<Command> batch_owned;
  if (batch_trees && !fuse_gradient_root)
    batch_owned.emplace("GPU tree batch", integer("metal_coalesce_encoders", 0), integer("metal_compute_fill", 0));
  Command* batch_cmd = batch_trees ? (fuse_gradient_root ? &command : &*batch_owned) : nullptr;
  std::vector<GpuTreeRound> batch_rounds(batch_trees ? total_trees : 0);
  GpuTreeRound async_tree;
  std::size_t k = 0;
  for (u32 group = 0; group < tree_groups; ++group) for (u32 t = 0; t < parallel; ++t, ++k) {
    if (batch_trees) {
      batch_rounds[k].emit_nodes = false;
      batch_rounds[k].batch_index = static_cast<u32>(k);
      batch_rounds[k].root_is_batch_only = !fuse_gradient_root;
      if (k > 0 && commit_early) batch_cmd->flush();
      auto tree = grow(data, quantized, scales, group, eta, false,
                       vector ? gradient_groups : 1, vector ? groups : 1,
                       batch_cmd, &batch_rounds[k]);
      if (batch_rounds[k]) continue;
      batch_trees = false;
      pending.publish();
      tree.group = group; tree.round = rounds;
      trees.push_back(std::move(tree));
      continue;
    }
    SampledGradients sampled;
    if (sampling) sampled = sample_gradients(gradient, data.rows, group, static_cast<u32>(budget), t == 0,
                                             vector ? gradient_groups : 1, gradient_groups);
    bool root = fuse_gradient_root && group == 0 && t == 0;
    auto tree = is_tree_update() ? std::move(trees_to_update_[trees.size()])
                                : grow(data, sampling ? sampled.quantized : quantized,
                                       sampling ? sampled.scales : scales, group, eta, sampling,
                                       vector ? gradient_groups : 1, vector ? groups : 1,
                                       root ? &command : nullptr, root && async ? &async_tree : nullptr);
    // An asynchronous GPU tree is still being built; eligibility implies a
    // single tree per round. Its cache version is published after commit.
    if (async_tree) break;
    // grow() has completed both the GPU wait and exact host materialization.
    // No cache version was advanced if either stage threw.
    pending.publish();
    tree.group = group; tree.round = rounds;
    if (value_gradient && vector) {
      if (sampling) {
        auto replay = sampled.params; replay.groups = replay.padding = groups; replay.group = 0;
        auto values = sample_gradients(value_gradient, replay, false, true);
        update_vector_leaves(tree, data, values.quantized, values.scales, eta);
      } else update_vector_leaves(tree, data, quantized_values, value_scales, eta);
    }
    // The pinned scalar hist updater samples in place only when it receives
    // one tree. Parallel forests copy their gradients before each sampling pass.
    bool sampled_refresh = sampling && parallel == 1;
    for (auto const& updater : tree_updaters_)
      if (updater != "grow_metal_hist")
        modify_tree(tree, data, sampled_refresh ? sampled.values : gradient, eta, updater, sampled_refresh);
    trees.push_back(std::move(tree));
  }
  if (batch_trees) {
    record(fuse_gradient_root ? "gradient_tree" : "gpu_tree", batch_cmd->wait());
    pending.publish();
    for (std::size_t idx = 0; idx < total_trees; ++idx) {
      auto const& round = batch_rounds[idx];
      if (round.unit_mismatch) active_tuning["histogram_unit_hessian"] = !round.unit_mismatch.as<u32>()[0];
      if (round.screen_counters) {
        auto const* values = round.screen_counters.as<u32>();
        split_screening.candidates += values[0]; split_screening.certified += values[1];
        split_screening.uncertain += values[2]; split_screening.exact += values[3];
        split_screening.groups += values[4]; split_screening.exact_groups += values[5];
      }
      u32 evaluated_levels = 0;
      auto tree = replay_gpu_tree(round, evaluated_levels);
      active_tuning["gpu_tree_levels"] = evaluated_levels;
      trees.push_back(std::move(tree));
    }
  }
  auto margin_bytes = static_cast<std::size_t>(data.rows) * groups * sizeof(float);
  if (async_tree) {
    auto next_prediction = round_buffers ? pooled_buffer(margin_pool_, margin_bytes) : Buffer(margin_bytes);
    command.commit();
    inflight_.push_back(InFlightRound{std::move(owned_command), async_tree});
    // The committed command now produces this cache version.
    pending.publish();
    ++rounds;
    active_tuning["prediction_cache_deferred"] = 1;
    active_tuning["prediction_cache_consumed"] = consumed_prediction;
    active_tuning["async_rounds"] = async_depth;
    active_tuning["batch_trees"] = 0;
    active_tuning["round_buffers"] = round_buffers;
    active_tuning["commit_early"] = commit_early;
    // Same tree range and floating-point additions as the forest recipe
    // below, reading the emitted single-tree table instead. Unused inputs
    // (base scores, categories, tree and leaf weights) bind a valid buffer.
    PredictParams pp{data.rows, data.cols, groups, 1, rounds - 1, rounds, objective(), 1, 1, groups, 0, 0};
    auto input = data.is_sparse() ? data.sparse_->csr_row_ptr : data.partition_data();
    auto bounds = data.partition_category_bounds();
    pp.bin_partition = static_cast<u32>(static_cast<bool>(bounds));
    auto const& unused = async_tree.meta;
    prediction_cache(data).deferred = DeferredPrediction{{input, async_tree.nodes, async_tree.meta, unused, prediction,
                                                          unused, unused, unused, bounds ? bounds : unused},
                                                         next_prediction, pp, rounds,
                                                         data.is_sparse() ? data.sparse_->csr_col_idx : Buffer{},
                                                         data.is_sparse() ? data.partition_data() : Buffer{}};
    async_tuning_round_ = async_tree.round;
    // Retire completed rounds in order, then bound the rounds in flight.
    while (!inflight_.empty() && (inflight_.size() > async_depth || inflight_.front().command->completed()))
      retire_round();
    return;
  }
  // Retirement of committed asynchronous rounds may precede this round's
  // trees; they occupy earlier rounds, so scan the same-round suffix now.
  auto first_new_tree = trees.size();
  while (first_new_tree && trees[first_new_tree - 1].round == rounds) --first_new_tree;
  normalize_trees(static_cast<std::size_t>(tree_groups) * parallel);
  ++rounds;
  // TreeBuilder replaces the tuning map. Publish these per-update selections
  // after tree construction so its workspace/profile reset cannot erase them.
  active_tuning["prediction_cache_deferred"] = 0;
  active_tuning["prediction_cache_consumed"] = consumed_prediction;
  active_tuning["async_rounds"] = 0;
  active_tuning["batch_trees"] = batch_trees;
  active_tuning["round_buffers"] = round_buffers;
  active_tuning["commit_early"] = commit_early;
  upload_model();
  if (tree_weights_.empty()) {
    Buffer next_prediction = round_buffers ? pooled_buffer(margin_pool_, margin_bytes) : Buffer(margin_bytes);
    PredictParams pp{data.rows, data.cols, groups, static_cast<u32>(trees.size()), rounds - 1, rounds, objective(), 1, 1, groups, 0, 0};
    pp.tree_begin = static_cast<u32>(first_new_tree);
    bool unpartitioned = is_tree_update() || has_refresh();
    auto bounds = unpartitioned ? Buffer{} : data.partition_category_bounds();
    pp.bin_partition = static_cast<u32>(static_cast<bool>(bounds));
    auto& cached = prediction_cache(data);
    // Prediction with a base margin resets the prefix version before growing.
    auto version = data.get_info("base_margin").empty() ? rounds : 1;
    if (data.is_sparse()) {
      auto csr_vals = unpartitioned ? data.sparse_->csr_val : data.partition_data();
      if (defer_prediction && active_tuning["gpu_tree"]) {
        cached.deferred = DeferredPrediction{{data.sparse_->csr_row_ptr, forest_nodes_, forest_trees_, base_buffer_, prediction,
                                              forest_categories_, forest_weights_, forest_leaf_weights_,
                                              bounds ? bounds : base_buffer_}, next_prediction, pp, version,
                                             data.sparse_->csr_col_idx, csr_vals};
        active_tuning["prediction_cache_deferred"] = 1;
      } else {
        Command cache("update training predictions");
        cache.dispatch("predict_csr", {data.sparse_->csr_row_ptr, data.sparse_->csr_col_idx, csr_vals,
                                       forest_nodes_, forest_trees_, base_buffer_, prediction, next_prediction,
                                       forest_categories_, forest_weights_, forest_leaf_weights_, bounds ? bounds : base_buffer_},
                       pp, {div_up(data.rows, 256), groups, 1});
        record("prediction_cache", cache.wait());
        cached.margins = next_prediction;
        cached.rounds = version;
      }
    } else {
      auto input = unpartitioned ? data.data : data.partition_data();
      if (defer_prediction && active_tuning["gpu_tree"]) {
        cached.deferred = DeferredPrediction{{input, forest_nodes_, forest_trees_, base_buffer_, prediction,
                                              forest_categories_, forest_weights_, forest_leaf_weights_,
                                              bounds ? bounds : base_buffer_}, next_prediction, pp, version};
        active_tuning["prediction_cache_deferred"] = 1;
      } else {
        Command cache("update training predictions");
        cache.dispatch("predict", {input, forest_nodes_, forest_trees_, base_buffer_, prediction, next_prediction, forest_categories_, forest_weights_, forest_leaf_weights_, bounds ? bounds : base_buffer_}, pp, {div_up(data.rows, 256), groups, 1});
        record("prediction_cache", cache.wait());
        cached.margins = next_prediction;
        cached.rounds = version;
      }
    }
  }
  if (objective() >= 17) {
    active_tuning["ranking_cache"] = integer("metal_ranking_cache", 1);
    active_tuning["ranking_aggregation"] = integer("metal_ranking_aggregation", 1);
  }
  if (sampling) {
    active_tuning["sampling_method"] = str("sampling_method", "uniform") == "gradient_based" ? 2 : 1;
    active_tuning["sampling_budget"] = static_cast<u32>(budget);
  }
}
void Learner::reset(bool clear_dropout) {
  drain();
  if (clear_dropout) {
    dropped_trees_.clear();
    trees_to_update_.clear(); update_original_weights_ = false;
    linear_mass_ = {}; linear_previous_ = {}; linear_order_.clear();
    linear_converged_ = false; linear_configuration_dirty_ = true;
    if (is_linear()) clear_linear_parameters();
    else { clear_tree_updater(); params["process_type"] = "default"; }
    ranking_objective_cache_.reset(); ranking_metric_caches_.clear();
    ranking_bias_work_initialized_ = false;
  }
  tree_workspace_.clear(); tree_workspace_inputs_.clear();
  round_buffers_ = {}; margin_pool_.clear(); round_output_pool_.clear();
  refresh_workspace_.clear();
  sampling_workspace_.clear();
  refresh_source_ = {}; refresh_matrix_.reset();
  linear_model_ = {}; linear_workspace_.clear(); linear_matrix_.reset(); base_buffer_ = {};
  prediction_caches_.clear();
  validated_matrix_.reset();
  survival_orders_.clear();
  forest_nodes_ = {}; forest_trees_ = {}; forest_categories_ = {}; forest_weights_ = {}; forest_leaf_weights_ = {}; forest_size_ = std::numeric_limits<std::size_t>::max();
  forest_node_count_ = 0; forest_category_count_ = 0; forest_leaf_count_ = 0;
  shap_paths_ = {}; shap_leaves_ = {}; shap_categories_ = {}; node_means_ = {}; shap_size_ = std::numeric_limits<std::size_t>::max();
  shap_path_count_ = 0; shap_category_count_ = 0; shap_mean_count_ = 0; shap_leaf_count_ = 0;
}
std::string Learner::default_metric() const {
  auto obj = objective();
  if (obj >= 17) {
    auto p = ranking_parameters(0);
    return std::string(obj == 18 ? "map" : "ndcg") + (p.mean ? "" : "@" + std::to_string(p.num_pairs));
  }
  return (obj == 1 && str("objective") != "reg:logistic") || obj == 2 ? "logloss" :
         obj == 3 || obj == 4 ? "mlogloss" : obj == 6 ? "rmsle" : obj == 9 ? "mphe" :
         obj == 5 ? "poisson-nloglik" : obj == 7 ? "gamma-deviance" : obj == 10 ? "error" :
         obj == 8 ? "tweedie-nloglik@" + str("tweedie_variance_power", "1.5") :
         obj == 11 ? "mae" : obj == 12 ? "quantile" : obj == 13 ? "expectile" :
         obj == 14 ? "normal-nloglik" : obj == 15 ? "aft-nloglik" : obj == 16 ? "cox-nloglik" : "rmse";
}
std::string Learner::evaluate(int iteration, std::span<std::shared_ptr<Matrix> const> datasets,
                              std::span<std::string const> names) {
  // Evaluation without data neither reads the model nor waits for rounds.
  if (!datasets.empty()) drain();
  configure();
  require(initialized, "Model is not yet initialized (not fitted).");
  if (metrics.empty() && !integer("disable_default_eval_metric", 0)) {
    auto name = default_metric();
    if (objective() >= 17) ranking_metric_configs_[name] = ranking_metric_config(name)["lambdarank_param"];
    metrics.push_back(std::move(name));
  }
  std::ostringstream output; output << '[' << iteration << ']' << std::setprecision(10);
  for (std::size_t i = 0; i < datasets.size(); ++i) {
    auto const& data = *datasets[i];
    auto prediction = predict(data, {{"type", 0}, {"eval", true}});
    for (auto const& name : metrics) {
      if (name != "aft-nloglik" && name != "interval-regression-accuracy")
        require(data.get_info("label").size() == static_cast<std::size_t>(data.rows) * data.label_cols,
                "Evaluation data requires labels.");
      output << '\t' << names[i] << '-' << name << ':' << metric(data, prediction.values, name);
    }
  }
  return output.str();
}
}  // namespace metalgb
