#include "metalgb/engine.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <iomanip>
#include <sstream>

namespace metalgb {
namespace {
template <class T> Buffer upload(std::vector<T> const& values) {
  return Buffer(values.data(), checked_product(values.size(), sizeof(T)));
}
Json defaults() {
  return {{"lambdarank_pair_method", "topk"}, {"lambdarank_num_pair_per_sample", "4294967295"},
          {"lambdarank_unbiased", "0"}, {"lambdarank_normalization", "1"},
          {"lambdarank_score_normalization", "1"}, {"lambdarank_bias_norm", "1"}, {"ndcg_exp_gain", "1"}};
}
Json cache_parameters(Json const& parameters) {
  auto flag = [&](char const* name) {
    auto value = parameters.at(name).get<std::string>();
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    require(value == "0" || value == "1" || value == "false" || value == "true", "Invalid ranking boolean parameter.");
    return value == "1" || value == "true";
  };
  return {{"method", parameters.at("lambdarank_pair_method")},
          {"pairs", std::stoull(parameters.at("lambdarank_num_pair_per_sample").get<std::string>())},
          {"unbiased", flag("lambdarank_unbiased")}, {"normalization", flag("lambdarank_normalization")},
          {"score_normalization", flag("lambdarank_score_normalization")},
          {"bias_norm", std::stod(parameters.at("lambdarank_bias_norm").get<std::string>())},
          {"exp_gain", flag("ndcg_exp_gain")}};
}
Json normalized_parameters(Json parameters) {
  auto values = cache_parameters(parameters);
  parameters["lambdarank_num_pair_per_sample"] = std::to_string(values.at("pairs").get<u64>());
  for (auto name : {"unbiased", "normalization", "score_normalization"})
    parameters[std::string("lambdarank_") + name] = values.at(name).get<bool>() ? "1" : "0";
  parameters["ndcg_exp_gain"] = values.at("exp_gain").get<bool>() ? "1" : "0";
  std::ostringstream bias;
  bias << std::setprecision(std::numeric_limits<double>::max_digits10) << values.at("bias_norm").get<double>();
  parameters["lambdarank_bias_norm"] = bias.str();
  return parameters;
}
Buffer reduce(Command& command, Buffer partial, u32 count) {
  while (count > 1) {
    SurvivalParams p{count, 0, 0, 0, 1, 1, 0, 0};
    count = div_up(count, 256);
    Buffer next(checked_product(count, kSurvivalSumBytes));
    command.dispatch("survival_reduce", {partial, next}, p, {count, 1, 1});
    partial = std::move(next);
  }
  return partial;
}
std::vector<float> save_bias(Buffer const& buffer) {
  std::vector<float> result(buffer.size() / kRankingRealBytes);
  auto values = buffer.as<MetricRatio>();
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<float>(std::ldexp(static_cast<double>(values[i].high) + values[i].low, values[i].exponent));
  return result;
}
Buffer load_bias(Json const& input) {
  auto values = input.get<std::vector<float>>();
  std::vector<MetricRatio> packed(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(std::isfinite(values[i]) && values[i] >= 0, "Invalid saved ranking position bias.");
    int exponent;
    auto high = std::frexp(values[i], &exponent);
    packed[i] = {high, 0, exponent, 0};
  }
  return upload(packed);
}
}

RankingParams Learner::ranking_parameters(u32 rows) const {
  RankingParams p{};
  p.rows = rows;
  p.objective = objective();
  auto method = str("lambdarank_pair_method", "topk");
  require(method == "topk" || method == "mean", "lambdarank_pair_method must be topk or mean.");
  p.mean = method == "mean";
  auto count = integer("lambdarank_num_pair_per_sample", UINT32_MAX);
  p.num_pairs = count == UINT32_MAX ? (p.mean ? 1 : 32) : count;
  require(p.num_pairs > 0, "lambdarank_num_pair_per_sample must be at least 1.");
  p.normalization = boolean("lambdarank_normalization", true);
  p.score_normalization = boolean("lambdarank_score_normalization", true);
  p.exp_gain = boolean("ndcg_exp_gain", true);
  p.unbiased = boolean("lambdarank_unbiased", false);
  auto text = str("lambdarank_bias_norm", "1");
  std::size_t read;
  auto bias_norm = std::stod(text, &read);
  require(read == text.size() && std::isfinite(bias_norm) && bias_norm >= 0, "lambdarank_bias_norm must be nonnegative.");
  int exponent;
  auto significand = std::frexp(1.0 / (1.0 + bias_norm), &exponent);
  p.regularizer_high = static_cast<float>(significand);
  p.regularizer_low = static_cast<float>(significand - p.regularizer_high);
  p.regularizer_exponent = exponent;
  p.subsample = 1;
  p.iteration = rounds;
  return p;
}

Json Learner::ranking_config() const {
  auto p = ranking_parameters(0);
  auto parameters = defaults();
  for (auto& [key, value] : parameters.items()) value = str(key, value.get<std::string>());
  parameters["lambdarank_unbiased"] = p.unbiased ? "1" : "0";
  parameters["lambdarank_normalization"] = p.normalization ? "1" : "0";
  parameters["lambdarank_score_normalization"] = p.score_normalization ? "1" : "0";
  parameters["ndcg_exp_gain"] = p.exp_gain ? "1" : "0";
  Json result{{"name", str("objective")}, {"lambdarank_param", normalized_parameters(std::move(parameters))}};
  if (p.unbiased) {
    result["ti+"] = save_bias(ranking_bias_plus_);
    result["tj-"] = save_bias(ranking_bias_minus_);
  }
  return result;
}
void Learner::load_ranking_state(Json const& objective, bool new_objective) {
  if (new_objective) {
    ranking_bias_plus_ = {}; ranking_bias_minus_ = {};
    ranking_bias_work_initialized_ = false;
  }
  if (objective.contains("ti+") && objective.contains("tj-")) {
    ranking_bias_plus_ = load_bias(objective["ti+"]);
    ranking_bias_minus_ = load_bias(objective["tj-"]);
    require(ranking_bias_plus_.size() == ranking_bias_minus_.size(), "Inconsistent ranking position bias sizes.");
  }
}
void Learner::prepare_objective(std::string_view name) {
  if (name == str("objective", "reg:squarederror")) return;
  // A replacement objective receives only this call's arguments. Its ranking
  // parameters and bias workspace do not survive a switch away and back.
  auto parameters = defaults();
  for (auto const& [key, value] : parameters.items()) params.erase(key);
  ranking_objective_cache_.reset();
  ranking_bias_plus_ = {}; ranking_bias_minus_ = {}; ranking_bias_work_initialized_ = false;
}
Json Learner::ranking_metric_config(std::string const& name) const {
  auto loaded = ranking_metric_configs_.find(name);
  Json p = loaded == ranking_metric_configs_.end() ? defaults() : loaded->second;
  auto separator = name.find('@');
  if (loaded == ranking_metric_configs_.end() && separator != std::string::npos) {
    auto tail = std::string_view(name).substr(separator + 1);
    if (tail.ends_with('-')) tail.remove_suffix(1);
    u32 count = UINT32_MAX;
    if (!tail.empty()) {
      auto parsed = std::from_chars(tail.data(), tail.data() + tail.size(), count);
      require(parsed.ec == std::errc{} && parsed.ptr == tail.data() + tail.size() && count > 0,
              "Invalid ranking metric cutoff: " + name);
    }
    p["lambdarank_num_pair_per_sample"] = std::to_string(count);
  }
  // NDCG and pairwise objectives pass their entire configuration to their
  // automatic metric. Explicit metrics, and MAP's default, use their own cutoff.
  if (loaded == ranking_metric_configs_.end() && metrics.empty() &&
      (objective() == 17 || objective() == 19) && name == default_metric())
    p = ranking_config()["lambdarank_param"];
  return {{"name", name}, {"lambdarank_param", normalized_parameters(std::move(p))}};
}

Learner::RankingCache Learner::make_ranking_cache(Matrix const& data, Json parameters, u32 label_validation) {
  RankingCache cache{};
  cache.matrix = data.weak_from_this(); cache.thread = std::this_thread::get_id();
  cache.parameters = std::move(parameters);
  cache.offsets = data.group_ptr.empty() ? std::vector<u32>{0, data.rows} : data.group_ptr;
  require(cache.offsets.size() >= 2 && cache.offsets.back() == data.rows,
          "Query group sizes must sum to the number of rows.");
  auto queries = static_cast<u32>(cache.offsets.size() - 1);
  ranking_weights(data, cache);
  auto const& labels = data.get_info("label");
  if (label_validation == 1 && cache.parameters.at("exp_gain").get<bool>())
    require(std::all_of(labels.begin(), labels.end(), [](float y) {
      return y >= 0 && std::abs(std::floor(y) - y) <= 1.0e-6f && y < 32;
    }), "NDCG relevance labels must be nonnegative integers no greater than 31 when ndcg_exp_gain is true.");
  if (label_validation == 2)
    require(std::all_of(labels.begin(), labels.end(), [](float y) {
      return std::abs(y) < 1.0e-6f || std::abs(y - 1) < 1.0e-6f;
    }), "MAP and precision can only be used with binary labels.");
  Command command("ranking cache");
  cache.query_ptr = upload(cache.offsets);
  std::vector<u32> row_query(data.rows);
  std::vector<RankingTile> tiles;
  cache.tile_offsets = {0};
  for (u32 g = 0; g < queries; ++g) {
    auto begin = cache.offsets[g], end = cache.offsets[g + 1];
    cache.max_size = std::max(cache.max_size, end - begin);
    std::fill(row_query.begin() + begin, row_query.begin() + end, g);
    for (std::uint64_t row = begin; row < end; row += 256)
      tiles.push_back({row, static_cast<u32>(std::min<std::uint64_t>(256, end - row)), g});
    cache.tile_offsets.push_back(tiles.size());
  }
  cache.row_query = upload(row_query); cache.row_tiles = upload(tiles); cache.tiles = tiles.size();
  cache.tile_ptr = upload(cache.tile_offsets);
  cache.label_order = ranking_sort(command, cache, data.labels, data.labels, 0, "label");
  RankingParams p{}; p.rows = data.rows; p.queries = queries; p.max_size = cache.max_size;
  cache.discount = Buffer(checked_product(cache.max_size, kRankingRealBytes));
  // Dynamic objective branches still require valid Metal buffer bindings.
  cache.inverse_idcg = Buffer(0);
  command.dispatch("ranking_discounts", {cache.discount}, p, {div_up(cache.max_size, 256), 1, 1});
  Buffer weight_inputs(checked_product(queries, kSurvivalSumBytes));
  command.dispatch("ranking_weight_inputs", {cache.query_weights, weight_inputs}, p, {div_up(queries, 256), 1, 1});
  auto total = reduce(command, weight_inputs, queries);
  cache.weight_norm = Buffer(sizeof(float));
  command.dispatch("ranking_weight_norm", {total, cache.weight_norm}, p, {1, 1, 1}, {1, 1, 1});
  if (label_validation == 1) {
    p.mean = cache.parameters.at("method") == "mean";
    p.num_pairs = cache.parameters.at("pairs").get<u32>();
    if (p.num_pairs == UINT32_MAX) p.num_pairs = p.mean ? 1 : 32;
    p.exp_gain = cache.parameters.at("exp_gain").get<bool>();
    ranking_idcg(command, data, cache, p);
  }
  // Publish only buffers whose setup completed. Later validation can reject a
  // gradient request without making an unsubmitted cache visible to its retry.
  record("ranking_cache", command.wait());
  return cache;
}
Learner::RankingCache& Learner::ranking_cache(Matrix const& data, Json parameters,
                                             std::string const& metric, u32 label_validation) {
  if (metric.empty()) {
    if (!ranking_objective_cache_ || ranking_objective_cache_->matrix.lock().get() != &data ||
        ranking_objective_cache_->parameters != parameters)
      ranking_objective_cache_ = std::make_unique<RankingCache>(
          make_ranking_cache(data, std::move(parameters), label_validation));
    return *ranking_objective_cache_;
  }
  auto& entries = ranking_metric_caches_[metric];
  std::erase_if(entries, [](auto const& entry) { return entry.matrix.expired(); });
  // Metrics use DMatrixCache's 32-entry default, independently per metric.
  if (entries.size() >= 32) entries.erase(entries.begin(), entries.end() - 15);
  auto thread = std::this_thread::get_id();
  for (auto& entry : entries) if (entry.thread == thread && entry.matrix.lock().get() == &data) {
    if (entry.parameters != parameters)
      entry = make_ranking_cache(data, std::move(parameters), label_validation);
    return entry;
  }
  entries.push_back(make_ranking_cache(data, std::move(parameters), label_validation));
  return entries.back();
}
void Learner::ranking_weights(Matrix const& data, RankingCache& cache, bool allow_extra) {
  auto queries = cache.offsets.size() - 1;
  auto const& weights = data.get_info("weight");
  require(weights.empty() || (allow_extra ? weights.size() >= queries : weights.size() == queries),
          "Ranking weights must have one entry per cached query group.");
  if (weights.empty()) {
    if (!cache.unit_weights) cache.unit_weights = upload(std::vector<float>(queries, 1));
    cache.query_weights = cache.unit_weights;
  } else cache.query_weights = data.weights;
}
Buffer Learner::ranking_sort(Command& command, RankingCache const& cache, Buffer const& primary,
                              Buffer const& secondary, u32 mode, std::string_view purpose) {
  auto rows = cache.offsets.back();
  RankingParams p{}; p.rows = rows; p.mode = mode;
  auto input = ranking_buffer(cache, std::string(purpose) + "_order_a", checked_product(rows, sizeof(u32)));
  auto output = ranking_buffer(cache, std::string(purpose) + "_order_b", input.size());
  command.dispatch("ranking_order_tiles", {primary, secondary, cache.row_tiles, input}, p, {cache.tiles, 1, 1});
  for (std::uint64_t width = 256; width < cache.max_size; width *= 2) {
    p.width = static_cast<u32>(width);
    command.dispatch("ranking_order_merge", {primary, secondary, cache.query_ptr, cache.row_query, input, output},
                     p, {div_up(rows, 256), 1, 1});
    std::swap(input, output);
  }
  return input;
}
Buffer Learner::ranking_buffer(RankingCache const& cache, std::string key, std::size_t bytes) const {
  if (!integer("metal_ranking_cache", 1)) return Buffer(bytes);
  auto& buffer = cache.workspace[std::move(key)];
  if (!buffer || buffer.size() != bytes) buffer = Buffer(bytes);
  return buffer;
}
Buffer Learner::ranking_scan(Command& command, RankingCache const& cache, Buffer const& input,
                              Buffer const& queries, u32 rows, u32 level) {
  RankingParams p{}; p.rows = rows;
  auto count = div_up(rows, 256);
  auto output = ranking_buffer(cache, "map_scan_" + std::to_string(level), checked_product(rows, kSurvivalSumBytes));
  auto totals = ranking_buffer(cache, "map_totals_" + std::to_string(level), checked_product(count, kSurvivalSumBytes));
  auto total_queries = ranking_buffer(cache, "map_queries_" + std::to_string(level), checked_product(count, sizeof(u32)));
  command.dispatch("ranking_scan_tiles", {input, queries, output, totals, total_queries}, p, {count, 1, 1});
  if (count > 1) {
    auto scanned = ranking_scan(command, cache, totals, total_queries, count, level + 1);
    command.dispatch("ranking_scan_offsets", {output, queries, scanned, total_queries}, p, {count, 1, 1});
  }
  return output;
}
void Learner::ranking_idcg(Command& command, Matrix const& data, RankingCache& cache, RankingParams const& params) {
  auto topk = params.mean ? UINT32_MAX : params.num_pairs;
  if (cache.inverse_idcg.size() && cache.idcg_topk == topk && cache.idcg_exp_gain == params.exp_gain) return;
  auto p = params; p.num_pairs = topk;
  Buffer partial(checked_product(cache.tiles, kSurvivalSumBytes));
  cache.inverse_idcg = Buffer(checked_product(p.queries, kRankingRealBytes));
  command.dispatch("ranking_idcg_partial", {data.labels, cache.label_order, cache.query_ptr, cache.row_tiles,
                                            cache.discount, partial}, p, {cache.tiles, 1, 1});
  command.dispatch("ranking_idcg_finish", {partial, cache.tile_ptr, cache.inverse_idcg}, p,
                   {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  cache.idcg_topk = topk; cache.idcg_exp_gain = params.exp_gain;
}

void Learner::ranking_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                                 Buffer const& gradient, GradientParams const& params, int iteration) {
  auto p = ranking_parameters(data.rows);
  auto parameters = defaults();
  for (auto& [name, value] : parameters.items()) value = str(name, value.get<std::string>());
  auto key = cache_parameters(parameters); key["objective"] = p.objective;
  auto& cache = ranking_cache(data, std::move(key), "", p.objective == 17 ? 1 : p.objective == 18 ? 2 : 0);
  ranking_weights(data, cache);
  p.queries = cache.offsets.size() - 1;
  p.positions = p.mean ? std::min(cache.max_size, 32u) : p.num_pairs;
  bool initialize_bias = p.unbiased && (!ranking_bias_plus_.size() || !ranking_bias_work_initialized_);
  if (initialize_bias)
    require(iteration == 0, "Unbiased ranking must start at iteration 0 after loading a model.");
  p.seed = p.mean ? rng_() : 0;
  p.mode = integer("metal_ranking_aggregation", 1);
  if (cache.pair_samples != p.num_pairs || cache.pair_method != p.mean) {
    std::vector<u64> offsets{0}; std::vector<u32> tile_offsets{0}; std::vector<RankingTile> tiles;
    auto max_buffer = parse_json(Runtime::get().info()).at("METAL_MAX_BUFFER_LENGTH").get<std::size_t>();
    for (u32 g = 0; g < p.queries; ++g) {
      u64 size = cache.offsets[g + 1] - cache.offsets[g], k = std::min<u64>(size, p.num_pairs);
      u64 count = size < 2 ? 0 : p.mean ? checked_product(size, p.num_pairs) : k * (2 * size - k - 1) / 2;
      require(count <= std::numeric_limits<std::size_t>::max() - offsets.back(), "Ranking pair count exceeds addressable storage.");
      auto begin = offsets.back(), end = begin + count;
      require(end <= max_buffer / (p.unbiased ? kRankingRealBytes : 2 * sizeof(u32)),
              "Ranking pair storage exceeds Metal maxBufferLength; reduce lambdarank_num_pair_per_sample.");
      for (u64 i = begin; i < end; i += 256)
        tiles.push_back({i, static_cast<u32>(std::min<u64>(256, end - i)), g});
      require(tiles.size() <= UINT32_MAX, "Too many ranking pair tiles.");
      offsets.push_back(end); tile_offsets.push_back(tiles.size());
    }
    cache.pair_ptr = upload(offsets); cache.pair_tile_ptr = upload(tile_offsets); cache.pair_work = upload(tiles);
    cache.pairs = offsets.back(); cache.pair_tiles = tiles.size();
    cache.pair_samples = p.num_pairs; cache.pair_method = p.mean;
  }
  p.tiles = cache.pair_tiles;
  auto order = ranking_sort(command, cache, prediction, prediction, 0);
  auto inverse = ranking_buffer(cache, "inverse", checked_product(data.rows, sizeof(u32)));
  command.dispatch("ranking_inverse", {order, cache.query_ptr, cache.row_query, inverse}, p, {div_up(data.rows, 256), 1, 1});
  if (p.objective == 17) ranking_idcg(command, data, cache, p);
  Buffer map_statistics(0);
  if (p.objective == 18) {
    auto inputs = ranking_buffer(cache, "map_inputs", checked_product(data.rows, kSurvivalSumBytes));
    command.dispatch("ranking_map_inputs", {data.labels, order, cache.query_ptr, cache.row_query, inputs}, p,
                     {div_up(data.rows, 256), 1, 1});
    map_statistics = ranking_scan(command, cache, inputs, cache.row_query, data.rows);
  }
  auto pairs = ranking_buffer(cache, "pair_indices", checked_product(cache.pairs, 2 * sizeof(u32)));
  if (p.mean) {
    auto labels = ranking_sort(command, cache, data.labels, prediction, 1, "mean_label");
    command.dispatch("ranking_mean_pairs", {data.labels, labels, cache.query_ptr, cache.pair_ptr, pairs}, p,
                     {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  }
  if (initialize_bias) {
    // The reference starts a new bias fit when its per-data workspace is empty,
    // including the first update after loading a saved model or snapshot.
    require(iteration == 0, "Unbiased ranking must start at iteration 0 after loading a model.");
    ranking_bias_plus_ = Buffer(checked_product(p.positions, kRankingRealBytes));
    ranking_bias_minus_ = Buffer(ranking_bias_plus_.size());
    command.dispatch("ranking_bias_init", {ranking_bias_plus_, ranking_bias_minus_}, p,
                     {div_up(p.positions, 256), 1, 1});
    ranking_bias_work_initialized_ = data.rows != 0;
  }
  require(!p.unbiased || ranking_bias_plus_.size() == checked_product(p.positions, kRankingRealBytes),
          "The ranking position-bias size changed; keep the pair method and position count fixed when continuing unbiased training.");
  Buffer unused(0);
  auto plus = p.unbiased ? ranking_bias_plus_ : unused;
  auto minus = p.unbiased ? ranking_bias_minus_ : unused;
  auto values = ranking_buffer(cache, "pair_gradients", checked_product(cache.pairs, sizeof(Pair)));
  auto costs = ranking_buffer(cache, "pair_costs", p.unbiased ? checked_product(cache.pairs, kRankingRealBytes) : 0);
  auto partial = ranking_buffer(cache, "pair_partial", checked_product(cache.pair_tiles, kRankingPartialBytes));
  auto statistics = ranking_buffer(cache, "statistics", checked_product(p.queries, sizeof(RankingStats)));
  command.dispatch("ranking_pair_gradients", {prediction, data.labels, order, inverse, cache.query_ptr,
      cache.pair_ptr, cache.pair_work, pairs, cache.discount, cache.inverse_idcg, map_statistics,
      plus, minus, values, costs, partial}, p, {cache.pair_tiles, 1, 1});
  command.dispatch("ranking_group_statistics", {partial, cache.pair_tile_ptr, cache.pair_ptr, statistics}, p,
                   {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  auto accumulated = ranking_buffer(cache, "accumulated", checked_product(data.rows, 4 * sizeof(u32)));
  auto position_costs = ranking_buffer(cache, "position_costs", p.unbiased ? accumulated.size() : 0);
  command.zero(accumulated); if (p.unbiased) command.zero(position_costs);
  command.dispatch("ranking_pair_accumulate", {values, costs, pairs, cache.pair_work, cache.query_ptr,
      statistics, plus, minus, accumulated, position_costs, inverse}, p, {cache.pair_tiles, 1, 1});
  auto sampling = p; sampling.seed = params.seed;
  command.dispatch("ranking_gradient_finish", {accumulated, cache.row_query, statistics, cache.query_weights,
                                                cache.weight_norm, gradient}, sampling, {div_up(data.rows, 256), 1, 1});
  if (p.unbiased && p.positions) {
    auto tiles = div_up(p.queries, 256);
    auto bias_partial = ranking_buffer(cache, "bias_partial", checked_product(checked_product(tiles, p.positions), kSurvivalSumBytes));
    auto bias_total = ranking_buffer(cache, "bias_total", checked_product(p.positions, kSurvivalSumBytes));
    command.dispatch("ranking_bias_partial", {position_costs, cache.query_ptr, statistics, bias_partial}, p,
                     {tiles, p.positions, 1});
    command.dispatch("ranking_bias_reduce", {bias_partial, bias_total}, p,
                     {div_up(p.positions, 64), 1, 1}, {64, 1, 1});
    command.dispatch("ranking_bias_update", {bias_total, ranking_bias_plus_, ranking_bias_minus_}, p,
                     {div_up(p.positions, 64), 1, 1}, {64, 1, 1});
  }
}

double Learner::ranking_metric(Matrix const& data, Buffer const& prediction, std::string const& name) {
  require(groups == 1 && data.label_cols == 1, "Ranking metrics require one label and prediction per row.");
  auto config = ranking_metric_config(name)["lambdarank_param"];
  RankingParams p{}; p.rows = data.rows;
  p.mode = name.starts_with("ndcg") ? 0 : name.starts_with("map") ? 1 : name == "auc" ? 3 : name == "aucpr" ? 4 : 2;
  p.minus = name.ends_with('-');
  p.exp_gain = config["ndcg_exp_gain"] == "1";
  p.mean = config["lambdarank_pair_method"] == "mean";
  p.num_pairs = static_cast<u32>(std::stoul(config["lambdarank_num_pair_per_sample"].get<std::string>()));
  if (p.num_pairs == UINT32_MAX) p.num_pairs = p.mean ? 1 : 32;
  if (p.mean) p.num_pairs = UINT32_MAX;
  Command command("ranking metric");
  auto const& weights = data.get_info("weight");
  require(weights.empty() || weights.size() == (data.group_ptr.empty() ? 1 : data.group_ptr.size() - 1),
          "Ranking weights must have one entry per query group.");
  auto key = p.mode >= 3 ? Json{{"generation", data.generation}} : cache_parameters(config);
  auto& cache = ranking_cache(data, std::move(key), name, p.mode == 0 ? 1 : p.mode < 3 ? 2 : 0);
  ranking_weights(data, cache, p.mode == 0);
  p.queries = cache.offsets.size() - 1;
  Buffer order, auc_counts;
  if (p.mode == 3) {
    // Count concordant and tied pairs by merging the stable label order.
    // This is O(n log n), including queries whose labels are all tied.
    order = cache.label_order;
    Buffer alternate(order.size()), scratch(order.size());
    auc_counts = Buffer(checked_product(data.rows, 2 * sizeof(u64))); command.zero(auc_counts);
    for (std::uint64_t width = 1; width < cache.max_size; width *= 2) {
      p.width = width;
      command.dispatch("ranking_auc_merge", {prediction, cache.query_ptr, cache.row_query, order, alternate, auc_counts},
                       p, {div_up(data.rows, 256), 1, 1});
      order = alternate;
      std::swap(alternate, scratch);
    }
  } else order = ranking_sort(command, cache, prediction, prediction, 0);
  if (p.mode == 0) ranking_idcg(command, data, cache, p);
  Buffer partial(checked_product(p.queries, kSurvivalSumBytes));
  if (p.mode == 3)
    command.dispatch("ranking_auc_queries", {auc_counts, cache.query_ptr, cache.query_weights, partial}, p,
                     {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  else if (p.mode == 4)
    command.dispatch("ranking_aucpr_queries", {prediction, data.labels, order, cache.query_ptr, cache.query_weights, partial}, p,
                     {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  else command.dispatch("ranking_metric_queries", {data.labels, order, cache.query_ptr, cache.query_weights,
      cache.discount, cache.inverse_idcg, partial}, p, {div_up(p.queries, 64), 1, 1}, {64, 1, 1});
  auto total = reduce(command, partial, p.queries);
  auto weight_total = total;
  if (p.mode == 0 && !weights.empty() && weights.size() != p.queries) {
    // NDCG's numerator uses cached queries, but its denominator sums all live
    // weights. The reference keeps this distinction after query-count changes.
    auto wp = p; wp.queries = weights.size();
    Buffer input(checked_product(wp.queries, kSurvivalSumBytes));
    command.dispatch("ranking_weight_inputs", {data.weights, input}, wp, {div_up(wp.queries, 256), 1, 1});
    weight_total = reduce(command, input, wp.queries);
  }
  Buffer output(sizeof(MetricRatio));
  command.dispatch("ranking_metric_finish", {total, weight_total, output}, p, {1, 1, 1}, {1, 1, 1});
  record("metrics", command.wait());
  auto value = *output.as<MetricRatio>();
  auto result = std::ldexp(static_cast<double>(value.high) + value.low, value.exponent);
  if (p.mode >= 3 && std::isnan(result)) return result;
  require(result <= 1.0 + 1.0e-6, "Invalid ranking score; check query group weights.");
  return std::min(1.0, result);
}
}  // namespace metalgb
