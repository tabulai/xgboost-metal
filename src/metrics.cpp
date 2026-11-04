#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <unordered_map>

namespace metalgb {
double Learner::metric(Matrix const& data, Buffer const& prediction, std::string const& name) {
  if (name == "aft-nloglik" || name == "interval-regression-accuracy" || name == "cox-nloglik")
    return survival_metric(data, prediction, name == "aft-nloglik" ? 0 : name == "interval-regression-accuracy" ? 1 : 2);
  static std::map<std::string, u32, std::less<>> const types{
      {"rmse", 0}, {"mae", 1}, {"rmsle", 2}, {"logloss", 3}, {"merror", 4},
      {"mlogloss", 5}, {"error", 6}, {"mape", 7}, {"mphe", 8}, {"poisson-nloglik", 9},
      {"gamma-deviance", 10}, {"tweedie-nloglik", 11}, {"gamma-nloglik", 12},
      {"auc", 13}, {"aucpr", 14}, {"quantile", 15}, {"expectile", 16}, {"normal-nloglik", 17}};
  auto separator = name.find('@');
  auto base = name.substr(0, separator);
  if ((base == "auc" || base == "aucpr") && !data.group_ptr.empty()) return ranking_metric(data, prediction, name);
  if (base == "ndcg" || base == "ndcg-" || base == "map" || base == "map-" || base == "pre" || base == "pre-")
    return ranking_metric(data, prediction, name);
  auto it = types.find(base);
  require(it != types.end(), "Metric has not been implemented yet: " + name);
  require(data.get_info("weight").empty() || data.get_info("weight").size() == data.rows,
          "Evaluation weights must have one entry per row.");
  MetricParams p{data.rows, groups, data.label_cols, it->second,
                 static_cast<u32>(div_up(data.rows, 256) * 256), static_cast<u32>(num_class > 0), 0, 0,
                 number("huber_slope", 1), 0, 0, 0};
  if (base == "error") p.parameter = separator == std::string::npos ? 0.5f : std::stof(name.substr(separator + 1));
  if (base == "tweedie-nloglik") {
    require(separator != std::string::npos, "tweedie-nloglik requires a variance power after @.");
    p.parameter = std::stof(name.substr(separator + 1));
    require(p.parameter >= 1 && p.parameter < 2, "Tweedie variance power must be in [1, 2).");
  }
  if (metric_validated_matrix_.lock().get() != &data ||
      metric_validated_generation_ != data.generation ||
      metric_validated_groups_ != groups) {
    metric_validated_matrix_ = data.weak_from_this();
    metric_validated_generation_ = data.generation;
    metric_validated_groups_ = groups;
    metric_validated_mask_ = 0;
  }
  if ((p.multiclass || base == "merror" || base == "mlogloss") && !(metric_validated_mask_ & 1u)) {
    for (auto y : data.get_info("label"))
      require(y >= 0 && y < groups && std::floor(y) == y, "Multiclass evaluation label is out of range.");
    metric_validated_mask_ |= 1u;
  }
  if ((base == "auc" || base == "aucpr") && !p.multiclass && !(metric_validated_mask_ & 2u)) {
    for (auto y : data.get_info("label")) require(y >= 0 && y <= 1, "AUC labels must be in [0,1].");
    metric_validated_mask_ |= 2u;
  }
  Command command("evaluation metric", boolean("metal_coalesce_encoders", true));
  if (base == "quantile" || base == "expectile") {
    auto levels = loss_alphas(base);
    require(checked_product(data.label_cols, levels.size()) == groups,
            "Prediction size must equal label size times the number of alpha values.");
    Buffer alpha(levels.data(), levels.size() * sizeof(float));
    auto tiles = div_up(data.rows, kObjectiveRowsPerTile);
    auto partial = pooled_buffer(metric_pool_, checked_product(checked_product(tiles, groups), (2 * kObjectiveMassWords + 1) * sizeof(u32)));
    auto output = pooled_buffer(metric_pool_, sizeof(MetricRatio));
    command.dispatch("alpha_metric_partial", {prediction, data.labels, data.weights, alpha, partial}, p,
                     {tiles, groups, 1});
    command.dispatch("alpha_metric_reduce", {partial, output}, p, {1, 1, 1}, {1, 1, 1});
    record("metrics", command.wait());
    auto value = *output.as<MetricRatio>();
    require(value.valid, base + " evaluation requires positive total weight.");
    return std::ldexp(static_cast<double>(value.high) + value.low, value.exponent);
  }
  if (base == "normal-nloglik") {
    require(data.label_cols == 1, "Normal NLL requires a single response column.");
    require(groups == 2, "Normal NLL requires two predictions per row: mean and log variance.");
    auto tiles = div_up(data.rows, kObjectiveRowsPerTile);
    auto partial = pooled_buffer(metric_pool_, checked_product(tiles, (3 * kObjectiveMassWords + 1) * sizeof(u32)));
    auto output = pooled_buffer(metric_pool_, sizeof(MetricRatio));
    command.dispatch("normal_metric_partial", {prediction, data.labels, data.weights, partial}, p, {tiles, 1, 1});
    command.dispatch("normal_metric_reduce", {partial, output}, p, {1, 1, 1}, {1, 1, 1});
    record("metrics", command.wait());
    auto value = *output.as<MetricRatio>();
    return std::ldexp(static_cast<double>(value.high) + value.low, value.exponent);
  }
  if (base == "auc" || base == "aucpr") {
    if (!data.rows) return std::numeric_limits<double>::quiet_NaN();
    bool radix = boolean("metal_auc_radix", true);
    active_tuning["auc_radix"] = radix;
    if (radix) {
      auto tiles = static_cast<u32>(div_up(data.rows, kRadixTileRows));
      auto keys_a = pooled_buffer(metric_pool_, checked_product(checked_product(groups, data.rows), sizeof(u32) * 2));
      auto keys_b = pooled_buffer(metric_pool_, keys_a.size());
      auto spine = pooled_buffer(metric_pool_, checked_product(checked_product(groups, 256), tiles * sizeof(u32)));
      auto digit_totals = pooled_buffer(metric_pool_, checked_product(groups, 256 * sizeof(u32)));
      RadixParams rp0{data.rows, groups, tiles, 0};
      command.dispatch("radix_init_desc_histogram", {prediction, keys_a, spine}, rp0, {tiles, groups, 1});
      command.dispatch("radix_scan_spine", {spine, digit_totals}, rp0, {256, groups, 1});
      command.dispatch("radix_scatter", {keys_a, keys_b, spine, digit_totals}, rp0, {tiles, groups, 1});
      std::swap(keys_a, keys_b);
      for (u32 pass = 1; pass < 4; ++pass) {
        RadixParams rp{data.rows, groups, tiles, pass * 8};
        command.dispatch("radix_histogram", {keys_a, spine}, rp, {tiles, groups, 1});
        command.dispatch("radix_scan_spine", {spine, digit_totals}, rp, {256, groups, 1});
        command.dispatch("radix_scatter", {keys_a, keys_b, spine, digit_totals}, rp, {tiles, groups, 1});
        std::swap(keys_a, keys_b);
      }
      auto summaries = pooled_buffer(metric_pool_, checked_product(checked_product(groups, tiles), sizeof(AucTileSummary)));
      auto prefixes = pooled_buffer(metric_pool_, checked_product(checked_product(groups, tiles), sizeof(AucTilePrefix)));
      auto class_totals = pooled_buffer(metric_pool_, checked_product(groups, sizeof(std::uint64_t) * 2));
      auto tile_areas = pooled_buffer(metric_pool_, checked_product(checked_product(groups, tiles), sizeof(std::uint64_t)));
      auto output = pooled_buffer(metric_pool_, checked_product(groups, sizeof(std::uint64_t) * 4));
      command.dispatch("auc_tile_stats", {keys_a, data.labels, data.weights, summaries}, p, {tiles, groups, 1});
      command.dispatch("auc_scan_tiles", {summaries, prefixes, class_totals}, p, {1, groups, 1});
      command.dispatch("auc_tile_integrate",
                       {keys_a, data.labels, data.weights, summaries, prefixes, class_totals, tile_areas}, p,
                       {tiles, groups, 1});
      command.dispatch("auc_final_reduce", {tile_areas, class_totals, output}, p, {1, groups, 1});
      record("metrics", command.wait());
      double sum = 0, total_weight = 0;
      for (u32 k = 0; k < groups; ++k) {
        auto row = output.as<std::uint64_t>() + k * 4;
        double area = std::bit_cast<double>(row[0]);
        double pos = std::bit_cast<double>(row[1]);
        double neg = std::bit_cast<double>(row[2]);
        if (!(pos > 0 && neg > 0)) return std::numeric_limits<double>::quiet_NaN();
        if (base == "auc") area /= (pos * neg);
        double weight = p.multiclass && base == "auc" ? pos : 1.0;
        sum += area * weight; total_weight += weight;
      }
      return total_weight > 0 ? sum / total_weight : std::numeric_limits<double>::quiet_NaN();
    }
    auto a = pooled_buffer(metric_pool_, checked_product(checked_product(p.stride, groups), sizeof(float) * 4));
    auto b = pooled_buffer(metric_pool_, a.size());
    auto output = pooled_buffer(metric_pool_, groups * sizeof(float) * 4);
    command.dispatch("metric_sort", {prediction, data.labels, data.weights, a}, p,
                     {div_up(data.rows, 256), groups, 1});
    for (u32 width = 256; width < p.stride; width *= 2) {
      SortParams sort{data.rows, groups, p.stride, width};
      command.dispatch("metric_merge", {a, b}, sort, {div_up(p.stride, 256), groups, 1});
      std::swap(a, b);
    }
    command.dispatch("auc_reduce", {a, output}, p, {div_up(groups, 32), 1, 1}, {32, 1, 1});
    record("metrics", command.wait());
    double sum = 0, weights = 0;
    for (u32 k = 0; k < groups; ++k) {
      auto row = output.as<float>() + k * 4;
      if (!(row[1] > 0 && row[2] > 0)) return std::numeric_limits<double>::quiet_NaN();
      double weight = p.multiclass && base == "auc" ? row[1] : 1;
      sum += row[0] * weight; weights += weight;
    }
    return weights > 0 ? sum / weights : std::numeric_limits<double>::quiet_NaN();
  }
  auto count = static_cast<std::size_t>(data.rows) * ((base == "merror" || base == "mlogloss") ? 1 : groups);
  auto partial = pooled_buffer(metric_pool_, div_up(count, 256) * sizeof(Pair));
  auto output = pooled_buffer(metric_pool_, sizeof(Pair));
  command.dispatch("metric_partial", {prediction, data.labels, data.weights, partial}, p,
                   {div_up(count, 256), 1, 1});
  command.dispatch("metric_reduce", {partial, output}, p, {1, 1, 1});
  record("metrics", command.wait());
  auto value = *output.as<Pair>();
  double mean = value.h > 0 ? value.g / value.h : std::numeric_limits<double>::quiet_NaN();
  return base == "rmse" || base == "rmsle" ? std::sqrt(mean) : mean;
}
}  // namespace metalgb
