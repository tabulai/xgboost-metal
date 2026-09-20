#include "metalgb/engine.hpp"
#include <algorithm>
#include <array>
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
  if (p.multiclass || base == "merror" || base == "mlogloss")
    for (auto y : data.get_info("label"))
      require(y >= 0 && y < groups && std::floor(y) == y, "Multiclass evaluation label is out of range.");
  if ((base == "auc" || base == "aucpr") && !p.multiclass)
    for (auto y : data.get_info("label")) require(y >= 0 && y <= 1, "AUC labels must be in [0,1].");
  Command command("evaluation metric");
  if (base == "quantile" || base == "expectile") {
    auto levels = loss_alphas(base);
    require(checked_product(data.label_cols, levels.size()) == groups,
            "Prediction size must equal label size times the number of alpha values.");
    Buffer alpha(levels.data(), levels.size() * sizeof(float));
    auto tiles = div_up(data.rows, kObjectiveRowsPerTile);
    Buffer partial(checked_product(checked_product(tiles, groups), (2 * kObjectiveMassWords + 1) * sizeof(u32)));
    Buffer output(sizeof(MetricRatio));
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
    Buffer partial(checked_product(tiles, (3 * kObjectiveMassWords + 1) * sizeof(u32)));
    Buffer output(sizeof(MetricRatio));
    command.dispatch("normal_metric_partial", {prediction, data.labels, data.weights, partial}, p, {tiles, 1, 1});
    command.dispatch("normal_metric_reduce", {partial, output}, p, {1, 1, 1}, {1, 1, 1});
    record("metrics", command.wait());
    auto value = *output.as<MetricRatio>();
    return std::ldexp(static_cast<double>(value.high) + value.low, value.exponent);
  }
  if (base == "auc" || base == "aucpr") {
    Buffer a(checked_product(checked_product(p.stride, groups), sizeof(float) * 4)), b(a.size());
    Buffer output(groups * sizeof(float) * 4);
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
  Buffer partial(div_up(count, 256) * sizeof(Pair)), output(sizeof(Pair));
  command.dispatch("metric_partial", {prediction, data.labels, data.weights, partial}, p,
                   {div_up(count, 256), 1, 1});
  command.dispatch("metric_reduce", {partial, output}, p, {1, 1, 1});
  record("metrics", command.wait());
  auto value = *output.as<Pair>();
  double mean = value.h > 0 ? value.g / value.h : std::numeric_limits<double>::quiet_NaN();
  return base == "rmse" || base == "rmsle" ? std::sqrt(mean) : mean;
}
}  // namespace metalgb
