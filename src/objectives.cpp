#include "metalgb/engine.hpp"
#include <algorithm>

namespace metalgb {
std::vector<float> Learner::loss_alphas(std::string_view loss) const {
  auto key = std::string(loss) + "_alpha";
  auto input = parse_json(str(key, "[]"));
  auto values = input.is_array() ? input.get<std::vector<float>>() : std::vector<float>{input.get<float>()};
  require(!values.empty(), key + " must contain at least one level.");
  require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; }),
          key + " must be in the range [0.0, 1.0].");
  require(std::is_sorted(values.begin(), values.end()), key + " must be sorted in ascending order.");
  require(values.size() <= UINT32_MAX, "Too many " + key + " values.");
  return values;
}

std::vector<float> Learner::score_margins(std::vector<float> const& scores) const {
  auto margins = scores;
  auto obj = objective();
  for (auto& value : margins) {
    require(std::isfinite(value), "base_score must be finite.");
    if (obj == 1) {
      require(value >= 0 && value <= 1, "base_score must be in [0,1] for logistic loss.");
      value = std::clamp(value, 1.0e-6f, 1 - 1.0e-6f);
      value = -std::log(1.0f / value - 1.0f);
    }
    if (obj == 5 || obj == 7 || obj == 8 || obj == 15 || obj == 16) {
      require(obj == 7 ? value > 0 : value >= 0, "Invalid base_score for log-link loss.");
      value = std::log(value);
    }
  }
  if (obj == 13) {
    require(margins.size() == loss_alphas("expectile").size(), "Invalid expectile base score shape.");
    for (std::size_t j = margins.size() - 1; j > 0; --j) {
      float gap = std::max(margins[j] - margins[j - 1] - 1.0e-6f, 1.0e-6f);
      margins[j] = gap + std::log(-std::expm1(-gap));
    }
  }
  return margins;
}

std::vector<float> Learner::estimate_quantiles(Matrix const& data) {
  auto levels = objective() == 12 ? loss_alphas("quantile") : std::vector<float>{0.5f};
  QuantileParams p{data.rows, data.label_cols, groups, static_cast<u32>(levels.size()), 0, 0};
  require(checked_product(data.label_cols, levels.size()) == groups, "Invalid quantile intercept shape.");
  Buffer alpha(levels.data(), levels.size() * sizeof(float));
  Buffer state(checked_product(groups, sizeof(QuantileState)));
  Buffer hist(checked_product(groups, 256 * kObjectiveMassWords * sizeof(u32)));
  Buffer output(checked_product(groups, sizeof(float)));
  Command command("weighted quantile intercept");
  command.zero(state);
  for (p.pass = 0; p.pass < 4; ++p.pass) {
    command.zero(hist);
    command.dispatch("quantile_radix_histogram", {data.labels, data.weights, state, hist}, p,
                     {div_up(data.rows, kObjectiveRowsPerTile), groups, 1});
    command.dispatch("quantile_radix_select", {hist, alpha, state, output}, p,
                     {div_up(groups, 64), 1, 1}, {64, 1, 1});
  }
  record("base_score", command.wait());
  return {output.as<float>(), output.as<float>() + groups};
}

void Learner::smooth_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                               Buffer const& gradient, GradientParams const& p) {
  auto levels = objective() == 12 ? loss_alphas("quantile") : std::vector<float>{0.5f};
  Buffer alpha(levels.data(), levels.size() * sizeof(float));
  auto tiles = div_up(data.rows, kObjectiveRowsPerTile);
  Buffer partial(checked_product(checked_product(tiles, groups), (2 * kObjectiveMassWords + 1) * sizeof(u32)));
  Buffer scale(checked_product(groups, sizeof(float)));
  command.dispatch("residual_scale_partial", {prediction, data.labels, data.weights, partial}, p, {tiles, groups, 1});
  command.dispatch("residual_scale_reduce", {partial, scale}, p, {div_up(groups, 64), 1, 1}, {64, 1, 1});
  command.dispatch("smoothed_gradients", {prediction, data.labels, data.weights, scale, alpha, gradient}, p,
                   {div_up(data.rows, 256), groups, 1});
}

std::vector<float> Learner::estimate_distribution(Matrix const& data) {
  GradientParams p{data.rows, groups, data.label_cols, objective(), 0, 0, 0, 0, 0, 1, 0, 0};
  auto tiles = div_up(data.rows, kObjectiveRowsPerTile);
  Buffer partial_mean(checked_product(tiles, (2 * kProductMassWords + kObjectiveMassWords) * sizeof(u32)));
  Buffer mean(sizeof(InitialMean)), output(checked_product(groups, sizeof(float)));
  Command command("distribution intercept");
  command.dispatch("distribution_mean_partial", {data.labels, data.weights, partial_mean}, p, {tiles, 1, 1});
  command.dispatch("distribution_mean_reduce", {partial_mean, mean}, p, {1, 1, 1}, {1, 1, 1});
  if (p.objective == 13) {
    auto levels = loss_alphas("expectile");
    Buffer alpha(levels.data(), levels.size() * sizeof(float));
    Buffer partial(checked_product(checked_product(tiles, groups), (3 * kObjectiveMassWords + 1) * sizeof(u32)));
    command.dispatch("expectile_intercept_partial", {data.labels, data.weights, mean, alpha, partial}, p, {tiles, groups, 1});
    command.dispatch("expectile_intercept_reduce", {partial, mean, output}, p, {div_up(groups, 64), 1, 1}, {64, 1, 1});
    command.dispatch("expectile_intercept_order", {output}, p, {1, 1, 1}, {1, 1, 1});
  } else {
    Buffer partial(checked_product(tiles, (2 * kVarianceMassWords + kObjectiveMassWords) * sizeof(u32)));
    command.dispatch("normal_variance_partial", {data.labels, data.weights, mean, partial}, p, {tiles, 1, 1});
    command.dispatch("normal_variance_reduce", {partial, mean, output}, p, {1, 1, 1}, {1, 1, 1});
  }
  record("base_score", command.wait());
  require(mean.as<InitialMean>()->valid, "weights must contain at least one non-zero value.");
  return {output.as<float>(), output.as<float>() + groups};
}

void Learner::distribution_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                                     Buffer const& gradient, GradientParams const& p) {
  if (p.objective == 13) {
    auto levels = loss_alphas("expectile");
    Buffer alpha(levels.data(), levels.size() * sizeof(float));
    command.dispatch("expectile_gradients", {prediction, data.labels, data.weights, alpha, gradient}, p,
                     {div_up(data.rows, 256), groups, 1});
  } else {
    command.dispatch("normal_gradients", {prediction, data.labels, data.weights, gradient}, p,
                     {div_up(data.rows, 256), 1, 1});
  }
}
}  // namespace metalgb
