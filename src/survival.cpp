#include "metalgb/engine.hpp"
#include "metalgb/cox.hpp"
#include <algorithm>

namespace metalgb {
SurvivalParams Learner::aft_parameters(u32 rows) const {
  auto distribution = str("aft_loss_distribution", "normal");
  require(distribution == "normal" || distribution == "logistic" || distribution == "extreme",
          "Unknown aft_loss_distribution: " + distribution);
  return {rows, distribution == "normal" ? 0u : distribution == "logistic" ? 1u : 2u,
          0, random_seed(), number("aft_loss_distribution_scale", 1),
          1, rounds, 0};
}

Buffer Learner::survival_order(Command& command, Matrix const& data) {
  for (auto const& entry : survival_orders_)
    if (entry.matrix.lock().get() == &data && entry.generation == data.generation) return entry.indices;
  std::erase_if(survival_orders_, [&](auto const& entry) {
    auto matrix = entry.matrix.lock();
    return !matrix || matrix.get() == &data;
  });
  Buffer source(checked_product(data.rows, sizeof(u32)));
  if (data.rows > 256) {
    auto radix_tiles = static_cast<u32>(div_up(data.rows, kRadixTileRows));
    Buffer keys_a(checked_product(data.rows, 2 * sizeof(u32))), keys_b(keys_a.size());
    Buffer spine(checked_product(256u, radix_tiles) * sizeof(u32)), digit_totals(256u * sizeof(u32));
    for (u32 pass = 0; pass < 4; ++pass) {
      RadixParams rp{data.rows, 1u, radix_tiles, pass * 8u};
      if (pass == 0) {
        command.dispatch("survival_radix_init", {data.labels, keys_a, spine}, rp, {radix_tiles, 1, 1});
      } else {
        command.dispatch("radix_histogram", {keys_a, spine}, rp, {radix_tiles, 1, 1});
      }
      command.dispatch("radix_scan_spine", {spine, digit_totals}, rp, {256, 1, 1});
      command.dispatch("radix_scatter", {keys_a, keys_b, spine, digit_totals}, rp, {radix_tiles, 1, 1});
      std::swap(keys_a, keys_b);
    }
    RadixParams rp{data.rows, 1u, radix_tiles, 0u};
    command.dispatch("radix_extract_payload", {keys_a, source}, rp, {div_up(data.rows, 256), 1, 1});
  } else {
    SortParams p{data.rows, 1, data.rows, 256};
    command.dispatch("survival_order_tiles", {data.labels, source}, p, {div_up(data.rows, 256), 1, 1});
  }
  survival_orders_.push_back({data.weak_from_this(), data.generation, source});
  return source;
}

Buffer Learner::survival_scan(Command& command, Buffer const& input, u32 rows) {
  auto blocks = div_up(rows, 256);
  Buffer output(checked_product(rows, kSurvivalSumBytes));
  Buffer totals(checked_product(blocks, kSurvivalSumBytes));
  SurvivalParams p{rows, 0, 0, 0, 1, 1, 0, 0};
  command.dispatch("survival_scan_tiles", {input, output, totals}, p, {blocks, 1, 1});
  if (blocks > 1) {
    auto scanned = survival_scan(command, totals, blocks);
    command.dispatch("survival_scan_offsets", {output, scanned}, p, {blocks, 1, 1});
  }
  return output;
}

void Learner::survival_gradients(Command& command, Matrix const& data, Buffer const& prediction,
                                 Buffer const& gradient, GradientParams const& params) {
  if (!data.rows) return;
  if (params.objective == 15) {
    auto p = aft_parameters(data.rows);
    command.dispatch("aft_gradients", {prediction, data.lower_bounds, data.upper_bounds, data.weights, gradient},
                     p, {div_up(data.rows, 256), 1, 1});
    return;
  }
  auto order = survival_order(command, data);
  Buffer inputs(checked_product(data.rows, 2 * sizeof(std::uint64_t)));
  Buffer increments(inputs.size()), terms(inputs.size()), zero(2 * sizeof(u64)), risk_seed(zero.size());
  command.zero(zero);
  command.dispatch("cox_exact_inputs", {prediction, order, inputs}, data.rows, {div_up(data.rows, 256), 1, 1});
  auto prefix = cox_ordered_scan(command, inputs, data.labels, order, zero, data.rows, true);
  command.dispatch("cox_risk_increments", {prefix, data.labels, order, increments, risk_seed}, data.rows,
                   {div_up(data.rows, 256), 1, 1});
  auto risks = cox_ordered_scan(command, increments, data.labels, order, risk_seed, data.rows, false);
  command.dispatch("cox_exact_terms", {risks, data.labels, order, terms}, data.rows, {div_up(data.rows, 256), 1, 1});
  auto cumulative = cox_ordered_scan(command, terms, data.labels, order, zero, data.rows, false);
  command.dispatch("cox_exact_gradients", {inputs, cumulative, data.labels, data.weights, order, gradient},
                   data.rows, {div_up(data.rows, 256), 1, 1});
}

double Learner::survival_metric(Matrix const& data, Buffer const& prediction, u32 type) {
  require(groups == 1 && data.label_cols == 1, "Survival metrics require one prediction per row.");
  require(data.get_info("weight").empty() || data.get_info("weight").size() == data.rows,
          "Evaluation weights must have one entry per row.");
  if (type != 2) {
    require(data.get_info("label_lower_bound").size() == data.rows &&
                data.get_info("label_upper_bound").size() == data.rows,
            "Survival evaluation requires one label_lower_bound and label_upper_bound per row.");
  }
  if (!data.rows) return type == 2 ? std::numeric_limits<double>::quiet_NaN() : 0.0;
  auto p = type == 2 ? SurvivalParams{data.rows, 0, 2, 0, 1, 1, 0, 0} : aft_parameters(data.rows);
  p.mode = type;
  auto count = div_up(data.rows, 256);
  Buffer partial(checked_product(count, kSurvivalSumBytes));
  Command command("survival metric");
  if (type == 2) {
    auto order = survival_order(command, data);
    Buffer inputs(checked_product(data.rows, kSurvivalSumBytes));
    command.dispatch("cox_metric_inputs", {prediction, order, inputs}, p, {div_up(data.rows, 256), 1, 1});
    auto risks = survival_scan(command, inputs, data.rows);
    command.dispatch("cox_metric_partial", {prediction, data.labels, order, risks, partial}, p, {count, 1, 1});
  } else {
    command.dispatch("aft_metric_partial", {prediction, data.lower_bounds, data.upper_bounds, data.weights, partial},
                     p, {count, 1, 1});
  }
  while (count > 1) {
    p.rows = count;
    count = div_up(count, 256);
    Buffer next(checked_product(count, kSurvivalSumBytes));
    command.dispatch("survival_reduce", {partial, next}, p, {count, 1, 1});
    partial = std::move(next);
  }
  Buffer output(sizeof(MetricRatio));
  command.dispatch("survival_metric_finish", {partial, output}, p, {1, 1, 1}, {1, 1, 1});
  record("metrics", command.wait());
  auto value = *output.as<MetricRatio>();
  return std::ldexp(static_cast<double>(value.high) + value.low, value.exponent);
}
}  // namespace metalgb
