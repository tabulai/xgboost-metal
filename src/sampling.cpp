#include "metalgb/engine.hpp"
#include <algorithm>

namespace metalgb {
Buffer Learner::sampling_buffer(std::string key, std::size_t bytes) {
  auto& buffer = sampling_workspace_[std::move(key)];
  if (!buffer || buffer.size() < bytes) buffer = Buffer(bytes);
  return buffer.prefix(bytes);
}

Learner::SampledGradients Learner::sample_gradients(Buffer const& gradient, u32 rows, u32 group,
                                                   u32 sample_rows, bool prepare, u32 targets, u32 source_groups) {
  SamplingParams p{rows, source_groups ? source_groups : groups, group, sample_rows, static_cast<u32>(rng_()),
                    str("sampling_method", "uniform") == "gradient_based" ? 1u : 0u,
                    0, targets, number("subsample", 1), 0, 0, 0};
  return sample_gradients(gradient, p, prepare, false);
}

Learner::SampledGradients Learner::sample_gradients(Buffer const& gradient, SamplingParams p, bool prepare, bool replay) {
  auto rows = p.rows, targets = std::max(p.padding, 1u);
  auto buffer = [&](std::string name, std::size_t bytes) {
    return sampling_buffer(replay ? "replay_" + name : name, bytes);
  };
  auto size = checked_product(rows, targets);
  auto values = buffer("values", checked_product(size, sizeof(Pair)));
  auto quantized = buffer("quantized", checked_product(size, 2 * sizeof(std::int64_t)));
  auto scales = buffer("scales", checked_product(targets, sizeof(Pair)));
  auto magnitude = sampling_buffer("magnitude", p.method && p.sample_rows ? checked_product(rows, sizeof(float)) : 0);
  auto threshold = sampling_buffer("threshold", sizeof(float));
  Command command("row sampling", integer("metal_coalesce_encoders", 0), integer("metal_compute_fill", 0));
  if (p.method && p.sample_rows && prepare) {
    auto stride = div_up(rows, 256) * 256;
    require(stride <= (std::size_t{1} << 31), "Sampling sort exceeds the dispatch index range.");
    p.stride = stride;
    auto first = sampling_buffer("sort_a", checked_product(stride, sizeof(Pair)));
    auto second = sampling_buffer("sort_b", first.size());
    auto prefix = sampling_buffer("prefix", checked_product(rows, sizeof(float)));
    command.dispatch("sampling_magnitudes", {gradient, magnitude, first}, p, {stride / 256, 1, 1});
    for (std::size_t width = 256; width < stride; width *= 2) {
      SortParams sort{rows, 1, static_cast<u32>(stride), static_cast<u32>(width)};
      command.dispatch("merge_sorted", {first, second}, sort, {div_up(stride, 256), 1, 1});
      std::swap(first, second);
    }
    // CPU uses a sequential float32 partial_sum. Keep that accumulation order
    // on the GPU; CUDA's fixed-point scan can select a different threshold.
    command.dispatch("sampling_prefix", {first, prefix}, p, {1, 1, 1}, {32, 1, 1});
    command.dispatch("sampling_threshold", {first, prefix, threshold}, p, {1, 1, 1}, {1, 1, 1});
  }
  command.dispatch("sampling_apply", {gradient, magnitude, threshold, values}, p, {div_up(rows, 256), 1, 1});
  GradientParams quantize{rows, targets, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0};
  auto partial = buffer("partial", checked_product(checked_product(div_up(rows, kGradientRowsPerTile), targets),
                                                  kGradientPartialWords * sizeof(u32)));
  command.dispatch("gradient_partial_sums", {values, partial}, quantize, {div_up(rows, kGradientRowsPerTile), targets, 1});
  command.dispatch("gradient_scales", {partial, scales}, quantize, {2, targets, 1});
  command.dispatch("quantize_gradients", {values, scales, quantized}, quantize, {div_up(rows, 256), targets, 1});
  record("sampling", command.wait());
  return {values, quantized, scales, p};
}
}  // namespace metalgb
