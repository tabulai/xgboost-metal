// Fuse sibling-histogram materialization with the numeric SIMD split search.
// The host selects this only for nonroot scalar GPU-driver levels with at most
// 256 bins. Scoring, prefix scans, candidate order and applied model arithmetic
// remain in their existing paths. FastWeight optionally replaces restoring
// division with an exactly certified quotient and the same rounding rules.
// FastGain uses an exact integer square for normal-range float32 weights;
// the original software path handles all other values.
inline void consider_split_fast(thread Split& best, long2 left_int, long2 right_int, float2 scale, float threshold,
                           uint default_left, uint feature, int monotone,
                           float2 bounds, float parent_gain, constant SplitParams& p) {
  auto left = scalar_histogram_sum(left_int, scale), right = scalar_histogram_sum(right_int, scale);
  if (!scalar_valid_hessian(left.y, p.min_child_weight) ||
      !scalar_valid_hessian(right.y, p.min_child_weight)) return;
  float lw = scalar_weight_fast(left, p, bounds), rw = scalar_weight_fast(right, p, bounds);
  if ((monotone > 0 && !fp32_less_equal(lw, rw)) || (monotone < 0 && !fp32_greater_equal(lw, rw))) return;
  float gain = scalar_float_add(scalar_float_add(scalar_gain(left, lw, p), scalar_gain(right, rw, p)), -parent_gain);
  if (gain > best.gain) {
    best.gain = gain; best.threshold = threshold; best.feature = feature;
    best.default_left = default_left;
    best.left_h = cox_float_result(left.y);
    best.right_h = cox_float_result(right.y);
    best.left_weight = lw; best.right_weight = rw;
    best.left_sum = IntPair{left_int.x, left_int.y};
    best.right_sum = IntPair{right_int.x, right_int.y};
  }
}

template <bool FastWeight>
inline void consider_split_gain(thread Split& best, long2 left_int, long2 right_int, float2 scale, float threshold,
                           uint default_left, uint feature, int monotone,
                           float2 bounds, float parent_gain, constant SplitParams& p) {
  auto left = scalar_histogram_sum(left_int, scale), right = scalar_histogram_sum(right_int, scale);
  if (!scalar_valid_hessian(left.y, p.min_child_weight) ||
      !scalar_valid_hessian(right.y, p.min_child_weight)) return;
  float lw = FastWeight ? scalar_weight_fast(left, p, bounds) : scalar_weight(left, p, bounds);
  float rw = FastWeight ? scalar_weight_fast(right, p, bounds) : scalar_weight(right, p, bounds);
  if ((monotone > 0 && !fp32_less_equal(lw, rw)) || (monotone < 0 && !fp32_greater_equal(lw, rw))) return;
  float gain = scalar_float_add(scalar_float_add(scalar_gain_fast(left, lw, p), scalar_gain_fast(right, rw, p)), -parent_gain);
  if (gain > best.gain) {
    best.gain = gain; best.threshold = threshold; best.feature = feature;
    best.default_left = default_left;
    best.left_h = cox_float_result(left.y);
    best.right_h = cox_float_result(right.y);
    best.left_weight = lw; best.right_weight = rw;
    best.left_sum = IntPair{left_int.x, left_int.y};
    best.right_sum = IntPair{right_int.x, right_int.y};
  }
}

template <bool SkipEmptyBins, bool FastWeight, bool FastGain, bool FastStats = false>
inline void evaluate_subtracted_feature(
    device uint* histogram, device const float2* scales,
    device const float* cuts, device const uint* counts, device const float* minima,
    device const int* monotone, device const float2* bounds, device const uchar* allowed,
    device Split* output, device const uchar* categorical, device const float2* sorted,
    device const uint* ranks, device uint* categories, device const long2* node_sums,
    constant SplitParams& p, device const uint* parent_histogram,
    device const HistSubtract* subtraction, uint2 id, uint lane) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  // plan_histograms assigns one built and one derived sibling to each pair.
  // Histogram construction completed in the preceding dispatch. Distinct
  // (node, feature) groups own disjoint slices, and no group writes a built
  // sibling, so subtraction has no cross-threadgroup producer dependency.
  HistSubtract task = subtraction[id.y / 2];
  if (id.y == task.other) {
    ulong entries = ulong(p.cols) * (p.bins + 1);
    ulong feature_offset = ulong(id.x) * (p.bins + 1);
    ulong parent_offset = (ulong(task.parent) * entries + feature_offset) * 4;
    ulong built_offset = (ulong(task.built) * entries + feature_offset) * 4;
    ulong output_offset = (ulong(task.other) * entries + feature_offset) * 4;
    for (uint bin = lane; bin <= p.bins; bin += 32) {
      long2 value = load_hist_int(parent_histogram, parent_offset + ulong(bin) * 4) -
                    load_hist_int(histogram, built_offset + ulong(bin) * 4);
      ulong g = as_type<ulong>(value.x), h = as_type<ulong>(value.y);
      ulong offset = output_offset + ulong(bin) * 4;
      histogram[offset] = uint(g); histogram[offset + 1] = uint(g >> 32);
      histogram[offset + 2] = uint(h); histogram[offset + 3] = uint(h >> 32);
    }
    // The next scan reads this same group's newly materialized feature slice.
    // Keep the complete histogram, including unused and missing bins, for
    // subtraction by a later tree level.
    threadgroup_barrier(mem_flags::mem_device);
  }
  if (p.has_categorical && categorical[id.x]) {
    if (!lane) evaluate_feature(histogram, scales, cuts, counts, minima, monotone, bounds, allowed, output,
                                  categorical, sorted, ranks, categories, node_sums, p, id);
    return;
  }
  uint feature = id.x, node = id.y, count = counts[feature];
  ulong offset = (ulong(node) * p.cols + feature) * (p.bins + 1) * 4;
  float2 scale = scales[p.group];
  long2 missing = load_hist_int(histogram, offset + p.bins * 4), local = 0;
  for (uint b = lane; b < count; b += 32) local += load_hist_int(histogram, offset + b * 4);
  long2 present = histogram_broadcast(histogram_prefix(local, lane), 31);
  long2 total_int = p.use_parent_sums ? node_sums[node] : present + missing;
  missing = total_int - present;
  auto total = FastStats ? scalar_histogram_sum_fast(total_int, scale) : scalar_histogram_sum(total_int, scale);
  float w = FastWeight ? scalar_weight_fast(total, p, bounds[node]) : scalar_weight(total, p, bounds[node]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, cox_float_result(total.x), cox_float_result(total.y)};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (allowed[ulong(node) * p.cols + feature]) {
    float parent_gain = FastGain ? scalar_gain_fast(total, w, p) : scalar_gain(total, w, p);
    long2 preceding = 0;
    for (uint begin = 0; begin < count; begin += 32) {
      uint bin = begin + lane;
      long2 value = bin < count ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto left = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (bin < count && (!SkipEmptyBins || bin == 0 || value.x != 0 || value.y != 0)) {
        float previous = best.gain;
        if (FastStats)
          consider_split_stats_fast<FastWeight, FastGain>(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + bin], 0,
                                          feature, monotone[feature], bounds[node], parent_gain, p);
        else if (FastGain)
          consider_split_gain<FastWeight>(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + bin], 0,
                                          feature, monotone[feature], bounds[node], parent_gain, p);
        else if (FastWeight)
          consider_split_fast(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + bin], 0,
                              feature, monotone[feature], bounds[node], parent_gain, p);
        else
          consider_split(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + bin], 0,
                         feature, monotone[feature], bounds[node], parent_gain, p);
        if (best.gain > previous) order = bin;
      }
    }
    preceding = 0;
    for (uint begin = 0; begin < ((missing.x != 0 || missing.y != 0) ? count : 0); begin += 32) {
      uint step = begin + lane;
      uint bin = count - min(step + 1, count);
      long2 value = step < count ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto right = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (step < count && (!SkipEmptyBins || step == 0 || value.x != 0 || value.y != 0)) {
        float previous = best.gain;
        float threshold = bin == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + bin - 1];
        if (FastStats)
          consider_split_stats_fast<FastWeight, FastGain>(best, total_int - right, right, scale, threshold, 1,
                                          feature, monotone[feature], bounds[node], parent_gain, p);
        else if (FastGain)
          consider_split_gain<FastWeight>(best, total_int - right, right, scale, threshold, 1,
                                          feature, monotone[feature], bounds[node], parent_gain, p);
        else if (FastWeight)
          consider_split_fast(best, total_int - right, right, scale, threshold, 1,
                              feature, monotone[feature], bounds[node], parent_gain, p);
        else
          consider_split(best, total_int - right, right, scale, threshold, 1,
                         feature, monotone[feature], bounds[node], parent_gain, p);
        if (best.gain > previous) order = count + step;
      }
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) output[ulong(node) * p.cols + feature] = best;
}

kernel void evaluate_splits_simd_subtracted(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<false, false, false>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_sparse_subtracted(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<true, false, false>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_subtracted_fast(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<false, true, false>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_sparse_subtracted_fast(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<true, true, false>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_subtracted_gain(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<false, false, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_sparse_subtracted_gain(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<true, false, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_subtracted_fast_gain(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<false, true, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}

kernel void evaluate_splits_simd_sparse_subtracted_fast_gain(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<true, true, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}



kernel void evaluate_splits_simd_subtracted_gain_stats(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<false, false, true, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}


kernel void evaluate_splits_simd_sparse_subtracted_gain_stats(
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]],
    device const HistSubtract* subtraction [[buffer(16)]],
    uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  evaluate_subtracted_feature<true, false, true, true>(histogram, scales, cuts, counts, minima, monotone, bounds,
      allowed, output, categorical, sorted, ranks, categories, node_sums, p, parent_histogram,
      subtraction, id, lane);
}
