// Root-only numeric split scoring with eight SIMD groups per feature. Retain
// the original integer prefixes, stored missing total, candidate order and
// downstream root-selection kernel. This path has no inherited node sums.
template <bool FastStats = false, bool FusedScore = false>
inline void evaluate_parallel_root_feature(
    device const uint* histogram, device const float2* scales,
    device const float* cuts, device const uint* counts, device const float* minima,
    device const int* monotone, device const float2* bounds, device const uchar* allowed,
    device Split* output, constant SplitParams& p, uint2 id, uint tid,
    threadgroup long2* group_sums, threadgroup uint* choices, threadgroup uint* parent) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  uint lane = tid & 31, group = tid >> 5;
  uint feature = id.x, node = id.y, count = counts[feature];
  ulong index = ulong(node) * p.cols + feature;
  ulong offset = index * (p.bins + 1) * 4;
  long2 value = tid < count ? load_hist_int(histogram, offset + ulong(tid) * 4) : long2(0);
  long2 prefix = histogram_prefix(value, lane);
  if (lane == 31) group_sums[group] = prefix;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  long2 preceding = 0, present = 0;
  for (uint g = 0; g < 8; ++g) {
    long2 sum = group_sums[g];
    present += sum;
    if (g < group) preceding += sum;
  }
  long2 left = preceding + prefix;
  long2 missing = load_hist_int(histogram, offset + ulong(p.bins) * 4);
  long2 total_int = present + missing;
  float2 scale = scales[p.group];
  if (tid == 0) {
    auto total = FastStats ? scalar_histogram_sum_fast(total_int, scale) : scalar_histogram_sum(total_int, scale);
    float2 score;
    if (FusedScore) score = scalar_weight_gain_fused(total, p, bounds[node]);
    else {
      score.x = scalar_weight(total, p, bounds[node]);
      score.y = scalar_gain_fast(total, score.x, p);
    }
    float w = score.x;
    parent[0] = as_type<uint>(w);
    parent[1] = as_type<uint>(cox_float_result(total.x));
    parent[2] = as_type<uint>(cox_float_result(total.y));
    parent[3] = as_type<uint>(score.y);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float w = as_type<float>(parent[0]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, as_type<float>(parent[1]), as_type<float>(parent[2])};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (tid < count && allowed[index]) {
    float parent_gain = as_type<float>(parent[3]);
    float previous = best.gain;
    if (FusedScore)
      consider_split_score_fused(best, left, total_int - left, scale,
          cuts[ulong(feature) * p.bins + tid], 0, feature, monotone[feature],
          bounds[node], parent_gain, p);
    else if (FastStats)
      consider_split_stats_fast<false, true>(best, left, total_int - left, scale,
          cuts[ulong(feature) * p.bins + tid], 0, feature, monotone[feature],
          bounds[node], parent_gain, p);
    else
      consider_split_gain<false>(best, left, total_int - left, scale,
          cuts[ulong(feature) * p.bins + tid], 0, feature, monotone[feature],
          bounds[node], parent_gain, p);
    if (best.gain > previous) order = tid;
    if (missing.x != 0 || missing.y != 0) {
      long2 right = present - left + value;
      previous = best.gain;
      float threshold = tid == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + tid - 1];
      if (FusedScore)
        consider_split_score_fused(best, total_int - right, right, scale, threshold, 1,
            feature, monotone[feature], bounds[node], parent_gain, p);
      else if (FastStats)
        consider_split_stats_fast<false, true>(best, total_int - right, right, scale, threshold, 1,
            feature, monotone[feature], bounds[node], parent_gain, p);
      else
        consider_split_gain<false>(best, total_int - right, right, scale, threshold, 1,
            feature, monotone[feature], bounds[node], parent_gain, p);
      if (best.gain > previous) order = count + (count - 1 - tid);
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) {
    choices[group * 3] = as_type<uint>(best.gain);
    choices[group * 3 + 1] = order;
    choices[group * 3 + 2] = tid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float gain = lane < 8 ? as_type<float>(choices[lane * 3]) : -INFINITY;
    uint position = lane < 8 ? choices[lane * 3 + 1] : 0xffffffffu;
    float maximum = simd_max(gain);
    uint first = simd_min(gain == maximum ? position : 0xffffffffu);
    uint selected = simd_min(lane < 8 && gain == maximum && position == first ? lane : 0xffffffffu);
    if (lane == 0) choices[24] = choices[selected * 3 + 2];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == choices[24]) output[index] = best;
}

#define ROOT_PARALLEL_SPLIT_ARGUMENTS \
    device const uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]], \
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]], \
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]], \
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]], \
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]], \
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]], \
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]], \
    constant SplitParams& p [[buffer(14)]], \
    uint2 id [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]

kernel void evaluate_splits_parallel_root_gain(ROOT_PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_root_feature<false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, p, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_root_gain_stats(ROOT_PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_root_feature<true>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, p, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_root_gain_stats_score(ROOT_PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_root_feature<true, true>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, p, id, tid, sums, choices, parent);
}
#undef ROOT_PARALLEL_SPLIT_ARGUMENTS
