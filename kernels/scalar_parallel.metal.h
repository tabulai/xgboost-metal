// Score a full numeric feature with eight SIMD groups. Integer prefix sums
// retain every histogram contribution; the final reduction retains the old
// forward-before-backward, first-bin tie order.
template <bool SkipEmptyBins, bool FastStats = false, bool Materialize = true>
inline void evaluate_parallel_subtracted_feature(
    device uint* histogram, device const float2* scales,
    device const float* cuts, device const uint* counts, device const float* minima,
    device const int* monotone, device const float2* bounds, device const uchar* allowed,
    device Split* output, device const long2* node_sums,
    constant SplitParams& p, device const uint* parent_histogram,
    device const HistSubtract* subtraction, uint2 id, uint tid,
    threadgroup long2* group_sums, threadgroup uint* choices, threadgroup uint* parent) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  uint lane = tid & 31, group = tid >> 5;
  HistSubtract task = subtraction[id.y / 2];
  long2 derived_value(0);
  if (id.y == task.other) {
    ulong entries = ulong(p.cols) * (p.bins + 1);
    ulong feature_offset = ulong(id.x) * (p.bins + 1);
    ulong parent_offset = (ulong(task.parent) * entries + feature_offset) * 4;
    ulong built_offset = (ulong(task.built) * entries + feature_offset) * 4;
    ulong output_offset = (ulong(task.other) * entries + feature_offset) * 4;
    if (Materialize) {
      for (uint bin = tid; bin <= p.bins; bin += 256) {
        long2 value = load_hist_int(parent_histogram, parent_offset + ulong(bin) * 4) -
                      load_hist_int(histogram, built_offset + ulong(bin) * 4);
        ulong g = as_type<ulong>(value.x), h = as_type<ulong>(value.y);
        ulong offset = output_offset + ulong(bin) * 4;
        histogram[offset] = uint(g); histogram[offset + 1] = uint(g >> 32);
        histogram[offset + 2] = uint(h); histogram[offset + 3] = uint(h >> 32);
      }
      threadgroup_barrier(mem_flags::mem_device);
    } else if (tid < counts[id.x]) {
      // Only the terminal evaluated level uses this specialization. No later
      // level consumes its histogram. The parent and built sibling are ready
      // before this dispatch; keep their exact subtraction in a register.
      derived_value = load_hist_int(parent_histogram, parent_offset + ulong(tid) * 4) -
                      load_hist_int(histogram, built_offset + ulong(tid) * 4);
    }
  }
  uint feature = id.x, node = id.y, count = counts[feature];
  ulong index = ulong(node) * p.cols + feature;
  ulong offset = index * (p.bins + 1) * 4;
  long2 value = tid < count ?
      (!Materialize && id.y == task.other ? derived_value :
       load_hist_int(histogram, offset + ulong(tid) * 4)) : long2(0);
  long2 prefix = histogram_prefix(value, lane);
  if (lane == 31) group_sums[group] = prefix;
  long2 total_int = node_sums[node];
  float2 scale = scales[p.group];
  if (tid == 0) {
    auto total = FastStats ? scalar_histogram_sum_fast(total_int, scale) : scalar_histogram_sum(total_int, scale);
    float w = scalar_weight(total, p, bounds[node]);
    parent[0] = as_type<uint>(w);
    parent[1] = as_type<uint>(cox_float_result(total.x));
    parent[2] = as_type<uint>(cox_float_result(total.y));
    parent[3] = as_type<uint>(scalar_gain_fast(total, w, p));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  long2 preceding = 0, present = 0;
  for (uint g = 0; g < 8; ++g) {
    long2 sum = group_sums[g];
    present += sum;
    if (g < group) preceding += sum;
  }
  long2 left = preceding + prefix;
  long2 missing = total_int - present;
  float w = as_type<float>(parent[0]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, as_type<float>(parent[1]), as_type<float>(parent[2])};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (tid < count && allowed[index]) {
    float parent_gain = as_type<float>(parent[3]);
    if (!SkipEmptyBins || tid == 0 || value.x != 0 || value.y != 0) {
      float previous = best.gain;
      if (FastStats) consider_split_stats_fast<false, true>(best, left, total_int - left, scale,
          cuts[ulong(feature) * p.bins + tid], 0, feature, monotone[feature],
          bounds[node], parent_gain, p);
      else consider_split_gain<false>(best, left, total_int - left, scale,
          cuts[ulong(feature) * p.bins + tid], 0, feature, monotone[feature],
          bounds[node], parent_gain, p);
      if (best.gain > previous) order = tid;
    }
    if ((missing.x != 0 || missing.y != 0) &&
        (!SkipEmptyBins || tid + 1 == count || value.x != 0 || value.y != 0)) {
      long2 right = present - left + value;
      float previous = best.gain;
      float threshold = tid == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + tid - 1];
      if (FastStats) consider_split_stats_fast<false, true>(best, total_int - right, right, scale, threshold, 1,
          feature, monotone[feature], bounds[node], parent_gain, p);
      else consider_split_gain<false>(best, total_int - right, right, scale, threshold, 1,
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

#define PARALLEL_SPLIT_ARGUMENTS \
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]], \
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]], \
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]], \
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]], \
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]], \
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]], \
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]], \
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]], \
    device const HistSubtract* subtraction [[buffer(16)]], \
    uint2 id [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]

kernel void evaluate_splits_parallel_subtracted_gain(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_sparse_subtracted_gain(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<true>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}


kernel void evaluate_splits_parallel_subtracted_gain_stats(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<false, true>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_sparse_subtracted_gain_stats(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<true, true>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_terminal_gain(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<false, false, false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_terminal_gain_stats(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<false, true, false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_sparse_terminal_gain(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<true, false, false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}

kernel void evaluate_splits_parallel_sparse_terminal_gain_stats(PARALLEL_SPLIT_ARGUMENTS) {
  threadgroup long2 sums[8];
  threadgroup uint choices[25], parent[4];
  evaluate_parallel_subtracted_feature<true, true, false>(histogram, scales, cuts, counts, minima, monotone,
      bounds, allowed, output, node_sums, p, parent_histogram, subtraction, id, tid, sums, choices, parent);
}
#undef PARALLEL_SPLIT_ARGUMENTS
