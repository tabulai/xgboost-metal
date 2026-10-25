// A zero integer histogram bin leaves both child sums unchanged from the
// preceding candidate in that scan direction. Its gain and validity are
// identical, and the preceding candidate wins the existing order tie-break.
// Keep each direction's first candidate, all prefix arithmetic, and categorical
// evaluation unchanged. A bin whose gradient cancels but Hessian is nonzero
// must still be evaluated.
kernel void evaluate_splits_simd_sparse(device const uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
                                 device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
                                 device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
                                 device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
                                 device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
                                 device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
                                 device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
                                 constant SplitParams& p [[buffer(14)]],
                                 uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
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
  auto total = scalar_histogram_sum(total_int, scale);
  float w = scalar_weight(total, p, bounds[node]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, cox_float_result(total.x), cox_float_result(total.y)};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (allowed[ulong(node) * p.cols + feature]) {
    float parent_gain = scalar_gain(total, w, p);
    long2 preceding = 0;
    for (uint begin = 0; begin < count; begin += 32) {
      uint bin = begin + lane;
      long2 value = bin < count ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto left = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (bin < count && (bin == 0 || value.x != 0 || value.y != 0)) {
        float previous = best.gain;
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
      if (step < count && (step == 0 || value.x != 0 || value.y != 0)) {
        float previous = best.gain;
        float threshold = bin == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + bin - 1];
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
