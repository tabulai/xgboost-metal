// CUDA's vector histogram evaluator separates target-major histogram scans
// from bin-major split scoring. Keep that layout here so target count does not
// impose a fixed-size register array or a limit on the public output shape.
inline ulong vector_histogram_offset(uint node, uint target, uint feature,
                                     constant VectorSplitParams& p) {
  return ((ulong(node) * p.targets + target) * p.split.cols + feature) * (p.split.bins + 1) * 4;
}
inline ulong vector_scan_offset(uint node, uint feature, uint reverse, uint bin,
                                constant VectorSplitParams& p) {
  return (((ulong(node) * p.split.cols + feature) * 2 + reverse) * p.split.bins + bin) * p.targets;
}

kernel void vector_node_sums(device const uint* histogram [[buffer(0)]],
                              device const uint* counts [[buffer(1)]],
                              device long2* totals [[buffer(2)]],
                              constant VectorSplitParams& p [[buffer(3)]],
                              uint2 block [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  auto offset = vector_histogram_offset(block.y, block.x, 0, p);
  long2 sum = 0;
  for (uint bin = lane; bin < counts[0]; bin += 32) sum += load_hist_int(histogram, offset + bin * 4);
  sum = histogram_prefix(sum, lane);
  if (lane == 31)
    totals[ulong(block.y) * p.targets + block.x] = sum + load_hist_int(histogram, offset + p.split.bins * 4);
}

inline SurvivalReal vector_gain_given_weight(SurvivalSum value, float weight, constant SplitParams& p) {
  // The parent and constrained child weights are f32, including their square;
  // the reference evaluates and sums their quadratic gains in double precision.
  return -(sr(2) * value.value * sr(weight) + (value.weight + sr(p.reg_lambda)) * sr(weight * weight) +
             sr(2) * sr(p.reg_alpha) * sr(fabs(weight)));
}
inline float vector_weight(SurvivalSum value, constant SplitParams& p, float2 bounds) {
  // Vector constraints box every target, including a target whose own Hessian
  // is nonpositive while the average child Hessian permits the shared split.
  return clamp(leaf_weight(value, p, bounds), bounds.x, bounds.y);
}
inline SurvivalReal vector_threshold(SurvivalReal gradient, float alpha) {
  if (survival_less(sr(alpha), gradient)) return gradient - sr(alpha);
  if (survival_less(gradient, sr(-alpha))) return gradient + sr(alpha);
  return sr(0);
}
inline SurvivalReal vector_optimal_gain(SurvivalSum value, constant SplitParams& p) {
  if (!survival_less(sr(0), value.weight)) return sr(0);
  auto gradient = vector_threshold(value.value, p.reg_alpha);
  auto denominator = value.weight + sr(p.reg_lambda);
  if (p.max_delta_step == 0) return gradient * gradient / denominator;
  auto weight = -gradient / denominator;
  if (survival_less(sr(p.max_delta_step), survival_abs(weight)))
    weight = sr(copysign(p.max_delta_step, weight.significand.x));
  return -(sr(2) * value.value * weight + denominator * weight * weight +
             sr(2) * sr(p.reg_alpha) * survival_abs(weight));
}
inline float vector_pooled_weight(SurvivalSum total, float2 bounds, constant SplitParams& p) {
  if (!survival_less(sr(0), total.weight)) return clamp(0.0f, bounds.x, bounds.y);
  auto gradient = vector_threshold(total.value, 2.0f * p.reg_alpha);
  float value = gradient.significand.x == 0 ? -0.0f :
                extended_result(-gradient / (total.weight + sr(2.0f * p.reg_lambda)));
  if (p.max_delta_step != 0) value = clamp(value, -p.max_delta_step, p.max_delta_step);
  return clamp(value, bounds.x, bounds.y);
}
inline float2 vector_split_weights(SurvivalSum left, SurvivalSum right, float2 bounds,
                                    int monotone, constant SplitParams& p) {
  float lw = vector_weight(left, p, bounds), rw = vector_weight(right, p, bounds);
  if ((monotone > 0 && lw > rw) || (monotone < 0 && lw < rw)) {
    float pooled = vector_pooled_weight(left + right, bounds, p);
    return float2(pooled);
  }
  return float2(lw, rw);
}

struct VectorCategory { float high, low; int exponent; uint bin; };
inline SurvivalReal vector_category_score(VectorCategory value) {
  return SurvivalReal{float2(value.high, value.low), value.exponent};
}
inline bool vector_category_less(VectorCategory a, VectorCategory b) {
  auto av = vector_category_score(a), bv = vector_category_score(b);
  if (survival_less(av, bv)) return true;
  return !survival_less(bv, av) && a.bin < b.bin;
}
kernel void vector_category_sort(device const uint* histogram [[buffer(0)]],
                                  device const float2* scales [[buffer(1)]],
                                  device const uint* counts [[buffer(2)]],
                                  device const uchar* categorical [[buffer(3)]],
                                  device const long2* totals [[buffer(4)]],
                                  device VectorCategory* output [[buffer(5)]],
                                  constant VectorSplitParams& p [[buffer(6)]],
                                  uint2 block [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup VectorCategory values[256];
  uint bin = block.x * 256 + tid, node = block.y / p.split.cols, feature = block.y % p.split.cols;
  auto score = sr(INFINITY);
  if (categorical[feature] && bin < counts[feature]) {
    score = sr(0);
    for (uint target = 0; target < p.targets; ++target) {
      auto scale = scales[target];
      auto parent = histogram_sum(totals[ulong(node) * p.targets + target], scale);
      auto child = histogram_sum(load_hist_int(histogram, vector_histogram_offset(node, target, feature, p) + bin * 4), scale);
      float pw = vector_weight(parent, p.split, float2(-INFINITY, INFINITY));
      float cw = vector_weight(child, p.split, float2(-INFINITY, INFINITY));
      score = score + sr(pw * cw);
    }
  }
  values[tid] = VectorCategory{score.significand.x, score.significand.y, score.exponent, bin};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint width = 2; width <= 256; width <<= 1) for (uint delta = width >> 1; delta; delta >>= 1) {
    auto a = values[tid], b = values[tid ^ delta];
    bool ascending = ((tid & width) == 0) == ((tid & delta) == 0);
    auto next = (ascending ? vector_category_less(b, a) : vector_category_less(a, b)) ? b : a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  output[ulong(block.y) * p.split.sort_stride + bin] = values[tid];
}
kernel void vector_category_merge(device const VectorCategory* source [[buffer(0)]],
                                   device VectorCategory* destination [[buffer(1)]],
                                   constant SortParams& p [[buffer(2)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.stride || id.y >= p.cols) return;
  uint begin = (id.x / (2 * p.width)) * (2 * p.width);
  uint middle = min(begin + p.width, p.stride), end = min(begin + 2 * p.width, p.stride);
  bool left = id.x < middle;
  uint own = left ? begin : middle, other = left ? middle : begin;
  uint lo = other, hi = left ? end : middle;
  ulong offset = ulong(id.y) * p.stride;
  auto value = source[offset + id.x];
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (vector_category_less(source[offset + mid], value)) lo = mid + 1;
    else hi = mid;
  }
  destination[offset + begin + id.x - own + lo - other] = value;
}
kernel void vector_category_ranks(device const VectorCategory* sorted [[buffer(0)]],
                                   device const uint* counts [[buffer(1)]],
                                   device const uchar* categorical [[buffer(2)]],
                                   device uint* ranks [[buffer(3)]],
                                   constant VectorSplitParams& p [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.split.nodes * p.split.cols || !categorical[id.y % p.split.cols] || id.x >= counts[id.y % p.split.cols]) return;
  uint bin = sorted[ulong(id.y) * p.split.sort_stride + id.x].bin;
  ranks[ulong(id.y) * p.split.bins + bin] = id.x;
}

kernel void vector_histogram_scan(device const uint* histogram [[buffer(0)]],
                                   device const uint* counts [[buffer(1)]],
                                   device const uchar* categorical [[buffer(2)]],
                                   device const VectorCategory* sorted [[buffer(3)]],
                                   device long2* scans [[buffer(4)]],
                                   device const long2* totals [[buffer(5)]],
                                   constant VectorSplitParams& p [[buffer(6)]],
                                   uint3 block [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  uint feature = block.x, node = block.y, target = block.z, count = counts[feature];
  bool cat = categorical[feature], onehot = cat && count < p.split.max_cat_to_onehot;
  auto hist = vector_histogram_offset(node, target, feature, p);
  auto order = (ulong(node) * p.split.cols + feature) * p.split.sort_stride;
  long2 missing = load_hist_int(histogram, hist + p.split.bins * 4);
  if (onehot && p.split.use_parent_sums) {
    long2 local = 0;
    for (uint bin = lane; bin < count; bin += 32) local += load_hist_int(histogram, hist + bin * 4);
    auto present = histogram_broadcast(histogram_prefix(local, lane), 31);
    missing = totals[ulong(node) * p.targets + target] - present;
  }
  for (uint reverse = 0; reverse < 2; ++reverse) {
    long2 preceding = 0;
    for (uint begin = 0; begin < count; begin += 32) {
      uint step = begin + lane;
      uint bin = reverse && !onehot ? count - min(step + 1, count) : step;
      if (step < count && cat && !onehot) bin = sorted[order + bin].bin;
      long2 value = step < count ? load_hist_int(histogram, hist + bin * 4) : long2(0);
      if (onehot) {
        if (reverse) value += missing;
      } else {
        auto prefix = histogram_prefix(value, lane);
        value = preceding + prefix;
        preceding += histogram_broadcast(prefix, 31);
      }
      if (step < count) scans[vector_scan_offset(node, feature, reverse, step, p) + target] = value;
    }
  }
}

inline void vector_children(long2 parent, long2 scanned, bool categorical, bool onehot, uint reverse,
                             thread long2& left, thread long2& right) {
  bool scanned_right = categorical ? (onehot || !reverse) : bool(reverse);
  left = scanned_right ? parent - scanned : scanned;
  right = scanned_right ? scanned : parent - scanned;
}
kernel void evaluate_vector_splits(device const uint* histogram [[buffer(0)]],
                                    device const float2* scales [[buffer(1)]],
                                    device const uint* counts [[buffer(2)]],
                                    device const uchar* categorical [[buffer(3)]],
                                    device const int* monotone [[buffer(4)]],
                                    device const float2* bounds [[buffer(5)]],
                                    device const uchar* allowed [[buffer(6)]],
                                    device const long2* totals [[buffer(7)]],
                                    device const long2* scans [[buffer(8)]],
                                    device VectorSplitChoice* output [[buffer(9)]],
                                    constant VectorSplitParams& p [[buffer(10)]],
                                    uint3 block [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
  uint feature = block.x, node = block.y, count = counts[feature];
  bool cat = categorical[feature], onehot = cat && count < p.split.max_cat_to_onehot;
  uint limit = cat && !onehot ? min(count, p.split.max_cat_threshold) - uint(count != 0) : count;
  uint begin = p.bin_tile ? block.z * p.bin_tile : 0;
  uint end = p.bin_tile ? min(limit, begin + p.bin_tile) : limit;
  auto destination = (ulong(node) * p.split.cols + feature) * max(p.bin_tiles, 1u) + block.z;
  VectorSplitChoice best{-INFINITY, 0, 0, 0};
  if (begin >= end || !allowed[ulong(node) * p.split.cols + feature]) {
    if (!lane) output[destination] = best;
    return;
  }
  bool missing = false;
  auto parent_gain = sr(0);
  for (uint target = 0; target < p.targets; ++target) {
    auto index = ulong(node) * p.targets + target;
    auto parent = histogram_sum(totals[index], scales[target]);
    float weight = vector_weight(parent, p.split, bounds[index]);
    parent_gain = parent_gain + vector_gain_given_weight(parent, weight, p.split);
    if (!cat) {
      auto present = count ? scans[vector_scan_offset(node, feature, 0, count - 1, p) + target] : long2(0);
      auto absent = totals[index] - present;
      missing = missing || absent.x != 0 || absent.y != 0;
    }
  }
  uint order = 0xffffffffu;
  for (uint reverse = 0; reverse < 2; ++reverse) {
    if (reverse && !cat && !missing) continue;
    for (uint step = begin + lane; step < end; step += 32) {
      auto gain = sr(0), left_h = sr(0), right_h = sr(0);
      auto scan = vector_scan_offset(node, feature, reverse, step, p);
      for (uint target = 0; target < p.targets; ++target) {
        auto index = ulong(node) * p.targets + target;
        long2 li, ri;
        vector_children(totals[index], scans[scan + target], cat, onehot, reverse, li, ri);
        auto left = histogram_sum(li, scales[target]), right = histogram_sum(ri, scales[target]);
        left_h = left_h + left.weight; right_h = right_h + right.weight;
        if (p.has_monotone) {
          auto weight = vector_split_weights(left, right, bounds[index], monotone[feature], p.split);
          gain = gain + vector_gain_given_weight(left, weight.x, p.split);
          gain = gain + vector_gain_given_weight(right, weight.y, p.split);
        } else {
          gain = gain + vector_optimal_gain(left, p.split);
          gain = gain + vector_optimal_gain(right, p.split);
        }
      }
      auto targets = ranking_integer(long(p.targets));
      if (!survival_less(sr(0), left_h) || !survival_less(sr(0), right_h) ||
          survival_less(left_h / targets, sr(p.split.min_child_weight)) ||
          survival_less(right_h / targets, sr(p.split.min_child_weight))) continue;
      float score = extended_result(gain - parent_gain);
      uint position = onehot ? step * 2 + reverse : reverse * count + step;
      if (isfinite(score) && (score > best.gain || (score == best.gain && position < order))) {
        best = VectorSplitChoice{score, step, reverse, 1}; order = position;
      }
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) output[destination] = best;
}

// Tiles score the existing integer scans independently. Reduce using the same
// forward/reverse candidate order as the untiled feature evaluator.
kernel void reduce_vector_split_tiles(device const VectorSplitChoice* tiles [[buffer(0)]],
                                       device const uint* counts [[buffer(1)]],
                                       device const uchar* categorical [[buffer(2)]],
                                       device VectorSplitChoice* output [[buffer(3)]],
                                       constant VectorSplitParams& p [[buffer(4)]],
                                       uint2 block [[threadgroup_position_in_grid]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  uint feature = block.x, node = block.y, count = counts[feature];
  bool onehot = categorical[feature] && count < p.split.max_cat_to_onehot;
  auto index = ulong(node) * p.split.cols + feature;
  VectorSplitChoice best{-INFINITY, 0, 0, 0};
  uint order = 0xffffffffu;
  for (uint tile = lane; tile < p.bin_tiles; tile += 32) {
    auto candidate = tiles[index * p.bin_tiles + tile];
    if (!candidate.valid) continue;
    uint position = onehot ? candidate.bin * 2 + candidate.reverse : candidate.reverse * count + candidate.bin;
    if (candidate.gain > best.gain || (candidate.gain == best.gain && position < order)) {
      best = candidate; order = position;
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) output[index] = best;
}

kernel void select_vector_splits(device const VectorSplitChoice* candidates [[buffer(0)]],
                                  device const float2* scales [[buffer(1)]],
                                  device const uint* counts [[buffer(2)]],
                                  device const uchar* categorical [[buffer(3)]],
                                  device const int* monotone [[buffer(4)]],
                                  device const float2* bounds [[buffer(5)]],
                                  device const long2* totals [[buffer(6)]],
                                  device const long2* scans [[buffer(7)]],
                                  device const float* cuts [[buffer(8)]],
                                  device const float* minima [[buffer(9)]],
                                  device const uint* ranks [[buffer(10)]],
                                  device Split* output [[buffer(11)]],
                                  device VectorWeight* weights [[buffer(12)]],
                                  device uint* categories [[buffer(13)]],
                                  constant VectorSplitParams& p [[buffer(14)]], uint node [[thread_position_in_grid]]) {
  if (node >= p.split.nodes) return;
  auto best = candidates[ulong(node) * p.split.cols];
  uint feature = 0;
  for (uint f = 1; f < p.split.cols; ++f) {
    auto other = candidates[ulong(node) * p.split.cols + f];
    if (other.gain > best.gain) { best = other; feature = f; }
  }
  uint count = counts[feature];
  bool cat = categorical[feature], onehot = cat && count < p.split.max_cat_to_onehot;
  Split result{-INFINITY, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0};
  if (best.valid) {
    result.gain = best.gain; result.feature = feature;
    result.default_left = cat ? !best.reverse : best.reverse;
    result.split_type = uint(cat);
    if (cat) {
      result.threshold = as_type<float>(1u);
      auto begin = (ulong(node) * p.split.cols + feature) * p.split.bins;
      result.category_begin = uint(begin);
      if (onehot) {
        result.category_count = 1;
        categories[begin] = uint(cuts[ulong(feature) * p.split.bins + best.bin]);
      } else {
        uint partition = best.reverse ? count - best.bin - 1 : best.bin + 1;
        uint written = 0;
        for (uint bin = 0; bin < count; ++bin) if (ranks[begin + bin] < partition) {
          uint code = uint(cuts[ulong(feature) * p.split.bins + bin]);
          if (!written || categories[begin + written - 1] != code) categories[begin + written++] = code;
        }
        result.category_count = written;
      }
    } else {
      uint bin = best.reverse ? count - best.bin - 1 : best.bin;
      result.threshold = best.reverse ? (bin == 0 ? minima[feature] : cuts[ulong(feature) * p.split.bins + bin - 1]) :
                                       cuts[ulong(feature) * p.split.bins + bin];
    }
  }
  auto total_h = sr(0), left_h = sr(0), right_h = sr(0);
  for (uint target = 0; target < p.targets; ++target) {
    auto index = ulong(node) * p.targets + target;
    auto parent = histogram_sum(totals[index], scales[target]);
    float pw = vector_weight(parent, p.split, bounds[index]);
    float2 child(pw);
    long2 li = 0, ri = 0;
    total_h = total_h + parent.weight;
    if (best.valid) {
      vector_children(totals[index], scans[vector_scan_offset(node, feature, best.reverse, best.bin, p) + target],
                      cat, onehot, best.reverse, li, ri);
      auto left = histogram_sum(li, scales[target]), right = histogram_sum(ri, scales[target]);
      child = vector_split_weights(left, right, bounds[index], monotone[feature], p.split);
      left_h = left_h + left.weight; right_h = right_h + right.weight;
    }
    weights[index] = VectorWeight{pw, child.x, child.y, extended_result(parent.weight),
                                  IntPair{li.x, li.y}, IntPair{ri.x, ri.y}};
    if (!target) { result.parent_weight = pw; result.left_weight = child.x; result.right_weight = child.y; }
  }
  result.total_h = extended_result(total_h);
  result.left_h = extended_result(left_h); result.right_h = extended_result(right_h);
  output[node] = result;
}

// A projected split gradient controls the shared structure; the original
// value gradient supplies each output's final leaf weight. Its row sampling is
// replayed with the split sampler's probabilities and seed before this pass.
kernel void vector_leaf_sums(device const float* data [[buffer(0)]],
                              device const Node* nodes [[buffer(1)]],
                              device const uint* categories [[buffer(2)]],
                              device const long2* gradient [[buffer(3)]],
                              device atomic_uint* sums [[buffer(4)]],
                              device const float* category_bounds [[buffer(5)]],
                              constant VectorLeafParams& p [[buffer(6)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.targets) return;
  uint index = 0;
  Node node = nodes[index];
  while (node.left >= 0) {
    float value = data[ulong(id.x) * p.split.cols + uint(node.feature)];
    bool numeric_category = node.split_type && p.padding;
    bool left = split_left(value, numeric_category ? category_bounds[node.feature] : node.threshold,
                           node.default_left, numeric_category ? 0 : node.split_type,
                           node.category_begin, node.category_count, categories);
    index = left ? uint(node.left) : uint(node.right);
    node = nodes[index];
  }
  auto pair = gradient[ulong(id.x) * p.targets + id.y];
  auto offset = (ulong(index) * p.targets + id.y) * 4;
  add64(sums + offset, pair.x); add64(sums + offset + 2, pair.y);
}
kernel void vector_leaf_weights(device const Node* nodes [[buffer(0)]],
                                 device const uint* sums [[buffer(1)]],
                                 device const float2* scales [[buffer(2)]],
                                 device float* weights [[buffer(3)]],
                                 constant VectorLeafParams& p [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.split.nodes || id.y >= p.targets || nodes[id.x].left >= 0) return;
  auto value = histogram_sum(load_hist_int(sums, (ulong(id.x) * p.targets + id.y) * 4), scales[id.y]);
  weights[ulong(nodes[id.x].right) * p.targets + id.y] =
      vector_weight(value, p.split, float2(-INFINITY, INFINITY)) * p.eta;
}
