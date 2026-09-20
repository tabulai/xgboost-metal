// Dense scalar CPU roots add the first feature's binary64 histogram bins in
// order. A selected numeric child also retains its ordered prefix, with its
// sibling formed by subtraction from that root. Preserve this history when
// storing root weights, including a root that never expands.
struct ScalarRootRange { ulong offset; uint count, reverse; };
struct ScalarRootCertificate { ulong2 sum, prefix, bound, lattice; uint2 overflow; };
struct ScalarRootValues { ulong2 sum, prefix; };

inline ScalarRootCertificate scalar_root_zero() {
  return ScalarRootCertificate{ulong2(0), ulong2(0), ulong2(0), ulong2(0), uint2(0)};
}
inline ulong2 scalar_root_shuffle(ulong2 value, ushort lane) {
  return as_type<ulong2>(simd_shuffle(as_type<uint4>(value), lane));
}
inline ulong2 scalar_root_shuffle_up(ulong2 value, ushort distance) {
  return as_type<ulong2>(simd_shuffle_up(as_type<uint4>(value), distance));
}
// Return the exact integer magnitude represented after int64 -> binary64 RNE.
inline ulong scalar_root_magnitude(long value) {
  ulong bits = as_type<ulong>(value);
  ulong magnitude = value < 0 ? 0ul - bits : bits;
  uint width = 64 - clz(magnitude);
  if (width <= 53) return magnitude;
  uint shift = width - 53;
  ulong whole = magnitude >> shift, tail = magnitude & ((1ul << shift) - 1);
  ulong midpoint = 1ul << (shift - 1);
  return (whole + ulong(tail > midpoint || (tail == midpoint && (whole & 1)))) << shift;
}
inline ScalarRootCertificate scalar_root_certificate(device const uint* histogram, ScalarRootRange range,
                                                       uint capture, uint tid) {
  auto value = scalar_root_zero();
  for (ulong step = tid; step < range.count; step += 256) {
    ulong bin = range.reverse ? range.count - step - 1 : step;
    auto pair = load_hist_int(histogram, range.offset + 4 * bin);
    ulong2 magnitude(scalar_root_magnitude(pair.x), scalar_root_magnitude(pair.y));
    ulong2 rounded(pair.x < 0 ? 0ul - magnitude.x : magnitude.x,
                     pair.y < 0 ? 0ul - magnitude.y : magnitude.y);
    value.sum += rounded;
    if (step < capture) value.prefix += rounded;
    auto previous = value.bound;
    value.bound += magnitude;
    value.overflow += uint2(value.bound < previous);
    value.lattice |= magnitude;
  }
  return value;
}
inline ScalarRootCertificate scalar_root_reduce(ScalarRootCertificate value, uint lane) {
  for (ushort distance = 1; distance < 32; distance *= 2) {
    ScalarRootCertificate previous{scalar_root_shuffle_up(value.sum, distance),
        scalar_root_shuffle_up(value.prefix, distance), scalar_root_shuffle_up(value.bound, distance),
        scalar_root_shuffle_up(value.lattice, distance), simd_shuffle_up(value.overflow, distance)};
    if (lane >= distance) {
      value.sum += previous.sum;
      value.prefix += previous.prefix;
      auto before = value.bound;
      value.bound += previous.bound;
      value.overflow += previous.overflow + uint2(value.bound < before);
      value.lattice |= previous.lattice;
    }
  }
  return ScalarRootCertificate{scalar_root_shuffle(value.sum, 31), scalar_root_shuffle(value.prefix, 31),
      scalar_root_shuffle(value.bound, 31), scalar_root_shuffle(value.lattice, 31), simd_shuffle(value.overflow, 31)};
}
inline bool scalar_root_exact(ulong bound, ulong lattice, uint overflow) {
  if (overflow || bound >= kCoxSign) return false;
  if (!lattice) return true;
  uint trailing = 63 - clz(lattice & (0ul - lattice));
  // All rounded bins lie on this power-of-two lattice. If their absolute
  // bound fits in 53 bits on that lattice, every partial sum is exact.
  return (64 - clz(bound)) - trailing <= 53;
}
inline ScalarRootValues scalar_root_ordered(device const uint* histogram, ScalarRootRange range,
                                              float2 scale, uint capture, ScalarRootCertificate proof, uint lane) {
  bool2 exact(scalar_root_exact(proof.bound.x, proof.lattice.x, proof.overflow.x),
               scalar_root_exact(proof.bound.y, proof.lattice.y, proof.overflow.y));
  ulong2 sum(split_stat_value_bits(as_type<long>(proof.sum.x), scale.x),
               split_stat_value_bits(as_type<long>(proof.sum.y), scale.y));
  ulong2 prefix(split_stat_value_bits(as_type<long>(proof.prefix.x), scale.x),
                  split_stat_value_bits(as_type<long>(proof.prefix.y), scale.y));
  if (all(exact)) return ScalarRootValues{sum, prefix};
  ulong2 carry = 0;
  for (ulong begin = 0; begin < range.count; begin += 32) {
    ulong step = begin + lane;
    long2 pair = 0;
    if (step < range.count) {
      ulong bin = range.reverse ? range.count - step - 1 : step;
      pair = load_hist_int(histogram, range.offset + 4 * bin);
    }
    ulong g, h;
    uint active = min(32ul, range.count - begin);
    ulong gx = exact.x ? 0 : cox_ordered_warp(split_stat_value_bits(pair.x, scale.x), 0,
                                             carry.x, lane, active, g);
    ulong hx = exact.y ? 0 : cox_ordered_warp(split_stat_value_bits(pair.y, scale.y), 0,
                                             carry.y, lane, active, h);
    if (!exact.x) carry.x = g;
    if (!exact.y) carry.y = h;
    if (capture > begin && capture <= begin + active) {
      if (!exact.x) prefix.x = cox_shuffle_bits(gx, ushort(capture - begin - 1));
      if (!exact.y) prefix.y = cox_shuffle_bits(hx, ushort(capture - begin - 1));
    }
  }
  if (!exact.x) sum.x = carry.x;
  if (!exact.y) sum.y = carry.y;
  return ScalarRootValues{sum, prefix};
}

kernel void select_dense_root_split(device const Split* candidates [[buffer(0)]],
                                      device Split* best [[buffer(1)]],
                                      device const float2* scales [[buffer(2)]],
                                      device const float2* bounds [[buffer(3)]],
                                      device const uint* histogram [[buffer(4)]],
                                      device const uint* counts [[buffer(5)]],
                                      device const float* cuts [[buffer(6)]],
                                      device const float* minima [[buffer(7)]],
                                      constant SplitParams& p [[buffer(8)]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup ScalarRootCertificate warps[16];
  uint lane = tid & 31, warp = tid / 32;
  float gain = -INFINITY;
  uint feature = 0xffffffffu;
  for (uint f = lane; f < p.cols; f += 32) {
    float other = candidates[f].gain;
    if (other > gain || (other == gain && f < feature)) { gain = other; feature = f; }
  }
  float maximum = simd_max(gain);
  feature = simd_min(gain == maximum ? feature : 0xffffffffu);
  Split result = candidates[feature];
  bool numeric = result.feature >= 0 && !result.split_type;
  ScalarRootRange root{0, counts[0], 0}, selected{0, 0, 0};
  uint capture = 0;
  bool separate = false;
  if (numeric) {
    uint count = counts[feature], lo = 0, hi = count;
    auto column = cuts + ulong(feature) * p.bins;
    while (lo < hi) {
      uint middle = lo + (hi - lo) / 2;
      if (fp32_less(column[middle], result.threshold)) lo = middle + 1; else hi = middle;
    }
    uint steps = lo + 1;
    if (result.default_left) {
      uint bin = fp32_equal(result.threshold, minima[feature]) ? 0 : lo + 1;
      steps = count - bin;
    }
    if (feature == 0 && !result.default_left) capture = steps;
    else {
      separate = true;
      selected = ScalarRootRange{(ulong(feature) * (p.bins + 1) +
                                   (result.default_left ? count - steps : 0)) * 4, steps, result.default_left};
    }
  }
  auto root_proof = scalar_root_reduce(scalar_root_certificate(histogram, root, capture, tid), lane);
  if (lane == 31) warps[warp] = root_proof;
  if (separate) {
    auto proof = scalar_root_reduce(scalar_root_certificate(histogram, selected, 0, tid), lane);
    if (lane == 31) warps[8 + warp] = proof;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp) return;
  root_proof = scalar_root_reduce(lane < 8 ? warps[lane] : scalar_root_zero(), lane);
  auto scale = scales[p.group];
  auto values = scalar_root_ordered(histogram, root, scale, capture, root_proof, lane);
  if (!lane) {
    result.parent_weight = split_parent_weight(values.sum, p.reg_alpha, p.reg_lambda, p.max_delta_step, bounds[0]);
    result.total_g = cox_float_result(values.sum.x);
    result.total_h = cox_float_result(values.sum.y);
  }
  if (numeric) {
    auto child = values.prefix;
    if (separate) {
      auto proof = scalar_root_reduce(lane < 8 ? warps[8 + lane] : scalar_root_zero(), lane);
      child = scalar_root_ordered(histogram, selected, scale, 0, proof, lane).sum;
    }
    if (!lane) {
      ulong2 other(cox_add_bits(values.sum.x, child.x ^ kCoxSign),
                     cox_add_bits(values.sum.y, child.y ^ kCoxSign));
      ulong2 applied(cox_add_bits(child.x, other.x), cox_add_bits(child.y, other.y));
      result.applied_parent_weight = split_parent_weight(applied, p.reg_alpha, p.reg_lambda, p.max_delta_step, bounds[0]);
      result.applied_parent_hessian = cox_float_result(applied.y);
    }
  } else if (!lane && result.feature >= 0) {
    auto sums = split_parent_sums(long2(result.left_sum.g, result.left_sum.h),
                                  long2(result.right_sum.g, result.right_sum.h), scale);
    result.applied_parent_weight = split_parent_weight(sums, p.reg_alpha, p.reg_lambda, p.max_delta_step, bounds[0]);
    result.applied_parent_hessian = cox_float_result(sums.y);
  }
  if (!lane) best[0] = result;
}
