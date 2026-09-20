// An ordered binary64 sum cannot be replaced by an ordinary parallel reduction.
// Inside one binade, however, each addition is an integer translation on the
// accumulator's ULP lattice. Only exact halfway cases depend on its low bit.
// Compose both parity paths per tile, then check the actual carry against the
// interval where every intermediate stays inside that binade. A failed check
// replays the tile with binary64 addition. Estimates only choose the lattice;
// they never determine an accepted result.
struct CoxEstimate { SurvivalSum sum; uint4 reset; };
struct CoxTransform { long2 increment; long lower, upper; int exponent; uint kind; };
static_assert(sizeof(CoxEstimate) == kCoxEstimateBytes, "Cox estimate layout");
static_assert(sizeof(CoxTransform) == kCoxTransformBytes, "Cox transform layout");

inline bool cox_segment_start(device const float* labels, device const uint* order, ulong row) {
  return !row || (as_type<uint>(labels[order[row]]) & 0x7fffffffu) !=
                    (as_type<uint>(labels[order[row - 1]]) & 0x7fffffffu);
}
inline SurvivalReal cox_estimate(ulong value) {
  uint power = uint((value >> 52) & 2047);
  if (power == 2047) return sr((value & kCoxFraction) ? NAN : (value & kCoxSign ? -INFINITY : INFINITY));
  if (!(value & ~kCoxSign)) return sr(value & kCoxSign ? -0.0f : 0.0f);
  int exponent;
  ulong mantissa = cox_finite_significand(value, exponent);
  float high = float(mantissa), low = float(long(mantissa) - long(high));
  auto result = extended_normalize(float2(high, low), exponent - 52);
  return value & kCoxSign ? -result : result;
}
inline CoxEstimate cox_estimate_zero() { return CoxEstimate{survival_zero(), uint4(0)}; }
inline CoxEstimate cox_estimate_join(CoxEstimate a, CoxEstimate b) {
  return CoxEstimate{SurvivalSum{b.reset.x ? b.sum.value : a.sum.value + b.sum.value,
                                 a.sum.weight + b.sum.weight}, a.reset | b.reset};
}
inline CoxEstimate cox_estimate_warp(CoxEstimate value, uint lane) {
  for (ushort delta = 1; delta < 32; delta *= 2) {
    CoxEstimate previous{SurvivalSum{survival_shuffle_up(value.sum.value, delta),
                                     survival_shuffle_up(value.sum.weight, delta)},
                          simd_shuffle_up(value.reset, delta)};
    if (lane >= delta) value = cox_estimate_join(previous, value);
  }
  return value;
}
inline CoxEstimate cox_estimate_tile(CoxEstimate value, threadgroup CoxEstimate* warps, uint tid) {
  uint lane = tid & 31, warp = tid / 32;
  value = cox_estimate_warp(value, lane);
  if (lane == 31) warps[warp] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!warp) {
    auto total = cox_estimate_warp(lane < 8 ? warps[lane] : cox_estimate_zero(), lane);
    if (lane < 8) warps[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return warp ? cox_estimate_join(warps[warp - 1], value) : value;
}
kernel void cox_scan_estimates(device const ulong2* input [[buffer(0)]],
                                device const float* labels [[buffer(1)]],
                                device const uint* order [[buffer(2)]],
                                device CoxEstimate* totals [[buffer(3)]],
                                constant CoxScanParams& p [[buffer(4)]],
                                uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup CoxEstimate warps[8];
  ulong row = ulong(block) * kCoxScanRows + tid;
  auto value = cox_estimate_zero();
  if (row < p.rows) {
    auto bits = input[row];
    value.sum = SurvivalSum{cox_estimate(bits.x), cox_estimate(bits.y)};
    value.reset.x = p.segmented && cox_segment_start(labels, order, row);
  }
  value = cox_estimate_tile(value, warps, tid);
  if (tid == 255) totals[block] = value;
}
kernel void cox_estimate_scan_tiles(device const CoxEstimate* input [[buffer(0)]],
                                    device CoxEstimate* output [[buffer(1)]],
                                    device CoxEstimate* totals [[buffer(2)]],
                                    constant CoxScanParams& p [[buffer(3)]],
                                    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup CoxEstimate warps[8];
  ulong row = ulong(block) * kCoxScanRows + tid;
  auto value = cox_estimate_tile(row < p.rows ? input[row] : cox_estimate_zero(), warps, tid);
  if (row < p.rows) output[row] = value;
  if (tid == 255) totals[block] = value;
}
kernel void cox_estimate_scan_offsets(device CoxEstimate* output [[buffer(0)]],
                                      device const CoxEstimate* totals [[buffer(1)]],
                                      constant CoxScanParams& p [[buffer(2)]], uint row [[thread_position_in_grid]]) {
  if (row >= p.rows || row < kCoxScanRows) return;
  output[row] = cox_estimate_join(totals[row / kCoxScanRows - 1], output[row]);
}

inline bool cox_lattice_increment(ulong value, int exponent, thread long2& result) {
  if (!(value & ~kCoxSign)) { result = 0; return true; }
  if (((value >> 52) & 2047) == 2047) return false;
  int power;
  ulong mantissa = cox_finite_significand(value, power);
  int shift = exponent - power;
  if (shift < -1) return false;  // Keeps 256 composed increments inside int64.
  if (shift <= 0) result = long2(long(mantissa << uint(-shift)));
  else if (shift > 53) result = 0;
  else {
    ulong whole = mantissa >> uint(shift), remainder = mantissa & ((1ul << uint(shift)) - 1);
    ulong halfway = 1ul << uint(shift - 1);
    if (remainder == halfway) result = long2(long(whole + (whole & 1)), long(whole + ((whole & 1) ^ 1)));
    else result = long2(long(whole + ulong(remainder > halfway)));
  }
  if (value & kCoxSign) result = -result;
  return true;
}
inline long cox_shuffle_up_integer(long value, ushort delta) {
  return as_type<long>(simd_shuffle_up(as_type<uint2>(value), delta));
}
inline long2 cox_compose_increments(long2 before, long2 after) {
  return before + long2(after[ulong(before.x) & 1], after[ulong(before.y + 1) & 1]);
}
// With the actual incoming carry known, scan the two parity translations in
// five SIMD steps and verify every proposed prefix. Only a warp that crosses
// a binade boundary (or contains a reset/special value) needs ordered replay.
inline ulong cox_ordered_warp(ulong own, uint reset, ulong initial, uint lane, uint count,
                               thread ulong& carry) {
  if (simd_all(reset || lane >= count)) {
    carry = cox_shuffle_bits(own, ushort(count - 1));
    return own;
  }
  if (simd_all(!reset)) {
    if (simd_all(own == 0)) { carry = cox_add_bits(initial, 0); return carry; }
    uint power = uint((initial >> 52) & 2047);
    bool valid = power && power != 2047;
    long2 increment(0);
    valid = cox_lattice_increment(own, int(power) - 1023, increment) && valid;
    for (ushort delta = 1; delta < 32; delta *= 2) {
      long2 previous(cox_shuffle_up_integer(increment.x, delta), cox_shuffle_up_integer(increment.y, delta));
      if (lane >= delta) increment = cox_compose_increments(previous, increment);
    }
    bool negative = (initial & kCoxSign) != 0;
    long mantissa = long((initial & kCoxFraction) | (1ul << 52));
    if (negative) mantissa = -mantissa;
    long result = mantissa + increment[ulong(mantissa) & 1];
    long lower = negative ? -(1l << 53) + 2 : (1l << 52) + 2;
    long upper = negative ? -(1l << 52) - 2 : (1l << 53) - 2;
    valid = valid && mantissa >= lower && mantissa <= upper && result >= lower && result <= upper;
    if (simd_all(valid)) {
      ulong bits = (initial & kCoxSign) | (ulong(power) << 52) |
                     (ulong(negative ? -result : result) & kCoxFraction);
      carry = cox_shuffle_bits(bits, ushort(count - 1));
      return bits;
    }
  }
  ulong result = 0;
  carry = initial;
  #pragma clang loop unroll_count(4)
  for (ushort j = 0; j < count; ++j) {
    auto value = cox_shuffle_bits(own, j);
    carry = simd_shuffle(reset, j) ? value : cox_add_bits(carry, value);
    if (lane == j) result = carry;
  }
  return result;
}
inline CoxTransform cox_build_transform(device const ulong2* input, device const float* labels,
                                        device const uint* order, uint block, uint channel,
                                        SurvivalReal estimate, constant CoxScanParams& p, uint lane) {
  int exponent = estimate.exponent - 1;
  bool valid = isfinite(estimate.significand.x) && estimate.significand.x != 0 &&
                 exponent >= -1022 && exponent <= 1023;
  bool negative = estimate.significand.x < 0, all_zero = true, has_reset = false;
  long2 path(0, 1);
  long low = 0, high = 0;
  ulong zero_sign = kCoxSign, exact = 0;
  ulong begin = ulong(block) * kCoxScanRows, end = min(begin + kCoxScanRows, ulong(p.rows));
  for (ulong base = begin; base < end; base += 32) {
    ulong row = base + lane;
    ulong own = row < end ? input[row][channel] : 0;
    uint reset = row < end && !channel && p.segmented && cox_segment_start(labels, order, row);
    #pragma clang loop unroll_count(4)
    for (ushort j = 0; j < 32; ++j) {
      if (base + j >= end) break;
      ulong value = cox_shuffle_bits(own, j);
      bool restart = simd_shuffle(reset, j);
      all_zero = all_zero && !(value & ~kCoxSign);
      zero_sign &= value;
      if (restart) { exact = value; has_reset = true; }
      else if (has_reset) exact = cox_add_bits(exact, value);
      if (valid && !has_reset) {
        long2 increment;
        valid = cox_lattice_increment(value, exponent, increment);
        if (valid) {
          path += long2(increment[ulong(path.x) & 1], increment[ulong(path.y) & 1]);
          low = min(low, min(path.x, path.y - 1));
          high = max(high, max(path.x, path.y - 1));
        }
      }
    }
  }
  CoxTransform result{long2(0), 0, -1, exponent, 0};
  if (has_reset) { result.kind = 2; result.increment.x = as_type<long>(exact); }
  else if (all_zero) { result.kind = 3; result.increment.x = as_type<long>(zero_sign); }
  else if (valid) {
    // A two-ULP guard keeps both rounded and unrounded intermediates away from
    // the boundary where the spacing changes. Include both incoming parities.
    long lower = negative ? -(1l << 53) + 2 : (1l << 52) + 2;
    long upper = negative ? -(1l << 52) - 2 : (1l << 53) - 2;
    result.increment = path - long2(0, 1);
    result.lower = lower - low;
    result.upper = upper - high;
    result.kind = 1;
  }
  return result;
}
kernel void cox_scan_transforms(device const ulong2* input [[buffer(0)]],
                                 device const float* labels [[buffer(1)]],
                                 device const uint* order [[buffer(2)]],
                                 device const ulong2* seed [[buffer(3)]],
                                 device const CoxEstimate* estimates [[buffer(4)]],
                                 device CoxTransform* output [[buffer(5)]],
                                 constant CoxScanParams& p [[buffer(6)]],
                                 uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  auto initial = seed[0];
  CoxEstimate start{SurvivalSum{cox_estimate(initial.x), cox_estimate(initial.y)}, uint4(0)};
  if (block) start = cox_estimate_join(start, estimates[block - 1]);
  auto first = cox_build_transform(input, labels, order, block, 0, start.sum.value, p, lane);
  auto second = cox_build_transform(input, labels, order, block, 1, start.sum.weight, p, lane);
  if (!lane) { output[2 * block] = first; output[2 * block + 1] = second; }
}
inline CoxTransform cox_shuffle_transform(CoxTransform value, ushort lane) {
  value.increment.x = as_type<long>(cox_shuffle_bits(as_type<ulong>(value.increment.x), lane));
  value.increment.y = as_type<long>(cox_shuffle_bits(as_type<ulong>(value.increment.y), lane));
  value.lower = as_type<long>(cox_shuffle_bits(as_type<ulong>(value.lower), lane));
  value.upper = as_type<long>(cox_shuffle_bits(as_type<ulong>(value.upper), lane));
  value.exponent = simd_shuffle(value.exponent, lane);
  value.kind = simd_shuffle(value.kind, lane);
  return value;
}
inline bool cox_apply_transform(CoxTransform t, thread ulong& sum) {
  if (t.kind == 2) { sum = as_type<ulong>(t.increment.x); return true; }
  uint power = uint((sum >> 52) & 2047);
  if (t.kind == 3 && power != 2047) {
    if (!(sum & ~kCoxSign)) sum &= as_type<ulong>(t.increment.x);
    return true;
  }
  if (t.kind != 1 || !power || power == 2047 || int(power) - 1023 != t.exponent) return false;
  long mantissa = long((sum & kCoxFraction) | (1ul << 52));
  if (sum & kCoxSign) mantissa = -mantissa;
  if (mantissa < t.lower || mantissa > t.upper) return false;
  mantissa += t.increment[ulong(mantissa) & 1];
  ulong magnitude = ulong(mantissa < 0 ? -mantissa : mantissa);
  sum = (mantissa < 0 ? kCoxSign : 0) | (ulong(power) << 52) | (magnitude & kCoxFraction);
  return true;
}
kernel void cox_scan_carries(device const ulong2* input [[buffer(0)]],
                              device const float* labels [[buffer(1)]],
                              device const uint* order [[buffer(2)]],
                              device const ulong2* seed [[buffer(3)]],
                              device const CoxTransform* transforms [[buffer(4)]],
                              device ulong2* starts [[buffer(5)]],
                              constant CoxScanParams& p [[buffer(6)]], uint lane [[thread_index_in_simdgroup]]) {
  ulong2 sum = seed[0];
  for (ulong base = 0; base < p.blocks; base += 32) {
    bool valid = base + lane < p.blocks;
    CoxTransform empty{long2(0), 0, -1, 0, 0};
    auto first = valid ? transforms[2 * (base + lane)] : empty;
    auto second = valid ? transforms[2 * (base + lane) + 1] : empty;
    for (ushort j = 0; j < 32 && base + j < p.blocks; ++j) {
      if (j == lane) starts[base + j] = sum;
      auto a = sum.x, b = sum.y;
      bool fallback_a = !cox_apply_transform(cox_shuffle_transform(first, j), a);
      bool fallback_b = !cox_apply_transform(cox_shuffle_transform(second, j), b);
      sum = ulong2(a, b);
      if (fallback_a || fallback_b) {
        ulong begin = (base + j) * kCoxScanRows, end = min(begin + kCoxScanRows, ulong(p.rows));
        for (ulong batch = begin; batch < end; batch += 32) {
          ulong row = batch + lane;
          ulong2 own = row < end ? input[row] : ulong2(0);
          uint reset = row < end && p.segmented && cox_segment_start(labels, order, row);
          uint count = uint(min(32ul, end - batch));
          ulong a = sum.x, b = sum.y;
          if (fallback_a) cox_ordered_warp(own.x, reset, sum.x, lane, count, a);
          if (fallback_b) cox_ordered_warp(own.y, 0, sum.y, lane, count, b);
          sum = ulong2(a, b);
        }
      }
    }
  }
  if (!lane) starts[p.blocks] = sum;
}
kernel void cox_scan_replay(device const ulong2* input [[buffer(0)]],
                             device const float* labels [[buffer(1)]],
                             device const uint* order [[buffer(2)]],
                             device const ulong2* starts [[buffer(3)]],
                             device ulong2* output [[buffer(4)]],
                             constant CoxScanParams& p [[buffer(5)]],
                             uint block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ulong2 sum = starts[block];
  ulong begin = ulong(block) * kCoxScanRows, end = min(begin + kCoxScanRows, ulong(p.rows));
  for (ulong base = begin; base < end; base += 32) {
    ulong row = base + lane;
    ulong2 own = row < end ? input[row] : ulong2(0);
    uint reset = row < end && p.segmented && cox_segment_start(labels, order, row);
    uint count = uint(min(32ul, end - base));
    ulong a = sum.x, b = sum.y;
    auto x = cox_ordered_warp(own.x, reset, sum.x, lane, count, a);
    auto y = cox_ordered_warp(own.y, 0, sum.y, lane, count, b);
    sum = ulong2(a, b);
    ulong2 result(x, y);
    if (row < end) output[row] = result;
  }
}
