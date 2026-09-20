// Survival losses need the reference's double range, particularly for censored
// tail probabilities. Keep a two-float significand and an explicit exponent.
// These operations also retain double overflow/underflow boundaries used by the
// reference's AFT limiting-gradient rules; they do not introduce a CPU fallback.
using SurvivalReal = Extended;
inline SurvivalReal sr(float value) { return extended_float(value); }
inline SurvivalReal sr(float2 value) { return extended_normalize(value); }
inline SurvivalReal survival_range(SurvivalReal value) {
  if (!isfinite(value.significand.x) || value.significand.x == 0) return value;
  if (value.exponent > 1024) return sr(copysign(INFINITY, value.significand.x));
  if (value.exponent > -1022) return value;
  if (value.exponent < -1074) return sr(copysign(0.0f, value.significand.x));
  float2 scaled = ldexp(value.significand, int2(value.exponent + 1074));
  float integral = rint(scaled.x);
  float difference = scaled.x - integral;
  if (fabs(difference) == 0.5f && scaled.y * difference > 0) integral += copysign(1.0f, difference);
  return extended_normalize(float2(integral, 0), -1074);
}
inline SurvivalReal operator+(SurvivalReal a, SurvivalReal b) {
  if (!isfinite(a.significand.x) || !isfinite(b.significand.x)) return sr(a.significand.x + b.significand.x);
  return survival_range(extended_add(a, b));
}
inline SurvivalReal operator-(SurvivalReal a) { return extended_negate(a); }
inline SurvivalReal operator-(SurvivalReal a, SurvivalReal b) { return a + -b; }
inline SurvivalReal operator*(SurvivalReal a, SurvivalReal b) {
  if (!isfinite(a.significand.x) || !isfinite(b.significand.x)) return sr(a.significand.x * b.significand.x);
  return survival_range(extended_multiply(a, b));
}
inline SurvivalReal operator/(SurvivalReal a, SurvivalReal b) {
  if (!isfinite(a.significand.x) || !isfinite(b.significand.x) || b.significand.x == 0)
    return sr(a.significand.x / b.significand.x);
  return survival_range(extended_divide(a, b));
}
inline bool survival_less(SurvivalReal a, SurvivalReal b) {
  if (!isfinite(a.significand.x) || !isfinite(b.significand.x)) return a.significand.x < b.significand.x;
  return extended_greater(b, a);
}
inline SurvivalReal survival_abs(SurvivalReal value) { return value.significand.x < 0 ? -value : value; }
inline SurvivalReal survival_round_double(SurvivalReal value) {
  if (!isfinite(value.significand.x) || value.significand.x == 0) return value;
  int shift = 53 + int(fabs(value.significand.x) == 0.5f && value.significand.x * value.significand.y < 0);
  value.significand.y = ldexp(rint(ldexp(value.significand.y, shift)), -shift);
  return value;
}
inline SurvivalReal survival_exp(SurvivalReal value) {
  float estimate = extended_result(value);
  if (isnan(estimate)) return sr(NAN);
  if (estimate > 710) return sr(INFINITY);
  if (estimate < -746) return sr(0);
  int exponent = int(rint(estimate * 0x1.715476p0f));
  float2 argument = ldexp(value.significand, int2(value.exponent));
  float2 residual = accurate_add(argument, -accurate_multiply(kExtendedLn2, float2(float(exponent), 0)));
  float2 polynomial = kExtendedExpCoefficients[12];
  for (int i = 11; i >= 0; --i)
    polynomial = accurate_add(kExtendedExpCoefficients[i], accurate_multiply(polynomial, residual));
  return survival_range(extended_normalize(polynomial, exponent));
}
inline SurvivalReal survival_log(SurvivalReal value) {
  if (isnan(value.significand.x) || value.significand.x < 0) return sr(NAN);
  if (value.significand.x == 0) return sr(-INFINITY);
  if (isinf(value.significand.x)) return sr(INFINITY);
  float2 t = accurate_divide(accurate_add(value.significand, float2(-1, 0)),
                             accurate_add(value.significand, float2(1, 0)));
  float2 t2 = accurate_multiply(t, t), polynomial = kExtendedLogCoefficients[14];
  for (int i = 13; i >= 0; --i)
    polynomial = accurate_add(kExtendedLogCoefficients[i], accurate_multiply(polynomial, t2));
  float2 logarithm = accurate_multiply(float2(2, 0), accurate_multiply(t, polynomial));
  return sr(accurate_add(logarithm, accurate_multiply(kExtendedLn2, float2(float(value.exponent), 0))));
}

struct SurvivalSum { SurvivalReal value, weight; };
static_assert(sizeof(SurvivalSum) == kSurvivalSumBytes, "Survival reduction buffer layout");
inline SurvivalSum survival_zero() { return SurvivalSum{sr(0), sr(0)}; }
inline SurvivalSum operator+(SurvivalSum a, SurvivalSum b) { return SurvivalSum{a.value + b.value, a.weight + b.weight}; }
inline SurvivalReal survival_shuffle_up(SurvivalReal value, ushort delta) {
  return SurvivalReal{simd_shuffle_up(value.significand, delta), simd_shuffle_up(value.exponent, delta)};
}
inline SurvivalSum survival_warp_scan(SurvivalSum value, uint lane) {
  for (ushort delta = 1; delta < 32; delta *= 2) {
    SurvivalSum previous{survival_shuffle_up(value.value, delta), survival_shuffle_up(value.weight, delta)};
    if (lane >= delta) value = previous + value;
  }
  return value;
}
kernel void survival_scan_tiles(device const SurvivalSum* input [[buffer(0)]],
                                 device SurvivalSum* output [[buffer(1)]],
                                 device SurvivalSum* totals [[buffer(2)]],
                                 constant SurvivalParams& p [[buffer(3)]],
                                 uint block [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum warps[8];
  ulong row = ulong(block) * 256 + tid;
  uint lane = tid & 31, warp = tid / 32;
  auto value = survival_warp_scan(row < p.rows ? input[row] : survival_zero(), lane);
  if (lane == 31) warps[warp] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0) {
    auto total = survival_warp_scan(lane < 8 ? warps[lane] : survival_zero(), lane);
    if (lane < 8) warps[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp) value = warps[warp - 1] + value;
  if (row < p.rows) output[row] = value;
  if (tid == 255) totals[block] = value;
}
kernel void survival_scan_offsets(device SurvivalSum* output [[buffer(0)]],
                                   device const SurvivalSum* totals [[buffer(1)]],
                                   constant SurvivalParams& p [[buffer(2)]],
                                   uint row [[thread_position_in_grid]]) {
  if (row >= p.rows || row < 256) return;
  output[row] = totals[row / 256 - 1] + output[row];
}
inline SurvivalSum survival_block_reduce(SurvivalSum value, threadgroup SurvivalSum* values, uint tid) {
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (tid < delta) values[tid] = values[tid] + values[tid + delta];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return values[0];
}
kernel void survival_reduce(device const SurvivalSum* input [[buffer(0)]],
                             device SurvivalSum* output [[buffer(1)]],
                             constant SurvivalParams& p [[buffer(2)]],
                             uint block [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum values[256];
  ulong row = ulong(block) * 256 + tid;
  auto total = survival_block_reduce(row < p.rows ? input[row] : survival_zero(), values, tid);
  if (!tid) output[block] = total;
}
kernel void survival_metric_finish(device const SurvivalSum* input [[buffer(0)]],
                                    device MetricRatio* output [[buffer(1)]],
                                    constant SurvivalParams& p [[buffer(2)]],
                                    uint row [[thread_position_in_grid]]) {
  if (row) return;
  auto value = input[0].weight.significand.x == 0 && p.mode != 2 ? input[0].value : input[0].value / input[0].weight;
  *output = MetricRatio{value.significand.x, value.significand.y, value.exponent, 1};
}

// Stable time ordering is observable for tied Cox labels. Sorting by absolute
// float bits and then original row index also distinguishes padding without NaNs.
inline bool survival_key_less(uint2 a, uint2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }
kernel void survival_order_tiles(device const float* labels [[buffer(0)]],
                                  device uint* output [[buffer(1)]],
                                  constant SortParams& p [[buffer(2)]],
                                  uint block [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint2 values[256];
  ulong row = ulong(block) * 256 + tid;
  values[tid] = uint2(row < p.rows ? as_type<uint>(labels[row]) & 0x7fffffffu : 0xffffffffu, uint(row));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) for (uint j = k >> 1; j; j >>= 1) {
    uint2 a = values[tid], b = values[tid ^ j];
    bool take_min = ((tid & k) == 0) == ((tid & j) == 0);
    uint2 next = survival_key_less(a, b) == take_min ? a : b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < p.rows) output[row] = values[tid].y;
}
kernel void survival_order_merge(device const float* labels [[buffer(0)]],
                                  device const uint* source [[buffer(1)]],
                                  device uint* output [[buffer(2)]],
                                  constant SortParams& p [[buffer(3)]],
                                  uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  ulong begin = (ulong(row) / (2ul * p.width)) * (2ul * p.width);
  ulong middle = min(begin + p.width, ulong(p.rows)), end = min(begin + 2ul * p.width, ulong(p.rows));
  bool left = row < middle;
  ulong own = left ? begin : middle, other = left ? middle : begin;
  ulong lo = other, hi = left ? end : middle;
  uint index = source[row];
  uint2 key(as_type<uint>(labels[index]) & 0x7fffffffu, index);
  while (lo < hi) {
    ulong mid = lo + (hi - lo) / 2;
    uint next = source[mid];
    if (survival_key_less(uint2(as_type<uint>(labels[next]) & 0x7fffffffu, next), key)) lo = mid + 1;
    else hi = mid;
  }
  output[begin + row - own + lo - other] = index;
}
#include "cox_math.metal.h"

#include "cox.metal.h"

kernel void cox_metric_inputs(device const float* prediction [[buffer(0)]],
                               device const uint* order [[buffer(1)]],
                               device SurvivalSum* output [[buffer(2)]],
                               constant SurvivalParams& p [[buffer(3)]],
                               uint row [[thread_position_in_grid]]) {
  if (row < p.rows) { auto value = sr(prediction[order[row]]); output[row] = SurvivalSum{value, value}; }
}
inline SurvivalReal cox_risk(device const float* labels, device const uint* order,
                              device const SurvivalSum* prefix, uint row, uint rows) {
  uint key = as_type<uint>(labels[order[row]]) & 0x7fffffffu, lo = 0, hi = row;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if ((as_type<uint>(labels[order[mid]]) & 0x7fffffffu) < key) lo = mid + 1;
    else hi = mid;
  }
  return prefix[rows - 1].weight - (lo ? prefix[lo - 1].value : sr(0));
}
kernel void cox_metric_partial(device const float* prediction [[buffer(0)]],
                                device const float* labels [[buffer(1)]],
                                device const uint* order [[buffer(2)]],
                                device const SurvivalSum* prefix [[buffer(3)]],
                                device SurvivalSum* output [[buffer(4)]],
                                constant SurvivalParams& p [[buffer(5)]],
                                uint block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum values[256];
  ulong row = ulong(block) * 256 + tid;
  auto value = survival_zero();
  if (row < p.rows && labels[order[row]] > 0) {
    auto risk = cox_risk(labels, order, prefix, uint(row), p.rows);
    value = SurvivalSum{survival_log(risk) - survival_log(sr(prediction[order[row]])), sr(1)};
  }
  auto total = survival_block_reduce(value, values, tid);
  if (!tid) output[block] = total;
}

#include "survival_wide.metal.h"

struct AFTDistribution { SurvivalReal pdf, cdf, first, second; };
inline SurvivalReal normal_cdf(SurvivalReal z, SurvivalReal pdf) {
  if (isnan(z.significand.x)) return sr(NAN);
  auto a = survival_abs(z);
  if (!survival_less(a, sr(9))) return sr(z.significand.x > 0 ? 1 : 0);
  SurvivalReal tail;
  if (survival_less(a, sr(3))) {
    // Integral from zero: phi(z) * (z + z^3/3 + z^5/(3*5) + ...).
    auto term = a, sum = a, square = a * a;
    for (uint k = 1; k <= 48; ++k) {
      term = term * square / sr(float(2 * k + 1));
      sum = sum + term;
    }
    tail = sr(0.5f) - pdf * sum;
  } else {
    // Laplace's continued fraction for the normal tail (Mills ratio).
    // At z >= 3, 96 terms exceed the two-float significand's precision.
    auto fraction = sr(0);
    for (uint k = 96; k > 0; --k) fraction = sr(float(k)) / (a + fraction);
    tail = pdf / (a + fraction);
  }
  // Match the reference's 0.5 * (1 + erf(z/sqrt(2))), including cancellation
  // when erf rounds to +/-1 at the double-precision boundary.
  auto erf_value = survival_round_double(sr(1) - sr(2) * tail);
  if (z.significand.x < 0) erf_value = -erf_value;
  return sr(0.5f) * survival_round_double(sr(1) + erf_value);
}
inline AFTDistribution aft_distribution(SurvivalReal z, uint kind) {
  auto one = sr(1);
  if (kind == 0) {
    auto square = z * z;
    auto pdf = survival_exp(sr(-0.5f) * square) * sr(float2(0x1.988454p-2f, -0x1.857936p-27f));
    return AFTDistribution{pdf, normal_cdf(z, pdf), -z * pdf, (square - one) * pdf};
  }
  auto w = survival_round_double(survival_exp(z)), square = w * w;
  bool infinite = isinf(w.significand.x), square_infinite = isinf(square.significand.x);
  if (kind == 1) {
    auto denominator = survival_round_double(one + w);
    auto pdf = infinite || square_infinite ? sr(0) : w / (denominator * denominator);
    auto cdf = infinite ? one : survival_round_double(w / denominator);
    auto first = infinite ? sr(0) : pdf * (one - w) / denominator;
    auto second = infinite || square_infinite ? sr(0) : pdf * (square - sr(4) * w + one) / (denominator * denominator);
    return AFTDistribution{pdf, cdf, first, second};
  }
  // In the reference CDF, exp(-w) rounds to double before subtraction from 1.
  // Keeping its tiny low component would change censored gradients near zero.
  auto negative_exp = survival_round_double(survival_exp(-w));
  auto pdf = infinite ? sr(0) : w * negative_exp;
  auto cdf = survival_round_double(one - negative_exp);
  auto first = infinite ? sr(0) : (one - w) * pdf;
  auto second = infinite || square_infinite ? sr(0) : (square - sr(3) * w + one) * pdf;
  return AFTDistribution{pdf, cdf, first, second};
}
inline SurvivalReal aft_gradient_limit(uint distribution, uint censoring, bool positive, SurvivalReal scale) {
  if ((censoring == 1 && !positive) || (censoring == 2 && positive)) return sr(0);
  if (distribution == 0) return sr(positive ? -15 : 15);
  if (distribution == 1) return sr(positive ? -1 : 1) / scale;
  return positive ? sr(-15) : sr(1) / scale;
}
inline SurvivalReal aft_hessian_limit(uint distribution, uint censoring, bool positive, SurvivalReal scale) {
  auto minimum = sr(float2(0x1.cd2b2ap-54f, -0x1.04eec8p-79f));
  if (distribution == 1) return minimum;
  if (distribution == 2) return positive && censoring != 2 ? sr(15) : minimum;
  if ((censoring == 1 && !positive) || (censoring == 2 && positive)) return minimum;
  return sr(1) / (scale * scale);
}
inline float survival_clip(SurvivalReal value, SurvivalReal lower, SurvivalReal upper) {
  if (survival_less(value, lower)) value = lower;
  if (survival_less(upper, value)) value = upper;
  return extended_result(value);
}
struct AFTValue { SurvivalReal loss; Pair gradient; };
inline AFTValue aft_value(float lower, float upper, float prediction, constant SurvivalParams& p) {
  auto scale = sr(p.scale), margin = sr(prediction);
  auto epsilon = sr(float2(0x1.197998p-40f, 0x1.2dea10p-68f));
  SurvivalReal probability, g_numerator, g_denominator, h_numerator, h_denominator;
  uint censoring = 0;
  bool positive;
  if (lower == upper) {
    auto z = (survival_log(sr(lower)) - margin) / scale;
    auto d = aft_distribution(z, p.distribution);
    probability = d.pdf / (scale * sr(lower));
    g_numerator = d.first;
    g_denominator = scale * d.pdf;
    h_numerator = -(d.pdf * d.second - d.first * d.first);
    h_denominator = scale * scale * d.pdf * d.pdf;
    positive = z.significand.x > 0;
  } else {
    censoring = 3;
    AFTDistribution lo{sr(0), sr(0), sr(0), sr(0)}, hi{sr(0), sr(1), sr(0), sr(0)};
    auto zl = sr(0), zu = sr(0);
    if (isinf(upper)) censoring = 1;
    else {
      zu = (survival_log(sr(upper)) - margin) / scale;
      hi = aft_distribution(zu, p.distribution);
    }
    if (lower <= 0) censoring = 2;
    else {
      zl = (survival_log(sr(lower)) - margin) / scale;
      lo = aft_distribution(zl, p.distribution);
    }
    probability = hi.cdf - lo.cdf;
    auto pdf_difference = hi.pdf - lo.pdf;
    auto first_difference = hi.first - lo.first;
    survival_censored_difference(lower, upper, prediction, p, zu, probability, pdf_difference, first_difference);
    g_numerator = pdf_difference;
    g_denominator = scale * probability;
    h_numerator = -(probability * first_difference - pdf_difference * pdf_difference);
    h_denominator = g_denominator * g_denominator;
    positive = zl.significand.x > 0 || zu.significand.x > 0;
  }
  auto gradient = g_numerator / g_denominator, hessian = h_numerator / h_denominator;
  if (survival_less(g_denominator, epsilon) && !isfinite(gradient.significand.x))
    gradient = aft_gradient_limit(p.distribution, censoring, positive, scale);
  if (survival_less(h_denominator, epsilon) && !isfinite(hessian.significand.x))
    hessian = aft_hessian_limit(p.distribution, censoring, positive, scale);
  if (isnan(probability.significand.x) || survival_less(probability, epsilon)) probability = epsilon;
  return AFTValue{-survival_log(probability),
                  Pair{survival_clip(gradient, sr(-15), sr(15)),
                       survival_clip(hessian, sr(float2(0x1.cd2b2ap-54f, -0x1.04eec8p-79f)), sr(15))}};
}
kernel void aft_gradients(device const float* prediction [[buffer(0)]],
                           device const float* lower [[buffer(1)]],
                           device const float* upper [[buffer(2)]],
                           device const float* weights [[buffer(3)]],
                           device Pair* output [[buffer(4)]],
                           constant SurvivalParams& p [[buffer(5)]],
                           uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  auto pair = aft_value(lower[row], upper[row], prediction[row], p).gradient;
  output[row] = Pair{signed_product(pair.g, weights[row]), signed_product(pair.h, weights[row])};
}
kernel void aft_metric_partial(device const float* prediction [[buffer(0)]],
                                device const float* lower [[buffer(1)]],
                                device const float* upper [[buffer(2)]],
                                device const float* weights [[buffer(3)]],
                                device SurvivalSum* output [[buffer(4)]],
                                constant SurvivalParams& p [[buffer(5)]],
                                uint block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum values[256];
  ulong row = ulong(block) * 256 + tid;
  auto value = survival_zero();
  if (row < p.rows) {
    SurvivalReal loss;
    if (p.mode == 1) {
      auto prediction_value = survival_exp(sr(prediction[row]));
      bool valid = !isnan(prediction_value.significand.x) &&
                   !survival_less(prediction_value, sr(lower[row])) && !survival_less(sr(upper[row]), prediction_value);
      loss = sr(float(valid));
    } else loss = aft_value(lower[row], upper[row], prediction[row], p).loss;
    value = SurvivalSum{loss * sr(weights[row]), sr(weights[row])};
  }
  auto total = survival_block_reduce(value, values, tid);
  if (!tid) output[block] = total;
}
