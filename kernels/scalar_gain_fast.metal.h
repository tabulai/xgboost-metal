#pragma once
// A float32 significand has only 24 bits, so its exact square fits in 48.
// Specialize the common normal-result range using integer arithmetic. Native
// float multiplication is deliberately avoided: MSL permits RTZ or RTNE for
// arithmetic even with fast math disabled. This path explicitly rounds RTNE.
inline ulong scalar_square_fast_bits(float weight, ulong widened_weight) {
  uint magnitude = as_type<uint>(weight) & 0x7fffffffu;
  // 2^-63 <= abs(weight) <= 2^63 guarantees a normal, finite squared result.
  // Everything else, including subnormals, zeros, infinities and NaNs, keeps
  // the original software multiply and float32 rounding without modification.
  if (magnitude < 0x20000000u || magnitude > 0x5f000000u)
    return cox_float_bits(cox_float_result(cox_multiply_bits(widened_weight, widened_weight)));
  uint mantissa = (magnitude & 0x7fffffu) | 0x800000u;
  ulong product = ulong(mantissa) * mantissa;
  uint leading = uint(product >> 47);  // Exact product has 47 or 48 bits.
  uint shift = 23 + leading;
  uint quotient = uint(product >> shift);
  ulong remainder = product & ((1ul << shift) - 1);
  ulong midpoint = 1ul << (shift - 1);
  if (remainder > midpoint || (remainder == midpoint && (quotient & 1))) ++quotient;
  uint exponent = 2 * (magnitude >> 23) - 127 + leading;
  if (quotient & 0x1000000u) { quotient >>= 1; ++exponent; }
  // The guard makes exponent lie in [1,253]. Encode the rounded float32 as
  // binary64 directly, exactly as cox_float_bits does for a normal input.
  return (ulong(exponent + 896) << 52) | (ulong(quotient & 0x7fffffu) << 29);
}

inline float scalar_gain_fast(ulong2 value, float weight, constant SplitParams& p) {
  if ((value.y & kCoxSign) || !(value.y & ~kCoxSign)) return 0;
  auto w = cox_float_bits(weight);
  auto square = scalar_square_fast_bits(weight, w);
  auto curvature = cox_add_bits(value.y, cox_float_bits(p.reg_lambda));
  auto quadratic = cox_multiply_bits(curvature, square);
  auto sum = cox_fma_bits(cox_add_bits(value.x, value.x), w, quadratic);
  auto alpha = cox_float_bits(p.reg_alpha);
  sum = cox_fma_bits(cox_add_bits(alpha, alpha), w & ~kCoxSign, sum);
  return cox_float_result(sum ^ kCoxSign);
}
