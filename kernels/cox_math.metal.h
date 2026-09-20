#include "cox_constants.metal.h"

// The pinned Cox objective mixes exp(float) and exp(double). Its float call
// uses a 128-entry power-of-two table and a quadratic residual on Apple silicon.
// Preserve this rounding independently of the wider exponential below: a
// one-ulp difference in the initial sum can dominate the last risk sets.
inline float cox_platform_expf(float value) {
  if (isnan(value)) return value + value;
  if (value >= 128) return INFINITY;
  if (value <= -128) return 0;
  auto scaled = accurate_multiply(float2(value, 0), kCoxExpInv);
  float nearest = rint(scaled.x);
  auto residual = accurate_add(scaled, float2(-nearest, 0));
  if (residual.x > 0.5f || (residual.x == 0.5f && residual.y > 0)) {
    nearest += 1;
    residual = accurate_add(residual, float2(-1, 0));
  } else if (residual.x < -0.5f || (residual.x == -0.5f && residual.y < 0)) {
    nearest -= 1;
    residual = accurate_add(residual, float2(1, 0));
  }
  int index = int(nearest);
  auto term = accurate_add(accurate_multiply(kCoxExpSecond, residual), kCoxExpFirst);
  term = accurate_multiply(term, residual);
  auto scale = kCoxExpTable[uint(index) & 127];
  auto result = accurate_add(accurate_multiply(term, scale), scale);
  return extended_result(extended_normalize(result, index >> 7));
}

#include "cox_arithmetic.metal.h"

inline int cox_floor_int(ulong value) {
  int exponent;
  ulong mantissa = cox_finite_significand(value, exponent);
  bool negative = value & kCoxSign;
  if (exponent < 0) return negative && mantissa ? -1 : 0;
  uint shift = uint(52 - exponent);
  int result = int(mantissa >> shift);
  bool fraction = (mantissa << (64 - shift)) != 0;
  return negative ? -result - int(fraction) : result;
}
inline ulong cox_double_exp(float input) {
  if (isnan(input)) return 0x7ff8000000000000ul;
  if (input >= 0x1.62e430p9f) return 0x7ff0000000000000ul;
  if (input <= -1024) return 0;
  auto x = cox_float_bits(input);
  auto product = cox_multiply_bits(x, kCoxDoubleCoefficients[0]);
  int n = cox_floor_int(product);
  auto residual = cox_add_bits(product, cox_float_bits(float(n)) ^ kCoxSign);
  auto error = cox_fma_bits(x ^ kCoxSign, kCoxDoubleCoefficients[0], product);
  auto correction = cox_fma_bits(x, kCoxDoubleCoefficients[1], error ^ kCoxSign);
  auto table = kCoxDoubleTable[uint(n) & 127];
  auto a = cox_add_bits(residual, kCoxDoubleCoefficients[2]);
  auto b = cox_add_bits(residual, kCoxDoubleCoefficients[3]);
  correction = cox_add_bits(table.y, correction);
  residual = cox_add_bits(residual, correction);
  a = cox_fma_bits(residual, a, kCoxDoubleCoefficients[4]);
  b = cox_fma_bits(residual, b, kCoxDoubleCoefficients[5]);
  auto p = cox_multiply_bits(residual, kCoxDoubleCoefficients[6]);
  p = cox_multiply_bits(cox_multiply_bits(p, a), b);
  int exponent = n >> 7;
  if (exponent >= -1022) return cox_scale_bits(cox_fma_bits(p, table.x, table.x), exponent);
  auto scale = cox_scale_bits(table.x, exponent + 1022);
  auto one = cox_float_bits(1);
  auto high = cox_add_bits(scale, one);
  auto low = cox_add_bits(scale, cox_add_bits(high, one ^ kCoxSign) ^ kCoxSign);
  low = cox_fma_bits(scale, p, low);
  return cox_scale_bits(cox_add_bits(cox_add_bits(high, low), one ^ kCoxSign), -1022);
}
