// Scalar CPU scoring converts histogram statistics to binary64, stores each
// weight and its square in float32, then rounds each child gain before adding.
// Keep those observable rounding points on the GPU. The pinned Apple CPU build
// contracts the linear and L1 terms into binary64 multiply-adds.
inline ulong2 scalar_histogram_sum(long2 value, float2 scale) {
  return ulong2(split_stat_value_bits(value.x, scale.x), split_stat_value_bits(value.y, scale.y));
}
// Scalar scoring needs RN32(RN64(a / b)), not the full binary64 quotient.
// For a normalized N/D, retain p <= 23 fraction bits and remainder R. A float32
// midpoint has an even binary64 significand; its complete binary64 rounding
// cell is abs(2R-D) <= floor(D / 2^(52-p)). Treat that cell as a float32 tie.
// This preserves double rounding with at most 23 restoring steps instead of 55.
inline bool scalar_divide_float(ulong a, ulong b, thread float& result) {
  uint ae = uint((a >> 52) & 2047), be = uint((b >> 52) & 2047);
  if (ae == 2047 || be == 2047 || !(a & ~kCoxSign) || !(b & ~kCoxSign)) return false;
  int ax, bx;
  ulong numerator = cox_finite_significand(a, ax);
  ulong denominator = cox_finite_significand(b, bx);
  int exponent = ax - bx;
  if (numerator < denominator) { numerator <<= 1; --exponent; }
  uint sign = uint((a ^ b) >> 32) & 0x80000000u;
  if (exponent > 127) { result = as_type<float>(sign | 0x7f800000u); return true; }
  if (exponent < -150) { result = as_type<float>(sign); return true; }
  if (exponent == -150) {
    // Any positive gap from the zero/subnormal midpoint is >= 1/D > 2^-53,
    // so binary64 rounding reaches that midpoint only when N == D.
    result = as_type<float>(sign | uint(numerator != denominator));
    return true;
  }
  uint precision = uint(min(23, exponent + 149));
  ulong remainder = numerator - denominator;
  uint quotient = 1;
  #pragma clang loop unroll_count(4)
  for (uint bit = 0; bit < precision; ++bit) {
    remainder <<= 1;
    bool selected = remainder >= denominator;
    if (selected) remainder -= denominator;
    quotient = (quotient << 1) | uint(selected);
  }
  ulong twice = remainder << 1;
  ulong distance = twice >= denominator ? twice - denominator : denominator - twice;
  bool tie = distance <= (denominator >> (52 - precision));
  if (tie ? bool(quotient & 1) : twice > denominator) ++quotient;
  uint bits;
  if (exponent < -126) {
    // Subnormal encodings also carry directly into the smallest normal.
    bits = quotient;
  } else {
    if (quotient & 0x1000000u) { quotient >>= 1; ++exponent; }
    bits = exponent > 127 ? 0x7f800000u :
        (uint(exponent + 127) << 23) | (quotient & 0x7fffffu);
  }
  result = as_type<float>(sign | bits);
  return true;
}
inline float scalar_weight(ulong2 sums, constant SplitParams& p, float2 bounds) {
  float weight = 0;
  if (!(sums.y & kCoxSign) && (sums.y & ~kCoxSign)) {
    ulong alpha = cox_float_bits(p.reg_alpha) & ~kCoxSign;
    ulong gradient = sums.x;
    if ((gradient & ~kCoxSign) <= alpha) gradient = 0;
    else gradient = cox_add_bits(gradient, gradient & kCoxSign ? alpha : alpha ^ kCoxSign);
    ulong denominator = cox_add_bits(sums.y, cox_float_bits(p.reg_lambda));
    uint limit = as_type<uint>(p.max_delta_step) & 0x7fffffffu;
    bool finite_limit = (limit & 0x7f800000u) != 0x7f800000u;
    if (finite_limit && scalar_divide_float(gradient ^ kCoxSign, denominator, weight)) {
      uint bits = as_type<uint>(weight);
      // A clamp to a finite float32 bound commutes with rounding.
      if (limit && (bits & 0x7fffffffu) > limit)
        weight = as_type<float>((bits & 0x80000000u) | limit);
    } else {
      // Keep the existing zero/nonfinite division and NaN clamp behavior.
      ulong result = cox_divide_bits(gradient ^ kCoxSign, denominator);
      ulong bound = cox_float_bits(p.max_delta_step) & ~kCoxSign;
      if (bound && (result & ~kCoxSign) > bound) result = (result & kCoxSign) | bound;
      weight = cox_float_result(result);
    }
  }
  if (fp32_less(weight, bounds.x)) weight = bounds.x;
  if (fp32_less(bounds.y, weight)) weight = bounds.y;
  return weight;
}
inline float scalar_float_add(float a, float b) {
  return cox_float_result(cox_add_bits(cox_float_bits(a), cox_float_bits(b)));
}
inline bool scalar_valid_hessian(ulong value, float minimum) {
  // The nonnegative parameter also accepts -0.0; its sign must not affect
  // the unsigned ordering of positive binary64 Hessians.
  return !(value & kCoxSign) && (value & ~kCoxSign) && value >= (cox_float_bits(minimum) & ~kCoxSign);
}
inline float scalar_gain(ulong2 value, float weight, constant SplitParams& p) {
  if ((value.y & kCoxSign) || !(value.y & ~kCoxSign)) return 0;
  auto w = cox_float_bits(weight);
  // Software float32 rounding also preserves subnormal squares, which ordinary
  // Metal multiplication would flush. Float32 operands multiply exactly in f64.
  auto square = cox_float_bits(cox_float_result(cox_multiply_bits(w, w)));
  auto curvature = cox_add_bits(value.y, cox_float_bits(p.reg_lambda));
  auto quadratic = cox_multiply_bits(curvature, square);
  auto sum = cox_fma_bits(cox_add_bits(value.x, value.x), w, quadratic);
  auto alpha = cox_float_bits(p.reg_alpha);
  sum = cox_fma_bits(cox_add_bits(alpha, alpha), w & ~kCoxSign, sum);
  return cox_float_result(sum ^ kCoxSign);
}
