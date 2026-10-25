// A native float quotient is only a search seed. Accept it exclusively after
// exact 128-bit arithmetic proves Q*D <= N*2^p < (Q+1)*D. A poor seed restores
// through the unchanged scalar_divide_float helper; it never changes a result.
// Normalization supplies D in [2^52,2^53), N in [D,2D), and p in [0,23].
// Thus the target uses at most 77 bits. Bounded seed/correction products stay
// below 2^78; the certified remainder is below D and fits in one 64-bit word.
inline bool scalar_quotient_certified(ulong numerator, ulong denominator, uint precision,
                                       float estimate, thread uint& quotient,
                                       thread ulong& remainder) {
  // Check before conversion: converting NaN, infinity, or an out-of-range float
  // to uint is not a valid way to reject an estimate. No accuracy bound on the
  // hardware division is required, because failed certificates fall back.
  uint upper = (1u << (precision + 1)) + 4u;
  if (!isfinite(estimate) || estimate < 0 || estimate > float(upper)) return false;
  uint candidate = uint(estimate);
  ulong2 divisor(denominator, 0);
  ulong2 target = cox_wide_left(ulong2(numerator, 0), precision);
  ulong2 product = cox_wide_product(ulong(candidate), denominator);
  for (uint correction = 0; correction < 4; ++correction) {
    if (cox_wide_less(target, product)) {
      if (!candidate) return false;
      --candidate;
      product = cox_wide_sub(product, divisor);
    } else {
      ulong2 residual = cox_wide_sub(target, product);
      if (!residual.y && residual.x < denominator) {
        quotient = candidate;
        remainder = residual.x;
        return true;
      }
      ++candidate;
      product = cox_wide_add(product, divisor);
    }
  }
  // A fourth correction can establish the certificate; check its result too.
  if (cox_wide_less(target, product)) return false;
  ulong2 residual = cox_wide_sub(target, product);
  if (residual.y || residual.x >= denominator) return false;
  quotient = candidate;
  remainder = residual.x;
  return true;
}

// The diagnostic specialization permits a forced seed and exposes which route
// completed. The production specialization discards that metadata and always
// computes its own seed: 0=unsupported, 1=early result, 2=certified, 3=restored.
template <bool ForcedSeed>
inline bool scalar_divide_float_fast_impl(ulong a, ulong b, float forced_seed,
                                         thread float& result, thread uint& route) {
  route = 1;
  uint ae = uint((a >> 52) & 2047), be = uint((b >> 52) & 2047);
  if (ae == 2047 || be == 2047 || !(a & ~kCoxSign) || !(b & ~kCoxSign)) {
    route = 0;
    return false;
  }
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
  float estimate = ForcedSeed ? forced_seed :
      (float(numerator) / float(denominator)) * float(1u << precision);
  uint quotient;
  ulong remainder;
  if (!scalar_quotient_certified(numerator, denominator, precision, estimate,
                                 quotient, remainder)) {
    route = 3;
    return scalar_divide_float(a, b, result);
  }
  route = 2;
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
inline bool scalar_divide_float_fast(ulong a, ulong b, thread float& result) {
  uint route;
  return scalar_divide_float_fast_impl<false>(a, b, 0, result, route);
}

inline float scalar_weight_fast(ulong2 sums, constant SplitParams& p, float2 bounds) {
  float weight = 0;
  if (!(sums.y & kCoxSign) && (sums.y & ~kCoxSign)) {
    ulong alpha = cox_float_bits(p.reg_alpha) & ~kCoxSign;
    ulong gradient = sums.x;
    if ((gradient & ~kCoxSign) <= alpha) gradient = 0;
    else gradient = cox_add_bits(gradient, gradient & kCoxSign ? alpha : alpha ^ kCoxSign);
    ulong denominator = cox_add_bits(sums.y, cox_float_bits(p.reg_lambda));
    uint limit = as_type<uint>(p.max_delta_step) & 0x7fffffffu;
    bool finite_limit = (limit & 0x7f800000u) != 0x7f800000u;
    if (finite_limit && scalar_divide_float_fast(gradient ^ kCoxSign, denominator, weight)) {
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
