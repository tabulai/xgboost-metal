#include "survival_constants.metal.h"

// Adjacent float32 bounds can require subtracting two CDF values only a few
// double ulps apart. Use a third float for these intervals, rounding observable
// distribution operations to 53 bits before subtraction. Ordinary intervals
// retain the less expensive extended two-float path.
inline float2 survival_two_sum(float a, float b) {
  float sum = a + b, back = sum - a;
  return float2(sum, (a - (sum - back)) + (b - back));
}
inline float3 survival_wide_normalize(float3 value) {
  for (uint pass = 0; pass < 2; ++pass) {
    auto low = survival_two_sum(value.y, value.z);
    auto high = survival_two_sum(value.x, low.x);
    auto middle = survival_two_sum(high.y, low.y);
    value = float3(high.x, middle.x, middle.y);
  }
  return value;
}
inline float3 survival_wide_add(float3 a, float3 b) {
  auto high = survival_two_sum(a.x, b.x);
  auto middle = survival_two_sum(a.y, b.y);
  auto carry = survival_two_sum(high.y, middle.x);
  auto result = survival_two_sum(high.x, carry.x);
  float low = ((a.z + b.z) + middle.y) + carry.y;
  return survival_wide_normalize(float3(result.x, result.y, low));
}
inline float3 survival_wide_multiply(float3 a, float3 b) {
  float high = a.x * b.x, error = fma(a.x, b.x, -high);
  float first = a.x * b.y, second = a.y * b.x;
  float low = fma(a.x, b.y, -first) + fma(a.y, b.x, -second) + a.x * b.z + a.z * b.x + a.y * b.y;
  auto result = survival_wide_add(float3(high, error, 0), float3(first, 0, 0));
  result = survival_wide_add(result, float3(second, 0, 0));
  return survival_wide_add(result, float3(low, 0, 0));
}
inline float3 survival_wide_divide(float3 a, float3 b) {
  float high = a.x / b.x;
  auto remainder = survival_wide_add(a, -survival_wide_multiply(b, float3(high, 0, 0)));
  float middle = remainder.x / b.x;
  remainder = survival_wide_add(remainder, -survival_wide_multiply(b, float3(middle, 0, 0)));
  return survival_wide_normalize(float3(high, middle, remainder.x / b.x));
}
inline float3 survival_wide_round(float3 value) {
  if (!isfinite(value.x) || value.x == 0) return value;
  int exponent = 0;
  float significand = frexp(value.x, exponent);
  if (fabs(significand) == 0.5f && value.x * (value.y != 0 ? value.y : value.z) < 0) --exponent;
  int shift = 53 - exponent;
  float2 scaled = accurate_add(float2(ldexp(value.y, shift), 0), float2(ldexp(value.z, shift), 0));
  int integral = int(floor(scaled.x));
  float2 fraction = accurate_add(scaled, float2(-float(integral), 0));
  int carry = int(floor(fraction.x));
  integral += carry;
  fraction = accurate_add(fraction, float2(-float(carry), 0));
  if (fraction.x < 0 || (fraction.x == 0 && fraction.y < 0)) {
    --integral;
    fraction = accurate_add(fraction, float2(1, 0));
  }
  bool above = fraction.x > 0.5f || (fraction.x == 0.5f && fraction.y > 0);
  bool tie = fraction.x == 0.5f && fraction.y == 0;
  if (above || (tie && (integral & 1))) ++integral;
  float middle = float(integral), low = float(integral - int(middle));
  return survival_wide_normalize(float3(value.x, ldexp(middle, -shift), ldexp(low, -shift)));
}
inline float3 survival_double_add(float3 a, float3 b) { return survival_wide_round(survival_wide_add(a, b)); }
inline float3 survival_double_multiply(float3 a, float3 b) { return survival_wide_round(survival_wide_multiply(a, b)); }
inline float3 survival_double_divide(float3 a, float3 b) { return survival_wide_round(survival_wide_divide(a, b)); }
inline float3 survival_wide_exp(float3 value) {
  int exponent = int(rint(value.x * 0x1.715476p0f));
  auto residual = survival_wide_add(value, -survival_wide_multiply(kSurvivalWideLn2, float3(float(exponent), 0, 0)));
  auto polynomial = kSurvivalWideExp[16];
  for (int i = 15; i >= 0; --i)
    polynomial = survival_wide_add(kSurvivalWideExp[i], survival_wide_multiply(polynomial, residual));
  return survival_wide_round(ldexp(polynomial, int3(exponent)));
}
inline float3 survival_wide_log(float value) {
  auto normalized = extended_float(value);
  auto t = survival_wide_divide(float3(normalized.significand.x - 1, 0, 0),
                                survival_wide_add(float3(normalized.significand.x, 0, 0), float3(1, 0, 0)));
  auto square = survival_wide_multiply(t, t), polynomial = kSurvivalWideLog[19];
  for (int i = 18; i >= 0; --i)
    polynomial = survival_wide_add(kSurvivalWideLog[i], survival_wide_multiply(polynomial, square));
  auto logarithm = survival_wide_multiply(float3(2, 0, 0), survival_wide_multiply(t, polynomial));
  logarithm = survival_wide_add(logarithm, survival_wide_multiply(kSurvivalWideLn2, float3(float(normalized.exponent), 0, 0)));
  return survival_wide_round(logarithm);
}
struct SurvivalWideDistribution { float3 pdf, cdf, first; };
inline SurvivalWideDistribution survival_wide_distribution(float3 z, uint kind) {
  auto one = float3(1, 0, 0);
  if (kind == 0) {
    auto square = survival_double_multiply(z, z);
    auto pdf = survival_double_multiply(survival_wide_exp(-0.5f * square), kSurvivalWideInvSqrt2Pi);
    auto a = z.x < 0 ? -z : z;
    float3 tail;
    if (a.x < 3) {
      auto term = a, sum = a, zz = survival_wide_multiply(a, a);
      for (uint k = 1; k <= 48; ++k) {
        term = survival_wide_divide(survival_wide_multiply(term, zz), float3(float(2 * k + 1), 0, 0));
        sum = survival_wide_add(sum, term);
      }
      // Use an unrounded density for erf, whose evaluation is independent of PDF.
      auto density = survival_wide_multiply(survival_wide_exp(-0.5f * square), kSurvivalWideInvSqrt2Pi);
      tail = survival_wide_add(float3(0.5f, 0, 0), -survival_wide_multiply(density, sum));
    } else {
      auto fraction = float3(0);
      for (uint k = 96; k > 0; --k)
        fraction = survival_wide_divide(float3(float(k), 0, 0), survival_wide_add(a, fraction));
      auto density = survival_wide_multiply(survival_wide_exp(-0.5f * square), kSurvivalWideInvSqrt2Pi);
      tail = survival_wide_divide(density, survival_wide_add(a, fraction));
    }
    auto erf_value = survival_wide_round(survival_wide_add(one, -2.0f * tail));
    if (z.x < 0) erf_value = -erf_value;
    return SurvivalWideDistribution{pdf, 0.5f * survival_double_add(one, erf_value), survival_double_multiply(-z, pdf)};
  }
  auto w = survival_wide_exp(z);
  if (kind == 1) {
    auto denominator = survival_double_add(one, w);
    auto pdf = survival_double_divide(w, survival_double_multiply(denominator, denominator));
    auto cdf = survival_double_divide(w, denominator);
    auto first = survival_double_divide(survival_double_multiply(pdf, survival_double_add(one, -w)), denominator);
    return SurvivalWideDistribution{pdf, cdf, first};
  }
  auto negative = survival_wide_exp(-w);
  auto pdf = survival_double_multiply(w, negative);
  return SurvivalWideDistribution{pdf, survival_double_add(one, -negative),
                                   survival_double_multiply(survival_double_add(one, -w), pdf)};
}
inline SurvivalReal survival_from_wide(float3 value) {
  return sr(accurate_add(float2(value.x, value.y), float2(value.z, 0)));
}
inline bool survival_censored_difference(float lower, float upper, float prediction,
                                      constant SurvivalParams& p, SurvivalReal upper_z, thread SurvivalReal& probability,
                                      thread SurvivalReal& pdf_difference, thread SurvivalReal& first_difference) {
  if (p.scale == 0 || !isfinite(upper)) return false;
  bool left_tail = lower <= 0 && p.distribution == 2;
  bool narrow = lower > 0 && upper > lower && as_type<uint>(upper) - as_type<uint>(lower) <= 1024;
  if (!left_tail && !narrow) return false;
  if (left_tail) {
    float estimate = extended_result(upper_z);
    if (!isfinite(estimate) || estimate > -7 || estimate < -41) return false;
  }
  auto zu = survival_double_divide(survival_double_add(survival_wide_log(upper), float3(-prediction, 0, 0)), float3(p.scale, 0, 0));
  if (left_tail) {
    // Extreme-value CDF is 1-exp(-exp(z)). In the left tail the reference's
    // rounded double exponential controls both the CDF and a cancelling
    // Hessian numerator. Two-float exp near one loses those rounding bits.
    if (!isfinite(zu.x) || zu.x >= -8 || zu.x < -40) return false;
    auto hi = survival_wide_distribution(zu, p.distribution);
    probability = survival_from_wide(hi.cdf);
    pdf_difference = survival_from_wide(hi.pdf);
    first_difference = survival_from_wide(hi.first);
    return true;
  }
  auto zl = survival_double_divide(survival_double_add(survival_wide_log(lower), float3(-prediction, 0, 0)), float3(p.scale, 0, 0));
  float limit = p.distribution == 0 ? 8 : 32;
  // Outside these ranges the CDF is saturated, or its small value already has
  // a full relative significand in the ordinary extended representation.
  if (!isfinite(zl.x) || !isfinite(zu.x) || fabs(zl.x) > limit || fabs(zu.x) > limit ||
      (p.distribution == 2 && max(zl.x, zu.x) > 4)) return false;
  auto lo = survival_wide_distribution(zl, p.distribution), hi = survival_wide_distribution(zu, p.distribution);
  probability = survival_from_wide(survival_double_add(hi.cdf, -lo.cdf));
  pdf_difference = survival_from_wide(survival_double_add(hi.pdf, -lo.pdf));
  first_difference = survival_from_wide(survival_double_add(hi.first, -lo.first));
  return true;
}
