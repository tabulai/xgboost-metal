// Weight and gain use the same rounded binary64 H + lambda. Compute that
// intermediate once, retaining every original division, square, FMA and
// binary32 rounding point. The original standalone helpers remain controls.
inline float2 scalar_weight_gain_fused(ulong2 sums, constant SplitParams& p, float2 bounds) {
  bool positive_hessian = !(sums.y & kCoxSign) && (sums.y & ~kCoxSign);
  ulong curvature = positive_hessian ? cox_add_bits(sums.y, cox_float_bits(p.reg_lambda)) : 0;
  float weight = 0;
  if (positive_hessian) {
    ulong gradient = sums.x;
    if (as_type<uint>(p.reg_alpha) == 0 && ((gradient >> 52) & 2047) != 2047) {
      // With +0 alpha, finite nonzero G + copysign(0,-G) is exactly G.
      // The original threshold maps either signed zero to positive zero.
      if (!(gradient & ~kCoxSign)) gradient = 0;
    } else {
      ulong alpha = cox_float_bits(p.reg_alpha) & ~kCoxSign;
      if ((gradient & ~kCoxSign) <= alpha) gradient = 0;
      else gradient = cox_add_bits(gradient, gradient & kCoxSign ? alpha : alpha ^ kCoxSign);
    }
    uint limit = as_type<uint>(p.max_delta_step) & 0x7fffffffu;
    bool finite_limit = (limit & 0x7f800000u) != 0x7f800000u;
    if (finite_limit && scalar_divide_float(gradient ^ kCoxSign, curvature, weight)) {
      uint bits = as_type<uint>(weight);
      if (limit && (bits & 0x7fffffffu) > limit)
        weight = as_type<float>((bits & 0x80000000u) | limit);
    } else {
      ulong result = cox_divide_bits(gradient ^ kCoxSign, curvature);
      ulong bound = cox_float_bits(p.max_delta_step) & ~kCoxSign;
      if (bound && (result & ~kCoxSign) > bound) result = (result & kCoxSign) | bound;
      weight = cox_float_result(result);
    }
  }
  if (fp32_less(weight, bounds.x)) weight = bounds.x;
  if (fp32_less(bounds.y, weight)) weight = bounds.y;
  if (!positive_hessian) return float2(weight, 0);
  auto w = cox_float_bits(weight);
  auto square = scalar_square_fast_bits(weight, w);
  auto quadratic = cox_multiply_bits(curvature, square);
  auto sum = cox_fma_bits(cox_add_bits(sums.x, sums.x), w, quadratic);
  if (as_type<uint>(p.reg_alpha) == 0 &&
      (as_type<uint>(weight) & 0x7f800000u) != 0x7f800000u &&
      ((sum >> 52) & 2047) != 2047) {
    // For finite weight and sum, fma(+0, abs(weight), sum) preserves every
    // nonzero sum but changes -0 to +0. Keep that observable zero sign before
    // gain negation. -0 alpha and all nonfinite cases retain the old FMA.
    if (!(sum & ~kCoxSign)) sum = 0;
  } else {
    auto alpha = cox_float_bits(p.reg_alpha);
    sum = cox_fma_bits(cox_add_bits(alpha, alpha), w & ~kCoxSign, sum);
  }
  return float2(weight, cox_float_result(sum ^ kCoxSign));
}

inline void consider_split_score_fused(thread Split& best, long2 left_int, long2 right_int,
                                       float2 scale, float threshold, uint default_left,
                                       uint feature, int monotone, float2 bounds,
                                       float parent_gain, constant SplitParams& p) {
  auto left = scalar_histogram_sum_fast(left_int, scale);
  auto right = scalar_histogram_sum_fast(right_int, scale);
  if (!scalar_valid_hessian(left.y, p.min_child_weight) ||
      !scalar_valid_hessian(right.y, p.min_child_weight)) return;
  float2 left_score = scalar_weight_gain_fused(left, p, bounds);
  float2 right_score = scalar_weight_gain_fused(right, p, bounds);
  float lw = left_score.x, rw = right_score.x;
  if ((monotone > 0 && !fp32_less_equal(lw, rw)) ||
      (monotone < 0 && !fp32_greater_equal(lw, rw))) return;
  float gain = scalar_float_add(scalar_float_add(left_score.y, right_score.y), -parent_gain);
  if (gain > best.gain) {
    best.gain = gain; best.threshold = threshold; best.feature = feature;
    best.default_left = default_left;
    best.left_h = cox_float_result(left.y);
    best.right_h = cox_float_result(right.y);
    best.left_weight = lw; best.right_weight = rw;
    best.left_sum = IntPair{left_int.x, left_int.y};
    best.right_sum = IntPair{right_int.x, right_int.y};
  }
}
