// Scalar ApplyTreeSplit recomputes its stored parent weight from the two
// selected child statistics. Convert each child to binary64 before adding.
// The original parent weight still governs unsplit leaves and split scoring.
inline ulong split_stat_integer_bits(long value) {
  if (!value) return 0;
  ulong bits = as_type<ulong>(value);
  ulong sign = value < 0 ? kCoxSign : 0;
  ulong magnitude = sign ? 0ul - bits : bits;
  return cox_round_wide(cox_wide_left(ulong2(magnitude, 0), 63), 63, sign);
}
inline ulong split_stat_value_bits(long value, float scale) {
  if (!value) return 0;
  uint bits = as_type<uint>(scale), power = (bits >> 23) & 255;
  int exponent = power ? int(power) - 127 : 31 - int(clz(bits & 0x7fffffu)) - 149;
  return cox_scale_bits(split_stat_integer_bits(value), -exponent);
}
inline ulong2 split_parent_sums(long2 left, long2 right, float2 scale) {
  return ulong2(cox_add_bits(split_stat_value_bits(left.x, scale.x), split_stat_value_bits(right.x, scale.x)),
                cox_add_bits(split_stat_value_bits(left.y, scale.y), split_stat_value_bits(right.y, scale.y)));
}
inline float split_parent_weight(ulong2 sums, float reg_alpha, float reg_lambda,
                                  float max_delta_step, float2 bounds) {
  ulong result = 0;
  if (!(sums.y & kCoxSign) && (sums.y & ~kCoxSign)) {
    ulong alpha = cox_float_bits(reg_alpha) & ~kCoxSign;
    ulong gradient = sums.x;
    if ((gradient & ~kCoxSign) <= alpha) gradient = 0;
    else gradient = cox_add_bits(gradient, gradient & kCoxSign ? alpha : alpha ^ kCoxSign);
    ulong denominator = cox_add_bits(sums.y, cox_float_bits(reg_lambda));
    result = cox_divide_bits(gradient ^ kCoxSign, denominator);
    ulong limit = cox_float_bits(max_delta_step) & ~kCoxSign;
    if (limit && (result & ~kCoxSign) > limit) result = (result & kCoxSign) | limit;
  }
  float weight = cox_float_result(result);
  if (fp32_less(weight, bounds.x)) weight = bounds.x;
  if (fp32_less(bounds.y, weight)) weight = bounds.y;
  return weight;
}
