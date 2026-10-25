// Specialize the existing integer -> binary64 conversion without changing its
// rounding point. A nonzero int64 magnitude has exponent e in [0, 63], hence
// its binary64 result is always finite and normal. At most 11 low bits need
// rounding; wide normalization, sticky shifts and underflow handling are not
// needed here. Keep scaling separate below: RN64(integer) is observable before
// cox_scale_bits, including if scaling introduces a second rounding.
inline ulong split_stat_integer_bits_fast(long value) {
  if (!value) return 0;
  ulong bits = as_type<ulong>(value);
  ulong sign = value < 0 ? kCoxSign : 0;
  // Unsigned subtraction also represents abs(INT64_MIN) without signed UB.
  ulong magnitude = sign ? 0ul - bits : bits;
  uint exponent = 63 - uint(clz(magnitude));
  ulong mantissa;
  if (exponent <= 52) {
    mantissa = magnitude << (52 - exponent);
  } else {
    uint shift = exponent - 52;  // 1..11, so every shift below is defined.
    mantissa = magnitude >> shift;
    ulong remainder = magnitude & ((1ul << shift) - 1);
    ulong halfway = 1ul << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (mantissa & 1))) ++mantissa;
    if (mantissa & (1ul << 53)) { mantissa >>= 1; ++exponent; }
  }
  return sign | (ulong(exponent + 1023) << 52) | (mantissa & kCoxFraction);
}

inline ulong split_stat_value_bits_fast(long value, float scale) {
  if (!value) return 0;
  // This is deliberately the original scale decoding and scaling operation.
  uint bits = as_type<uint>(scale), power = (bits >> 23) & 255;
  int exponent = power ? int(power) - 127 : 31 - int(clz(bits & 0x7fffffu)) - 149;
  return cox_scale_bits(split_stat_integer_bits_fast(value), -exponent);
}

inline ulong2 scalar_histogram_sum_fast(long2 value, float2 scale) {
  return ulong2(split_stat_value_bits_fast(value.x, scale.x),
                split_stat_value_bits_fast(value.y, scale.y));
}

// Candidate order, validation, weight/gain arithmetic and writes are identical
// to the existing consider_split paths. Only the four int64 conversions use
// the specialized normal binary64 construction above.
template <bool FastWeight, bool FastGain>
inline void consider_split_stats_fast(thread Split& best, long2 left_int, long2 right_int,
                                     float2 scale, float threshold, uint default_left,
                                     uint feature, int monotone, float2 bounds,
                                     float parent_gain, constant SplitParams& p) {
  auto left = scalar_histogram_sum_fast(left_int, scale);
  auto right = scalar_histogram_sum_fast(right_int, scale);
  if (!scalar_valid_hessian(left.y, p.min_child_weight) ||
      !scalar_valid_hessian(right.y, p.min_child_weight)) return;
  float lw = FastWeight ? scalar_weight_fast(left, p, bounds) : scalar_weight(left, p, bounds);
  float rw = FastWeight ? scalar_weight_fast(right, p, bounds) : scalar_weight(right, p, bounds);
  if ((monotone > 0 && !fp32_less_equal(lw, rw)) ||
      (monotone < 0 && !fp32_greater_equal(lw, rw))) return;
  float left_gain = FastGain ? scalar_gain_fast(left, lw, p) : scalar_gain(left, lw, p);
  float right_gain = FastGain ? scalar_gain_fast(right, rw, p) : scalar_gain(right, rw, p);
  float gain = scalar_float_add(scalar_float_add(left_gain, right_gain), -parent_gain);
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
