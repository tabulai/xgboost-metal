// Binary64 operations used by Cox's ordered reductions. IEEE significands are
// represented with integer bits, including guard/sticky bits for round-to-nearest.
// A 128-bit pair retains the exact product through fused multiply-add. All
// operations execute in Metal; no host arithmetic is used for training rows.
constant ulong kCoxSign = 0x8000000000000000ul;
constant ulong kCoxFraction = 0x000ffffffffffffful;
inline ulong cox_jam(ulong value, uint shift) {
  if (!shift) return value;
  return shift < 64 ? (value >> shift) | ulong((value << (64 - shift)) != 0) : ulong(value != 0);
}
inline ulong cox_add_bits(ulong a, ulong b) {
  if ((a & ~kCoxSign) < (b & ~kCoxSign)) { auto temporary = a; a = b; b = temporary; }
  uint ae = uint((a >> 52) & 2047), be = uint((b >> 52) & 2047);
  ulong sign = a & kCoxSign;
  if (ae == 2047) {
    if ((a & kCoxFraction) || (be == 2047 && ((b & kCoxFraction) || ((a ^ b) & kCoxSign))))
      return 0x7ff8000000000000ul;
    return a;
  }
  ulong av = (a & kCoxFraction) | (ae ? 1ul << 52 : 0);
  ulong bv = (b & kCoxFraction) | (be ? 1ul << 52 : 0);
  ae = max(ae, 1u); be = max(be, 1u);
  av <<= 3; bv = cox_jam(bv << 3, ae - be);
  ulong value = (a ^ b) & kCoxSign ? av - bv : av + bv;
  if (!value) return (a & b) & kCoxSign;
  if (value & (1ul << 56)) { value = cox_jam(value, 1); ++ae; }
  else {
    uint shift = min(uint(clz(value)) - 8, ae - 1);
    value <<= shift; ae -= shift;
  }
  uint tail = uint(value & 7);
  value >>= 3;
  if (tail > 4 || (tail == 4 && (value & 1))) ++value;
  if (value & (1ul << 53)) { value >>= 1; ++ae; }
  if (ae >= 2047) return sign | 0x7ff0000000000000ul;
  if (ae == 1 && value < (1ul << 52)) ae = 0;
  return sign | (ulong(ae) << 52) | (value & kCoxFraction);
}
inline ulong cox_float_bits(float value) {
  uint bits = as_type<uint>(value), power = (bits >> 23) & 255, mantissa = bits & 0x7fffffu;
  ulong sign = ulong(bits & 0x80000000u) << 32;
  if (power == 255) return sign | 0x7ff0000000000000ul | (ulong(mantissa) << 29);
  if (!power) {
    if (!mantissa) return sign;
    uint shift = clz(mantissa) - 8;
    mantissa <<= shift;
    return sign | (ulong(897 - shift) << 52) | (ulong(mantissa & 0x7fffffu) << 29);
  }
  return sign | (ulong(power + 896) << 52) | (ulong(mantissa) << 29);
}
inline ulong cox_scale_bits(ulong value, int shift) {
  int power = int((value >> 52) & 2047);
  ulong sign = value & kCoxSign, mantissa = value & kCoxFraction;
  if (power == 2047 || !(value & ~kCoxSign)) return value;
  if (!power) {
    uint amount = uint(clz(mantissa)) - 11;
    mantissa <<= amount;
    power = 1 - int(amount);
  } else mantissa |= 1ul << 52;
  power += shift;
  if (power >= 2047) return sign | 0x7ff0000000000000ul;
  if (power > 0) return sign | (ulong(power) << 52) | (mantissa & kCoxFraction);
  auto small = cox_jam(mantissa << 3, uint(1 - power));
  uint tail = uint(small & 7); small >>= 3;
  if (tail > 4 || (tail == 4 && (small & 1))) ++small;
  return sign | small;
}
// Wide words are low first. Keep the exact 106-bit product through FMA.
inline ulong2 cox_wide_product(ulong a, ulong b) {
  ulong low = ulong(uint(a)) * uint(b);
  ulong middle = (a >> 32) * uint(b) + (low >> 32);
  ulong next = ulong(uint(a)) * (b >> 32) + uint(middle);
  return ulong2((next << 32) | uint(low), (a >> 32) * (b >> 32) + (middle >> 32) + (next >> 32));
}
inline ulong2 cox_wide_left(ulong2 value, uint shift) {
  if (!shift) return value;
  if (shift < 64) return ulong2(value.x << shift, (value.y << shift) | (value.x >> (64 - shift)));
  return ulong2(0, value.x << (shift - 64));
}
inline ulong2 cox_wide_jam(ulong2 value, uint shift) {
  if (!shift) return value;
  if (shift < 64) return ulong2((value.x >> shift) | (value.y << (64 - shift)) | ulong((value.x << (64 - shift)) != 0), value.y >> shift);
  if (shift == 64) return ulong2(value.y | ulong(value.x != 0), 0);
  if (shift < 128) return ulong2((value.y >> (shift - 64)) | ulong(value.x != 0 || (value.y << (128 - shift)) != 0), 0);
  return ulong2(ulong(value.x != 0 || value.y != 0), 0);
}
inline bool cox_wide_less(ulong2 a, ulong2 b) { return a.y < b.y || (a.y == b.y && a.x < b.x); }
inline ulong2 cox_wide_add(ulong2 a, ulong2 b) { ulong low = a.x + b.x; return ulong2(low, a.y + b.y + ulong(low < a.x)); }
inline ulong2 cox_wide_sub(ulong2 a, ulong2 b) { return ulong2(a.x - b.x, a.y - b.y - ulong(a.x < b.x)); }
inline ulong cox_round_wide(ulong2 value, int exponent, ulong sign) {
  if (!(value.x | value.y)) return sign;
  if (value.y & (1ul << 63)) { value = cox_wide_jam(value, 1); ++exponent; }
  else {
    uint shift = value.y ? uint(clz(value.y)) - 1 : 63 + uint(clz(value.x));
    value = cox_wide_left(value, shift); exponent -= int(shift);
  }
  uint shift = 71 + uint(max(-1022 - exponent, 0));
  ulong extended = cox_wide_jam(value, shift).x;
  uint tail = uint(extended & 7);
  ulong mantissa = extended >> 3;
  if (tail > 4 || (tail == 4 && (mantissa & 1))) ++mantissa;
  int power = max(0, exponent + 1023);
  if (mantissa & (1ul << 53)) { mantissa >>= 1; ++power; }
  if (power >= 2047) return sign | 0x7ff0000000000000ul;
  if (!power && mantissa >= (1ul << 52)) power = 1;
  return sign | (ulong(power) << 52) | (mantissa & kCoxFraction);
}
inline ulong cox_finite_significand(ulong value, thread int& exponent) {
  uint power = uint((value >> 52) & 2047);
  ulong mantissa = value & kCoxFraction;
  if (!power) {
    uint shift = mantissa ? uint(clz(mantissa)) - 11 : 0;
    exponent = -1022 - int(shift);
    return mantissa << shift;
  }
  exponent = int(power) - 1023;
  return mantissa | (1ul << 52);
}
inline ulong cox_fma_bits(ulong a, ulong b, ulong c) {
  uint ae = uint((a >> 52) & 2047), be = uint((b >> 52) & 2047), ce = uint((c >> 52) & 2047);
  ulong sign = (a ^ b) & kCoxSign;
  if (ae == 2047 || be == 2047 || ce == 2047) {
    if ((ae == 2047 && (a & kCoxFraction)) || (be == 2047 && (b & kCoxFraction)) ||
        (ce == 2047 && (c & kCoxFraction))) return 0x7ff8000000000000ul;
    if (ae == 2047 || be == 2047) {
      if (!(a & ~kCoxSign) || !(b & ~kCoxSign) || (ce == 2047 && ((c ^ sign) & kCoxSign))) return 0x7ff8000000000000ul;
      return sign | 0x7ff0000000000000ul;
    }
    return c;
  }
  if (!(a & ~kCoxSign) || !(b & ~kCoxSign)) return c & ~kCoxSign ? c : sign & c;
  int ax, bx, cx;
  auto av = cox_finite_significand(a, ax), bv = cox_finite_significand(b, bx);
  int exponent = ax + bx;
  auto product = cox_wide_left(cox_wide_product(av, bv), 22);
  if (product.y & (1ul << 63)) { product = cox_wide_jam(product, 1); ++exponent; }
  if (!(c & ~kCoxSign)) return cox_round_wide(product, exponent, sign);
  auto cv = cox_finite_significand(c, cx);
  auto addend = cox_wide_left(ulong2(cv, 0), 74);
  int common = max(exponent, cx);
  product = cox_wide_jam(product, uint(common - exponent));
  addend = cox_wide_jam(addend, uint(common - cx));
  ulong2 result;
  if (!((sign ^ c) & kCoxSign)) result = cox_wide_add(product, addend);
  else if (cox_wide_less(product, addend)) { result = cox_wide_sub(addend, product); sign = c & kCoxSign; }
  else result = cox_wide_sub(product, addend);
  if (!(result.x | result.y)) return 0;
  return cox_round_wide(result, common, sign);
}
inline ulong cox_multiply_bits(ulong a, ulong b) { return cox_fma_bits(a, b, (a ^ b) & kCoxSign); }
inline ulong cox_divide_bits(ulong a, ulong b) {
  uint ae = uint((a >> 52) & 2047), be = uint((b >> 52) & 2047);
  ulong sign = (a ^ b) & kCoxSign;
  if (ae == 2047 || be == 2047) {
    if ((ae == 2047 && (a & kCoxFraction)) || (be == 2047 && (b & kCoxFraction)) || (ae == be)) return 0x7ff8000000000000ul;
    return ae == 2047 ? sign | 0x7ff0000000000000ul : sign;
  }
  if (!(b & ~kCoxSign)) return a & ~kCoxSign ? sign | 0x7ff0000000000000ul : 0x7ff8000000000000ul;
  if (!(a & ~kCoxSign)) return sign;
  int ax, bx;
  ulong numerator = cox_finite_significand(a, ax), denominator = cox_finite_significand(b, bx);
  int exponent = ax - bx;
  if (numerator < denominator) { numerator <<= 1; --exponent; }
  ulong remainder = numerator - denominator, quotient = 1;
  #pragma clang loop unroll_count(5)
  for (uint i = 0; i < 55; ++i) {
    remainder <<= 1;
    bool bit = remainder >= denominator;
    if (bit) remainder -= denominator;
    quotient = (quotient << 1) | ulong(bit);
  }
  quotient |= ulong(remainder != 0);
  return cox_round_wide(cox_wide_left(ulong2(quotient, 0), 71), exponent, sign);
}
