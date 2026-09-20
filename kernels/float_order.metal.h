// Metal arithmetic flushes float32 subnormals. Preserve their ordering when
// comparing user data, histogram cuts and model thresholds.
inline bool fp32_subnormal(float value) {
  uint magnitude = as_type<uint>(value) & 0x7fffffffu;
  return magnitude - 1u < 0x7fffffu;
}
inline uint fp32_order_key(float value) {
  uint bits = as_type<uint>(value);
  if (!(bits & 0x7fffffffu)) bits = 0;
  return bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}
inline bool fp32_equal(float a, float b) {
  if (fp32_subnormal(a) || fp32_subnormal(b)) return as_type<uint>(a) == as_type<uint>(b);
  return a == b;
}
inline bool fp32_less(float a, float b) {
  if (fp32_subnormal(a) || fp32_subnormal(b))
    return !isnan(a) && !isnan(b) && fp32_order_key(a) < fp32_order_key(b);
  return a < b;
}
inline bool fp32_less_equal(float a, float b) {
  if (fp32_subnormal(a) || fp32_subnormal(b))
    return !isnan(a) && !isnan(b) && fp32_order_key(a) <= fp32_order_key(b);
  return a <= b;
}
inline bool fp32_greater_equal(float a, float b) { return fp32_less_equal(b, a); }

inline float fp32_next_down(float value) {
  uint bits = as_type<uint>(value), magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u || bits == 0xff800000u) return value;
  if (!magnitude) return as_type<float>(0x80000001u);
  return as_type<float>(bits + ((bits & 0x80000000u) ? 1u : 0xffffffffu));
}
