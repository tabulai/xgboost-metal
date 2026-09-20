// Keep the reference's float32 rounding, including gradual underflow. The
// ordinary path uses hardware arithmetic; only subnormal inputs/results need
// the verified integer binary64 operations followed by float32 rounding.
inline float sketch_add(float a, float b) {
  float result = a + b;
  uint aa = as_type<uint>(a), bb = as_type<uint>(b);
  bool cancellation = (as_type<uint>(result) & 0x7fffffffu) < 0x800000u &&
                      ((aa | bb) & 0x7fffffffu) && (aa ^ bb) != 0x80000000u;
  if (fp32_subnormal(a) || fp32_subnormal(b) || cancellation)
    return cox_float_result(cox_add_bits(cox_float_bits(a), cox_float_bits(b)));
  return result;
}
inline float sketch_subtract(float a, float b) { return sketch_add(a, as_type<float>(as_type<uint>(b) ^ 0x80000000u)); }
inline float sketch_multiply(float a, float b) {
  float result = a * b;
  bool underflow = (as_type<uint>(result) & 0x7fffffffu) < 0x800000u &&
                   (as_type<uint>(a) & 0x7fffffffu) && (as_type<uint>(b) & 0x7fffffffu);
  if (fp32_subnormal(a) || fp32_subnormal(b) || underflow)
    return cox_float_result(cox_multiply_bits(cox_float_bits(a), cox_float_bits(b)));
  return result;
}
inline float sketch_divide(float a, float b) {
  float result = a / b;
  bool underflow = (as_type<uint>(result) & 0x7fffffffu) < 0x800000u && (as_type<uint>(a) & 0x7fffffffu);
  if (fp32_subnormal(a) || fp32_subnormal(b) || underflow)
    return cox_float_result(cox_divide_bits(cox_float_bits(a), cox_float_bits(b)));
  return result;
}
inline float sketch_max(float a, float b) { return fp32_less(a, b) ? b : a; }
