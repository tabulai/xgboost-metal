// Preserve the reference's mixed float/double exponential and rounding order.
// All arithmetic and ordered scans execute on the GPU.
inline ulong cox_shuffle_bits(ulong value, ushort lane) { return as_type<ulong>(simd_shuffle(as_type<uint2>(value), lane)); }
#include "cox_scan.metal.h"
inline float cox_float_result(ulong value) {
  uint power = uint((value >> 52) & 2047);
  uint sign = uint(value >> 32) & 0x80000000u;
  if (power == 2047) return as_type<float>(sign | 0x7f800000u | ((value & kCoxFraction) ? 0x400000u : 0u));
  if (!(value & ~kCoxSign)) return as_type<float>(sign);
  int exponent;
  ulong mantissa = cox_finite_significand(value, exponent);
  if (exponent > 127) return as_type<float>(sign | 0x7f800000u);
  uint shift = 29 + uint(max(-126 - exponent, 0));
  auto scaled = cox_jam(mantissa << 3, shift);
  uint tail = uint(scaled & 7), rounded = uint(scaled >> 3);
  if (tail > 4 || (tail == 4 && (rounded & 1))) ++rounded;
  int result_power = max(0, exponent + 127);
  if (rounded & 0x1000000u) { rounded >>= 1; ++result_power; }
  if (result_power >= 255) return as_type<float>(sign | 0x7f800000u);
  if (!result_power && rounded >= 0x800000u) result_power = 1;
  return as_type<float>(sign | (uint(result_power) << 23) | (rounded & 0x7fffffu));
}
kernel void cox_exact_inputs(device const float* prediction [[buffer(0)]],
                               device const uint* order [[buffer(1)]],
                               device ulong2* output [[buffer(2)]],
                               constant uint& rows [[buffer(3)]], uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  float value = prediction[order[row]];
  output[row] = ulong2(cox_double_exp(value), cox_float_bits(cox_platform_expf(value)));
}
kernel void cox_risk_increments(device const ulong2* prefix [[buffer(0)]],
                                 device const float* labels [[buffer(1)]],
                                 device const uint* order [[buffer(2)]],
                                 device ulong2* output [[buffer(3)]],
                                 device ulong2* seed [[buffer(4)]],
                                 constant uint& rows [[buffer(5)]], uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  if (!row) seed[0] = ulong2(prefix[rows - 1].y, 0);
  output[row] = ulong2(row && cox_segment_start(labels, order, row) ? prefix[row - 1].x ^ kCoxSign : 0, 0);
}
kernel void cox_exact_terms(device const ulong2* risks [[buffer(0)]],
                              device const float* labels [[buffer(1)]],
                              device const uint* order [[buffer(2)]],
                              device ulong2* output [[buffer(3)]],
                              constant uint& rows [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  if (labels[order[row]] > 0) {
    auto risk = risks[row].x, one = cox_float_bits(1);
    output[row] = ulong2(cox_divide_bits(one, risk), cox_divide_bits(one, cox_multiply_bits(risk, risk)));
  } else output[row] = 0;
}
kernel void cox_exact_gradients(device const ulong2* inputs [[buffer(0)]],
                                  device const ulong2* cumulative [[buffer(1)]],
                                  device const float* labels [[buffer(2)]],
                                  device const float* weights [[buffer(3)]],
                                  device const uint* order [[buffer(4)]],
                                  device Pair* output [[buffer(5)]],
                                  constant uint& rows [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  uint index = order[row];
  auto e = inputs[row].x, r = cumulative[row].x, s = cumulative[row].y;
  auto g = cox_fma_bits(e, r, cox_float_bits(float(labels[index] > 0)) ^ kCoxSign);
  auto correction = cox_multiply_bits(cox_multiply_bits(e, e), s);
  auto h = cox_fma_bits(e, r, correction ^ kCoxSign);
  auto weight = cox_float_bits(weights[index]);
  output[index] = Pair{cox_float_result(cox_multiply_bits(g, weight)), cox_float_result(cox_multiply_bits(h, weight))};
}
