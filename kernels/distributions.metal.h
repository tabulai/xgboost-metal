// Two-float significands with an explicit exponent keep the normal objective's
// double intermediates within range on Metal. Arithmetic remains on the GPU.
struct Extended { float2 significand; int exponent; };

inline float2 accurate_divide(float2 a, float2 b) {
  float high = a.x / b.x;
  float2 remainder = accurate_add(a, -accurate_multiply(b, float2(high, 0)));
  return accurate_add(float2(high, 0), float2((remainder.x + remainder.y) / b.x, 0));
}
inline Extended extended_normalize(float2 value, int exponent = 0) {
  if (value.x == 0) return Extended{float2(0), 0};
  if (!isfinite(value.x)) return Extended{value, 0};
  int shift = 0;
  frexp(value.x, shift);
  return Extended{ldexp(value, int2(-shift)), exponent + shift};
}
inline Extended extended_float(float value) {
  uint bits = as_type<uint>(value), magnitude = bits & 0x7fffffffu;
  if (!magnitude) return Extended{float2(0), 0};
  uint exponent = magnitude >> 23, mantissa = magnitude & 0x7fffffu;
  if (exponent == 255) return Extended{float2(value, 0), 0};
  int power = exponent ? int(exponent) - 150 : -149;
  if (exponent) mantissa |= 0x800000u;
  float sign = bits & 0x80000000u ? -1.0f : 1.0f;
  return extended_normalize(float2(sign * float(mantissa), 0), power);
}
inline Extended extended_negate(Extended value) { return Extended{-value.significand, value.exponent}; }
inline Extended extended_add(Extended a, Extended b) {
  if (a.significand.x == 0) return b;
  if (b.significand.x == 0) return a;
  int exponent = max(a.exponent, b.exponent);
  // Values beyond the significand's precision cannot affect the result.
  if (a.exponent < exponent - 96) return b;
  if (b.exponent < exponent - 96) return a;
  auto value = accurate_add(ldexp(a.significand, int2(a.exponent - exponent)),
                            ldexp(b.significand, int2(b.exponent - exponent)));
  return extended_normalize(value, exponent);
}
inline Extended extended_multiply(Extended a, Extended b) {
  return extended_normalize(accurate_multiply(a.significand, b.significand), a.exponent + b.exponent);
}
inline Extended extended_divide(Extended a, Extended b) {
  return extended_normalize(accurate_divide(a.significand, b.significand), a.exponent - b.exponent);
}
inline bool extended_greater(Extended a, Extended b) {
  return extended_add(a, extended_negate(b)).significand.x > 0;
}
inline float extended_result(Extended value) {
  if (!isfinite(value.significand.x) || value.significand.x == 0) return value.significand.x;
  if (value.exponent > -125) return ldexp(value.significand.x + value.significand.y, value.exponent);
  uint sign = value.significand.x < 0 ? 0x80000000u : 0;
  if (value.exponent < -149) return as_type<float>(sign);
  float2 scaled = ldexp(sign ? -value.significand : value.significand, int2(value.exponent + 149));
  float integral = floor(scaled.x);
  float2 fraction = accurate_add(float2(scaled.x - integral, 0), float2(scaled.y, 0));
  if (fraction.x < 0) { integral -= 1; fraction = accurate_add(fraction, float2(1, 0)); }
  uint rounded = uint(integral);
  bool above = fraction.x > 0.5f || (fraction.x == 0.5f && fraction.y > 0);
  bool tie = fraction.x == 0.5f && fraction.y == 0;
  if (above || (tie && (rounded & 1))) ++rounded;
  return as_type<float>(sign | rounded);
}

constant float2 kExtendedLn2 = float2(0x1.62e430p-1f, -0x1.05c610p-29f);
// Rounded high and low parts of 1/k!, k = 0..12. Range reduction gives
// |r| <= log(2)/2, so the omitted Taylor tail is below two-float precision.
constant float2 kExtendedExpCoefficients[13] = {
  float2(1, 0), float2(1, 0), float2(0.5f, 0),
  float2(0x1.555556p-3f, -0x1.555556p-28f),
  float2(0x1.555556p-5f, -0x1.555556p-30f),
  float2(0x1.111112p-7f, -0x1.dddddep-32f),
  float2(0x1.6c16c2p-10f, -0x1.27d27ep-35f),
  float2(0x1.a01a02p-13f, -0x1.7f97fap-39f),
  float2(0x1.a01a02p-16f, -0x1.7f97fap-42f),
  float2(0x1.71de3ap-19f, 0x1.55b1ccp-45f),
  float2(0x1.27e4fcp-22f, -0x1.10ec14p-47f),
  float2(0x1.ae6456p-26f, 0x1.fd5138p-52f),
  float2(0x1.1eed8ep-29f, 0x1.ff1b14p-54f)
};
inline Extended extended_exp(float value) {
  int exponent = int(rint(value * 0x1.715476p0f));
  float2 residual = accurate_add(float2(value, 0), -accurate_multiply(kExtendedLn2, float2(float(exponent), 0)));
  float2 polynomial = kExtendedExpCoefficients[12];
  for (int i = 11; i >= 0; --i)
    polynomial = accurate_add(kExtendedExpCoefficients[i], accurate_multiply(polynomial, residual));
  return extended_normalize(polynomial, exponent);
}
// log(m) = 2 * (t + t^3/3 + ...), t = (m-1)/(m+1), m in [0.5, 1).
constant float2 kExtendedLogCoefficients[15] = {
  float2(1, 0), float2(0x1.555556p-2f, -0x1.555556p-27f),
  float2(0x1.99999ap-3f, -0x1.99999ap-29f),
  float2(0x1.24924ap-3f, -0x1.b6db6ep-28f),
  float2(0x1.c71c72p-4f, -0x1.c71c72p-31f),
  float2(0x1.745d18p-4f, -0x1.745d18p-29f),
  float2(0x1.3b13b2p-4f, -0x1.89d89ep-29f),
  float2(0x1.111112p-4f, -0x1.dddddep-29f),
  float2(0x1.e1e1e2p-5f, -0x1.e1e1e2p-33f),
  float2(0x1.af286cp-5f, -0x1.af286cp-32f),
  float2(0x1.861862p-5f, -0x1.e79e7ap-31f),
  float2(0x1.642c86p-5f, -0x1.bd37a8p-31f),
  float2(0x1.47ae14p-5f, 0x1.eb851ep-31f),
  float2(0x1.2f684cp-5f, -0x1.2f684cp-32f),
  float2(0x1.1a7b96p-5f, 0x1.1a7b96p-33f)
};
inline float extended_log(Extended value) {
  float2 numerator = accurate_add(value.significand, float2(-1, 0));
  float2 denominator = accurate_add(value.significand, float2(1, 0));
  float2 t = accurate_divide(numerator, denominator), t2 = accurate_multiply(t, t);
  float2 polynomial = kExtendedLogCoefficients[14];
  for (int i = 13; i >= 0; --i)
    polynomial = accurate_add(kExtendedLogCoefficients[i], accurate_multiply(polynomial, t2));
  float2 logarithm = accurate_multiply(float2(2, 0), accurate_multiply(t, polynomial));
  logarithm = accurate_add(logarithm, accurate_multiply(kExtendedLn2, float2(float(value.exponent), 0)));
  return logarithm.x + logarithm.y;
}

inline uint float_mantissa(float value, thread int& power) {
  uint bits = as_type<uint>(value), exponent = (bits >> 23) & 255;
  power = exponent ? int(exponent) - 150 : -149;
  return (bits & 0x7fffffu) | (exponent ? 0x800000u : 0);
}
template<uint N, class Storage>
inline void mass_atomic_limb(Storage output, uint index, uint value, uint shift) {
  mass_atomic_word<N>(output, index + shift / 32, value << (shift % 32));
  if (shift % 32) mass_atomic_word<N>(output, index + shift / 32 + 1, value >> (32 - shift % 32));
}
template<class Storage>
inline void mass_atomic_product(Storage output, float a, float b) {
  int ae = 0, be = 0;
  ulong product = ulong(float_mantissa(a, ae)) * float_mantissa(b, be);
  uint shift = uint(ae + be + 298);
  mass_atomic_limb<kProductMassWords>(output, 0, uint(product), shift);
  mass_atomic_limb<kProductMassWords>(output, 1, uint(product >> 32), shift);
}
template<class Storage>
inline void mass_atomic_triple(Storage output, float a, float b, float c, uint extra_shift = 0) {
  int ae = 0, be = 0, ce = 0;
  ulong product = ulong(float_mantissa(a, ae)) * float_mantissa(b, be);
  uint cm = float_mantissa(c, ce), shift = uint(ae + be + ce + 447) + extra_shift;
  ulong low = ulong(uint(product)) * cm;
  ulong high = ulong(uint(product >> 32)) * cm + (low >> 32);
  mass_atomic_limb<kVarianceMassWords>(output, 0, uint(low), shift);
  mass_atomic_limb<kVarianceMassWords>(output, 1, uint(high), shift);
  mass_atomic_limb<kVarianceMassWords>(output, 2, uint(high >> 32), shift);
}
inline ObjectiveMass mass_float(float value) {
  ObjectiveMass result = mass_zero();
  int power = 0;
  uint mantissa = float_mantissa(value, power), shift = uint(power + 149), index = shift / 32;
  result.word[index] = mantissa << (shift % 32);
  if (shift % 32) result.word[index + 1] = mantissa >> (32 - shift % 32);
  return result;
}

kernel void distribution_mean_partial(device const float* labels [[buffer(0)]],
                                        device const float* weights [[buffer(1)]],
                                        device uint* partial [[buffer(2)]],
                                        constant GradientParams& p [[buffer(3)]],
                                        uint block [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint words = 2 * kProductMassWords + kObjectiveMassWords;
  threadgroup atomic_uint mass[words];
  if (lane < words) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  for (ulong row = begin + lane; row < end; row += 256) {
    uint offset = as_type<uint>(labels[row]) & 0x80000000u ? kProductMassWords : 0;
    mass_atomic_product(mass + offset, labels[row], weights[row]);
    mass_atomic_float(mass + 2 * kProductMassWords, weights[row]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < words) partial[ulong(block) * words + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}
kernel void distribution_mean_reduce(device const uint* partial [[buffer(0)]],
                                       device InitialMean* output [[buffer(1)]],
                                       constant GradientParams& p [[buffer(2)]],
                                       uint lane [[thread_position_in_grid]]) {
  if (lane) return;
  auto positive = mass_zero<kProductMassWords>(), negative = mass_zero<kProductMassWords>();
  auto weight = mass_zero();
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  for (ulong i = 0; i < tiles; ++i) {
    auto entry = partial + i * (2 * kProductMassWords + kObjectiveMassWords);
    positive = mass_sum(positive, mass_load<kProductMassWords>(entry));
    negative = mass_sum(negative, mass_load<kProductMassWords>(entry + kProductMassWords));
    weight = mass_sum(weight, mass_load(entry + 2 * kProductMassWords));
  }
  if (mass_empty(weight)) { *output = InitialMean{0, 0}; return; }
  bool minus = mass_compare(positive, negative) < 0;
  auto magnitude = minus ? mass_subtract(negative, positive) : mass_subtract(positive, negative);
  int exponent = 0;
  float2 ratio = mass_ratio(magnitude, weight, exponent, -298);
  *output = InitialMean{extended_result(extended_normalize(minus ? -ratio : ratio, exponent)), 1};
}

kernel void normal_variance_partial(device const float* labels [[buffer(0)]],
                                     device const float* weights [[buffer(1)]],
                                     device const InitialMean* mean [[buffer(2)]],
                                     device uint* partial [[buffer(3)]],
                                     constant GradientParams& p [[buffer(4)]],
                                     uint block [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint words = 2 * kVarianceMassWords + kObjectiveMassWords;
  threadgroup atomic_uint mass[words];
  if (lane < words) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  float center = mean->value;
  for (ulong row = begin + lane; row < end; row += 256) {
    float label = labels[row], weight = weights[row];
    // Accumulate w*y*y + w*m*m - 2*w*y*m exactly before subtraction. The
    // integer products avoid both overflow and cancellation for finite labels.
    mass_atomic_triple(mass, label, label, weight);
    mass_atomic_triple(mass, center, center, weight);
    bool different_sign = (as_type<uint>(label) ^ as_type<uint>(center)) & 0x80000000u;
    mass_atomic_triple(mass + (different_sign ? 0 : kVarianceMassWords), label, center, weight, 1);
    mass_atomic_float(mass + 2 * kVarianceMassWords, weight);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < words) partial[ulong(block) * words + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}
kernel void normal_variance_reduce(device const uint* partial [[buffer(0)]],
                                    device const InitialMean* mean [[buffer(1)]],
                                    device float* output [[buffer(2)]],
                                    constant GradientParams& p [[buffer(3)]],
                                    uint lane [[thread_position_in_grid]]) {
  if (lane) return;
  output[0] = mean->value;
  if (!mean->valid) { output[1] = 0; return; }
  auto positive = mass_zero<kVarianceMassWords>(), negative = mass_zero<kVarianceMassWords>();
  auto weight = mass_zero();
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  for (ulong i = 0; i < tiles; ++i) {
    auto entry = partial + i * (2 * kVarianceMassWords + kObjectiveMassWords);
    positive = mass_sum(positive, mass_load<kVarianceMassWords>(entry));
    negative = mass_sum(negative, mass_load<kVarianceMassWords>(entry + kVarianceMassWords));
    weight = mass_sum(weight, mass_load(entry + 2 * kVarianceMassWords));
  }
  auto sum = mass_subtract(positive, negative);
  int exponent = 0;
  float2 ratio = mass_ratio(sum, weight, exponent, -447);
  Extended variance = extended_normalize(ratio, exponent);
  output[1] = extended_log(extended_add(variance, extended_float(0x1p-23f)));
}

inline float softplus(float value) {
  return value > 0 ? value + log1p_stable(exp(-value)) : log1p_stable(exp(value));
}
inline float expectile_sigmoid(float value) {
  float denominator = exp(min(-value, 88.7f)) + 1;
  return extended_result(extended_divide(extended_float(1), extended_float(denominator)));
}
inline float signed_product(float a, float b) {
  uint sign = (as_type<uint>(a) ^ as_type<uint>(b)) & 0x80000000u;
  return as_type<float>(positive_product_bits(a, b) | sign);
}

kernel void expectile_intercept_partial(device const float* labels [[buffer(0)]],
                                        device const float* weights [[buffer(1)]],
                                        device const InitialMean* mean [[buffer(2)]],
                                        device const float* alphas [[buffer(3)]],
                                        device uint* partial [[buffer(4)]],
                                        constant GradientParams& p [[buffer(5)]],
                                        uint2 block [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint words = 3 * kObjectiveMassWords + 1;
  threadgroup atomic_uint mass[words];
  if (lane < words) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block.x) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  float alpha = alphas[block.y];
  for (ulong row = begin + lane; row < end; row += 256) {
    float difference = mean->value - labels[row];
    float scale = difference >= 0 ? 1 - alpha : alpha;
    uint gradient = positive_product_bits(scale * difference, weights[row]);
    uint hessian = positive_product_bits(scale, weights[row]);
    if (gradient >= 0x7f800000u || hessian >= 0x7f800000u)
      atomic_store_explicit(mass + 3 * kObjectiveMassWords, 1, memory_order_relaxed);
    else {
      mass_atomic_bits(mass + (difference < 0 ? kObjectiveMassWords : 0), gradient);
      mass_atomic_bits(mass + 2 * kObjectiveMassWords, hessian);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  if (lane < words) partial[(ulong(block.y) * tiles + block.x) * words + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}
kernel void expectile_intercept_reduce(device const uint* partial [[buffer(0)]],
                                       device const InitialMean* mean [[buffer(1)]],
                                       device float* output [[buffer(2)]],
                                       constant GradientParams& p [[buffer(3)]],
                                       uint group [[thread_position_in_grid]]) {
  if (group >= p.groups) return;
  if (!mean->valid) { output[group] = 0; return; }
  auto positive = mass_zero(), negative = mass_zero(), hessian = mass_zero();
  uint invalid = 0;
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  for (ulong i = 0; i < tiles; ++i) {
    auto entry = partial + (ulong(group) * tiles + i) * (3 * kObjectiveMassWords + 1);
    positive = mass_sum(positive, mass_load(entry));
    negative = mass_sum(negative, mass_load(entry + kObjectiveMassWords));
    hessian = mass_sum(hessian, mass_load(entry + 2 * kObjectiveMassWords));
    invalid |= entry[3 * kObjectiveMassWords];
  }
  if (invalid) { output[group] = NAN; return; }
  bool minus = mass_compare(positive, negative) < 0;
  auto magnitude = minus ? mass_subtract(negative, positive) : mass_subtract(positive, negative);
  auto floor = mass_float(1.0e-6f);
  if (mass_compare(hessian, floor) < 0) hessian = floor;
  int exponent = 0;
  float2 ratio = mass_ratio(magnitude, hessian, exponent);
  float delta = extended_result(extended_normalize(minus ? ratio : -ratio, exponent));
  output[group] = mean->value + delta;
}
kernel void expectile_intercept_order(device float* output [[buffer(0)]],
                                      constant GradientParams& p [[buffer(1)]],
                                      uint lane [[thread_position_in_grid]]) {
  if (lane) return;
  for (uint i = 1; i < p.groups; ++i) if (output[i] < output[i - 1]) output[i] = output[i - 1];
}
kernel void expectile_gradients(device const float* predictions [[buffer(0)]],
                                 device const float* labels [[buffer(1)]],
                                 device const float* weights [[buffer(2)]],
                                 device const float* alphas [[buffer(3)]],
                                 device Pair* output [[buffer(4)]],
                                 constant GradientParams& p [[buffer(5)]],
                                 uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  ulong start = ulong(id.x) * p.groups;
  float prediction = predictions[start], gradient = 0, hessian = 0, label = labels[id.x];
  for (uint k = 0; k < p.groups; ++k) {
    if (k) prediction += 1.0e-6f + softplus(predictions[start + k]);
    if (k >= id.y) {
      float difference = prediction - label;
      float scale = difference >= 0 ? 1 - alphas[k] : alphas[k];
      gradient += signed_product(scale * difference, weights[id.x]);
      hessian += signed_product(scale, weights[id.x]);
    }
  }
  float scale = id.y ? expectile_sigmoid(predictions[start + id.y]) : 1;
  output[start + id.y] = Pair{signed_product(scale, gradient), signed_product(signed_product(scale, scale), hessian)};
}

inline Pair normal_finite_pair(Extended gradient, Extended hessian) {
  Extended magnitude = gradient.significand.x < 0 ? extended_negate(gradient) : gradient;
  if (extended_greater(hessian, magnitude)) magnitude = hessian;
  // sqrt(FLT_MAX)/2, including its low component. The final clamp resolves the
  // rounding boundary immediately below the midpoint to the next float.
  Extended limit{float2(0x1.fffffep-1f, 0x1p-25f), 63};
  if (extended_greater(magnitude, limit)) {
    Extended scale = extended_divide(limit, magnitude);
    gradient = extended_multiply(gradient, scale);
    hessian = extended_multiply(hessian, scale);
  }
  return Pair{clamp(extended_result(gradient), -0x1.fffffep62f, 0x1.fffffep62f),
              min(extended_result(hessian), 0x1.fffffep62f)};
}
kernel void normal_gradients(device const float* predictions [[buffer(0)]],
                              device const float* labels [[buffer(1)]],
                              device const float* weights [[buffer(2)]],
                              device Pair* output [[buffer(3)]],
                              constant GradientParams& p [[buffer(4)]],
                              uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  ulong start = ulong(row) * 2;
  if (as_type<uint>(weights[row]) << 1 == 0) { output[start] = Pair{0, 0}; output[start + 1] = Pair{0, 0}; return; }
  Extended one = extended_float(1), weight = extended_float(weights[row]);
  Extended residual = extended_add(extended_float(predictions[start]), extended_negate(extended_float(labels[row])));
  float log_variance = predictions[start + 1];
  Extended maximum = extended_float(as_type<float>(0x7f7fffffu));
  // Adjacent float32 inputs straddle the double-precision log(FLT_MAX) boundary.
  Extended precision = log_variance <= -88.72283935546875f ? maximum :
                       log_variance >= 88.72283935546875f ? extended_divide(one, maximum) : extended_exp(-log_variance);
  Extended standardized = extended_multiply(
      extended_add(extended_multiply(residual, residual), extended_float(0x1p-23f)), precision);
  Extended weighted_precision = extended_multiply(weight, precision);
  output[start] = normal_finite_pair(extended_multiply(weighted_precision, residual), weighted_precision);
  Extended gradient = extended_multiply(extended_float(0.5f), extended_add(one, extended_negate(standardized)));
  Extended hessian = extended_divide(extended_add(one, extended_multiply(extended_float(2), standardized)), extended_float(6));
  output[start + 1] = normal_finite_pair(extended_multiply(weight, gradient), extended_multiply(weight, hessian));
}

kernel void normal_metric_partial(device const float* predictions [[buffer(0)]],
                                   device const float* labels [[buffer(1)]],
                                   device const float* weights [[buffer(2)]],
                                   device uint* partial [[buffer(3)]],
                                   constant MetricParams& p [[buffer(4)]],
                                   uint block [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
  constexpr uint words = 3 * kObjectiveMassWords + 1;
  threadgroup atomic_uint mass[words];
  if (lane < words) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  for (ulong row = begin + lane; row < end; row += 256) {
    float residual = labels[row] - predictions[row * 2], log_variance = predictions[row * 2 + 1];
    float standardized = residual == 0 ? 0.0f : exp(2 * log(fabs(residual)) - log_variance);
    float loss = 0.5f * (1.8378770664093453f + log_variance + standardized);
    uint product = positive_product_bits(loss, weights[row]);
    bool negative = as_type<uint>(loss) & 0x80000000u;
    if (product >= 0x7f800000u)
      atomic_fetch_or_explicit(mass + 3 * kObjectiveMassWords, product > 0x7f800000u ? 4u : negative ? 2u : 1u, memory_order_relaxed);
    else mass_atomic_bits(mass + (negative ? kObjectiveMassWords : 0), product);
    mass_atomic_float(mass + 2 * kObjectiveMassWords, weights[row]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < words) partial[ulong(block) * words + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}
kernel void normal_metric_reduce(device const uint* partial [[buffer(0)]],
                                  device MetricRatio* output [[buffer(1)]],
                                  constant MetricParams& p [[buffer(2)]],
                                  uint lane [[thread_position_in_grid]]) {
  if (lane) return;
  auto positive = mass_zero(), negative = mass_zero(), weight = mass_zero();
  uint invalid = 0;
  ulong count = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  for (ulong i = 0; i < count; ++i) {
    auto entry = partial + i * (3 * kObjectiveMassWords + 1);
    positive = mass_sum(positive, mass_load(entry));
    negative = mass_sum(negative, mass_load(entry + kObjectiveMassWords));
    weight = mass_sum(weight, mass_load(entry + 2 * kObjectiveMassWords));
    invalid |= entry[3 * kObjectiveMassWords];
  }
  if (invalid) {
    float value = (invalid & 4) || ((invalid & 3) == 3) ? NAN : invalid & 2 ? -INFINITY : INFINITY;
    *output = MetricRatio{value, 0, 0, 1}; return;
  }
  if (mass_empty(weight)) { *output = MetricRatio{0, 0, 0, 1}; return; }
  bool minus = mass_compare(positive, negative) < 0;
  auto magnitude = minus ? mass_subtract(negative, positive) : mass_subtract(positive, negative);
  int exponent = 0;
  float2 ratio = mass_ratio(magnitude, weight, exponent);
  if (minus) ratio = -ratio;
  *output = MetricRatio{ratio.x, ratio.y, exponent, 1};
}
