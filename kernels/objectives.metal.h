// Float32 weights are integers in units of 2^-149. Ten 32-bit limbs hold the
// exact sum of any UINT32_MAX finite nonnegative weights, including subnormals.
// This replaces CUDA's float64 atomics while retaining order-independent ranks.
template<uint N> struct ExactMass { uint word[N]; };
using ObjectiveMass = ExactMass<kObjectiveMassWords>;

template<uint N = kObjectiveMassWords>
inline ExactMass<N> mass_zero() {
  ExactMass<N> value;
  for (uint i = 0; i < N; ++i) value.word[i] = 0;
  return value;
}
template<uint N = kObjectiveMassWords>
inline ExactMass<N> mass_load(device const uint* input) {
  ExactMass<N> value;
  for (uint i = 0; i < N; ++i) value.word[i] = input[i];
  return value;
}
template<uint N>
inline bool mass_empty(ExactMass<N> value) {
  uint bits = 0;
  for (uint i = 0; i < N; ++i) bits |= value.word[i];
  return bits == 0;
}
template<uint N>
inline int mass_compare(ExactMass<N> a, ExactMass<N> b) {
  for (uint i = N; i-- > 0;)
    if (a.word[i] != b.word[i]) return a.word[i] < b.word[i] ? -1 : 1;
  return 0;
}
template<uint N>
inline ExactMass<N> mass_sum(ExactMass<N> a, ExactMass<N> b) {
  ulong carry = 0;
  for (uint i = 0; i < N; ++i) {
    ulong value = ulong(a.word[i]) + b.word[i] + carry;
    a.word[i] = uint(value); carry = value >> 32;
  }
  return a;
}
template<uint N>
inline ExactMass<N> mass_subtract(ExactMass<N> a, ExactMass<N> b) {
  ulong borrow = 0;
  for (uint i = 0; i < N; ++i) {
    ulong value = ulong(b.word[i]) + borrow;
    uint previous = a.word[i];
    a.word[i] -= uint(value); borrow = ulong(previous) < value;
  }
  return a;
}
template<uint N = kObjectiveMassWords, class Storage>
inline void mass_atomic_word(Storage output, uint word, uint value) {
  while (value && word < N) {
    uint before = atomic_fetch_add_explicit(output + word, value, memory_order_relaxed);
    value = uint(before + value < before);
    ++word;
  }
}
template<class Storage>
inline void mass_atomic_bits(Storage output, uint bits) {
  uint exponent = (bits >> 23) & 255;
  uint mantissa = (bits & 0x7fffffu) | (exponent ? 0x800000u : 0u);
  uint shift = exponent ? exponent - 1 : 0, word = shift / 32;
  shift %= 32;
  mass_atomic_word(output, word, mantissa << shift);
  if (shift) mass_atomic_word(output, word + 1, mantissa >> (32 - shift));
}
template<class Storage>
inline void mass_atomic_float(Storage output, float value) {
  mass_atomic_bits(output, as_type<uint>(value));
}
inline ObjectiveMass mass_rank(ObjectiveMass total, float alpha) {
  // All cumulative masses are integers, so comparison with alpha * total is
  // equivalent to comparison with its exact ceiling. Do not round a rank to f32.
  uint bits = as_type<uint>(alpha), exponent = (bits >> 23) & 255;
  uint mantissa = (bits & 0x7fffffu) | (exponent ? 0x800000u : 0u);
  uint shift = exponent ? 150 - exponent : 149;
  uint product[kObjectiveMassWords + 1];
  ulong carry = 0;
  for (uint i = 0; i < kObjectiveMassWords; ++i) {
    ulong value = ulong(total.word[i]) * mantissa + carry;
    product[i] = uint(value); carry = value >> 32;
  }
  product[kObjectiveMassWords] = uint(carry);
  ObjectiveMass rank = mass_zero();
  uint begin = shift / 32, remainder = shift % 32;
  for (uint i = 0; i < kObjectiveMassWords; ++i) {
    uint source = i + begin;
    if (source <= kObjectiveMassWords) rank.word[i] = product[source] >> remainder;
    if (remainder && source < kObjectiveMassWords) rank.word[i] |= product[source + 1] << (32 - remainder);
  }
  uint discarded = 0;
  for (uint i = 0; i < begin; ++i) discarded |= product[i];
  if (remainder) discarded |= product[begin] & ((1u << remainder) - 1);
  if (discarded) {
    ObjectiveMass one = mass_zero(); one.word[0] = 1;
    rank = mass_sum(rank, one);
  }
  return rank;
}

kernel void quantile_radix_histogram(device const float* labels [[buffer(0)]],
                                     device const float* weights [[buffer(1)]],
                                     device const QuantileState* states [[buffer(2)]],
                                     device atomic_uint* histogram [[buffer(3)]],
                                     constant QuantileParams& p [[buffer(4)]],
                                     uint2 block [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint local[256 * kObjectiveMassWords];
  for (uint i = lane; i < 256 * kObjectiveMassWords; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint output = block.y, column = output / p.alphas;
  uint shift = 24 - 8 * p.pass, mask = p.pass ? 0xffffffffu << (shift + 8) : 0;
  uint prefix = states[output].prefix;
  ulong begin = ulong(block.x) * kObjectiveRowsPerTile;
  ulong end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  for (ulong row = begin + lane; row < end; row += 256) {
    uint bits = as_type<uint>(labels[row * p.label_cols + column]);
    uint key = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    if ((key & mask) == prefix)
      mass_atomic_float(local + ((key >> shift) & 255) * kObjectiveMassWords, weights[row]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto global = histogram + (ulong(output) * 256 + lane) * kObjectiveMassWords;
  for (uint i = 0; i < kObjectiveMassWords; ++i)
    mass_atomic_word(global, i, atomic_load_explicit(local + lane * kObjectiveMassWords + i, memory_order_relaxed));
}

kernel void quantile_radix_select(device const uint* histogram [[buffer(0)]],
                                  device const float* alpha [[buffer(1)]],
                                  device QuantileState* states [[buffer(2)]],
                                  device float* output [[buffer(3)]],
                                  constant QuantileParams& p [[buffer(4)]],
                                  uint group [[thread_position_in_grid]]) {
  if (group >= p.outputs) return;
  auto bins = histogram + ulong(group) * 256 * kObjectiveMassWords;
  float level = alpha[group % p.alphas];
  ObjectiveMass rank = mass_zero();
  if (p.pass == 0) {
    ObjectiveMass total = mass_zero();
    for (uint b = 0; b < 256; ++b) total = mass_sum(total, mass_load(bins + b * kObjectiveMassWords));
    states[group].valid = !mass_empty(total);
    rank = mass_rank(total, level);
  } else for (uint i = 0; i < kObjectiveMassWords; ++i) rank.word[i] = states[group].rank[i];
  if (!states[group].valid) { output[group] = 0; return; }
  ObjectiveMass sum = mass_zero(), before = mass_zero(), selected_weight = mass_zero();
  uint selected = 0;
  for (uint bin = 0; bin < 256; ++bin) {
    ObjectiveMass weight = mass_load(bins + bin * kObjectiveMassWords);
    if (!mass_empty(weight)) {
      selected = bin; before = sum; selected_weight = weight;
    }
    sum = mass_sum(sum, weight);
    if (!mass_empty(weight) && (level == 0 || (level != 1 && mass_compare(sum, rank) >= 0))) break;
  }
  rank = mass_compare(rank, before) < 0 ? mass_zero() : mass_subtract(rank, before);
  if (mass_compare(rank, selected_weight) > 0) rank = selected_weight;
  states[group].prefix |= selected << (24 - p.pass * 8);
  for (uint i = 0; i < kObjectiveMassWords; ++i) states[group].rank[i] = rank.word[i];
  if (p.pass == 3) {
    uint key = states[group].prefix;
    output[group] = as_type<float>(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
  }
}

inline float2 accurate_add(float2 a, float2 b) {
  float sum = a.x + b.x, back = sum - a.x;
  float error = (a.x - (sum - back)) + (b.x - back) + a.y + b.y;
  float high = sum + error;
  return float2(high, error - (high - sum));
}
inline float2 accurate_multiply(float2 a, float2 b) {
  float product = a.x * b.x;
  float error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
  float high = product + error;
  return float2(high, error - (high - product));
}
template<uint N>
inline float2 mass_normalized(ExactMass<N> value, thread int& exponent, int unit = -149) {
  uint last = N - 1;
  while (last && value.word[last] == 0) --last;
  int highest = int(last * 32 + 31 - clz(value.word[last]));
  exponent = highest + unit;
  float2 result = 0;
  // Three limbs provide more bits than the two-float result can retain.
  for (int i = int(last); i >= max(0, int(last) - 2); --i) {
    result = accurate_add(result, float2(ldexp(float(value.word[i] >> 16), i * 32 + 16 - highest), 0));
    result = accurate_add(result, float2(ldexp(float(value.word[i] & 65535), i * 32 - highest), 0));
  }
  return result;
}
template<uint N, uint M>
inline float2 mass_ratio(ExactMass<N> numerator, ExactMass<M> denominator, thread int& exponent,
                          int numerator_unit = -149, int denominator_unit = -149) {
  exponent = 0;
  if (mass_empty(numerator)) return float2(0);
  int numerator_exponent = 0, denominator_exponent = 0;
  float2 a = mass_normalized(numerator, numerator_exponent, numerator_unit);
  float2 b = mass_normalized(denominator, denominator_exponent, denominator_unit);
  float high = a.x / b.x;
  float2 remainder = accurate_add(a, -accurate_multiply(b, float2(high, 0)));
  float low = (remainder.x + remainder.y) / b.x;
  exponent = numerator_exponent - denominator_exponent;
  return accurate_add(float2(high, 0), float2(low, 0));
}

inline ulong rounded_shift(ulong value, int shift) {
  if (shift <= 0) return value << -shift;
  if (shift >= 64) return 0;  // Products below have at most 48 significant bits.
  ulong result = value >> shift, tail = value & ((1ul << shift) - 1), midpoint = 1ul << (shift - 1);
  return result + uint(tail > midpoint || (tail == midpoint && (result & 1)));
}
inline uint positive_product_bits(float a, float b) {
  uint ab = as_type<uint>(a) & 0x7fffffffu, bb = as_type<uint>(b) & 0x7fffffffu;
  uint ae = ab >> 23, be = bb >> 23;
  if (ae == 255 || be == 255) {
    if (ab > 0x7f800000u || bb > 0x7f800000u || !ab || !bb) return 0x7fc00000u;
    return 0x7f800000u;
  }
  // Normal products use the native instruction. Handle subnormal inputs and
  // results with integer rounding so Metal's denormal flushing loses no mass.
  if (ae && be && ae + be >= 128) return as_type<uint>(a * b) & 0x7fffffffu;
  uint am = (ab & 0x7fffffu) | (ae ? 0x800000u : 0), bm = (bb & 0x7fffffu) | (be ? 0x800000u : 0);
  ulong product = ulong(am) * bm;
  if (!product) return 0;
  int power = (ae ? int(ae) - 150 : -149) + (be ? int(be) - 150 : -149);
  uint upper = uint(product >> 32);
  int leading = upper ? 63 - int(clz(upper)) : 31 - int(clz(uint(product)));
  int exponent = power + leading;
  if (exponent < -126) return uint(rounded_shift(product, -149 - power));
  ulong mantissa = rounded_shift(product, leading - 23);
  if (mantissa == 0x1000000ul) { mantissa >>= 1; ++exponent; }
  if (exponent > 127) return 0x7f800000u;
  return (uint(exponent + 127) << 23) | (uint(mantissa) & 0x7fffffu);
}
kernel void residual_scale_partial(device const float* predictions [[buffer(0)]],
                                    device const float* labels [[buffer(1)]],
                                    device const float* weights [[buffer(2)]],
                                    device uint* partial [[buffer(3)]],
                                    constant GradientParams& p [[buffer(4)]],
                                    uint2 block [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint mass[2 * kObjectiveMassWords + 1];
  if (lane < 2 * kObjectiveMassWords + 1) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block.x) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  for (ulong row = begin + lane; row < end; row += 256) {
    float residual = predictions[row * p.groups + block.y] - labels[row * p.label_cols + min(block.y, p.label_cols - 1)];
    float weight = weights[row], root = weight * sqrt(fabs(residual));
    if (isfinite(root)) mass_atomic_float(mass, root);
    else atomic_store_explicit(mass + 2 * kObjectiveMassWords, 1, memory_order_relaxed);
    mass_atomic_float(mass + kObjectiveMassWords, weight);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  ulong offset = (ulong(block.y) * tiles + block.x) * (2 * kObjectiveMassWords + 1);
  if (lane < 2 * kObjectiveMassWords + 1) partial[offset + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}
kernel void residual_scale_reduce(device const uint* partial [[buffer(0)]],
                                   device float* scale [[buffer(1)]],
                                   constant GradientParams& p [[buffer(2)]],
                                   uint group [[thread_position_in_grid]]) {
  if (group >= p.groups) return;
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  ObjectiveMass root = mass_zero(), weight = mass_zero();
  uint invalid = 0;
  for (ulong i = 0; i < tiles; ++i) {
    auto entry = partial + (ulong(group) * tiles + i) * (2 * kObjectiveMassWords + 1);
    root = mass_sum(root, mass_load(entry));
    weight = mass_sum(weight, mass_load(entry + kObjectiveMassWords));
    invalid |= entry[2 * kObjectiveMassWords];
  }
  if (invalid) { scale[group] = INFINITY; return; }
  if (mass_empty(root) || mass_empty(weight)) { scale[group] = 0; return; }
  int exponent = 0;
  float2 mean = mass_ratio(root, weight, exponent);
  float2 squared = accurate_multiply(mean, mean);
  scale[group] = ldexp(squared.x + squared.y, 2 * exponent);
}
kernel void smoothed_gradients(device const float* predictions [[buffer(0)]],
                                device const float* labels [[buffer(1)]],
                                device const float* weights [[buffer(2)]],
                                device const float* scale [[buffer(3)]],
                                device const float* alpha [[buffer(4)]],
                                device Pair* output [[buffer(5)]],
                                constant GradientParams& p [[buffer(6)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  ulong index = ulong(id.x) * p.groups + id.y;
  float residual = predictions[index] - labels[ulong(id.x) * p.label_cols + min(id.y, p.label_cols - 1)];
  float delta = scale[id.y], weight = weights[id.x], g = 0, h = 0;
  if (p.objective == 11) {
    h = delta == 0 && residual == 0 ? 1 : huber_curvature(delta, residual);
    g = weight * residual * h; h *= weight;
  } else if (delta > 0 && weight != 0) {
    float x = residual / (0.04f * delta), t = tanh(x);
    float ratio = max(x == 0 ? 1.0f : t / x, 3.0e-4f);
    g = weight * (0.5f * delta * (t + 1 - 2 * alpha[id.y]));
    h = weight * (12.5f * ratio);
  }
  output[index] = Pair{g, h};
}

kernel void alpha_metric_partial(device const float* predictions [[buffer(0)]],
                                     device const float* labels [[buffer(1)]],
                                     device const float* weights [[buffer(2)]],
                                     device const float* alphas [[buffer(3)]],
                                     device uint* partial [[buffer(4)]],
                                     constant MetricParams& p [[buffer(5)]],
                                     uint2 block [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint mass[2 * kObjectiveMassWords + 1];
  if (lane < 2 * kObjectiveMassWords + 1) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block.x) * kObjectiveRowsPerTile, end = min(begin + kObjectiveRowsPerTile, ulong(p.rows));
  float alpha = alphas[block.y / p.label_cols];
  for (ulong row = begin + lane; row < end; row += 256) {
    float residual = predictions[row * p.groups + block.y] - labels[row * p.label_cols + block.y % p.label_cols];
    float scale = residual >= 0 ? 1 - alpha : alpha;
    float loss = p.type == 16 ? (scale * residual) * residual : scale * fabs(residual);
    uint product = positive_product_bits(loss, weights[row]);
    if (product >= 0x7f800000u)
      atomic_fetch_or_explicit(mass + 2 * kObjectiveMassWords, product == 0x7f800000u ? 1u : 2u, memory_order_relaxed);
    else mass_atomic_bits(mass, product);
    mass_atomic_float(mass + kObjectiveMassWords, weights[row]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong tiles = (ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile;
  ulong offset = (ulong(block.y) * tiles + block.x) * (2 * kObjectiveMassWords + 1);
  if (lane < 2 * kObjectiveMassWords + 1) partial[offset + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}

kernel void alpha_metric_reduce(device const uint* partial [[buffer(0)]],
                                    device MetricRatio* output [[buffer(1)]],
                                    constant MetricParams& p [[buffer(2)]],
                                    uint lane [[thread_position_in_grid]]) {
  if (lane) return;
  ulong count = ((ulong(p.rows) + kObjectiveRowsPerTile - 1) / kObjectiveRowsPerTile) * p.groups;
  ObjectiveMass loss = mass_zero(), weight = mass_zero();
  uint invalid = 0;
  for (ulong i = 0; i < count; ++i) {
    auto entry = partial + i * (2 * kObjectiveMassWords + 1);
    loss = mass_sum(loss, mass_load(entry));
    weight = mass_sum(weight, mass_load(entry + kObjectiveMassWords));
    invalid |= entry[2 * kObjectiveMassWords];
  }
  if (mass_empty(weight)) { *output = MetricRatio{0, 0, 0, 0}; return; }
  if (invalid) { *output = MetricRatio{invalid & 2 ? NAN : INFINITY, 0, 0, 1}; return; }
  int exponent = 0;
  float2 ratio = mass_ratio(loss, weight, exponent);
  *output = MetricRatio{ratio.x, ratio.y, exponent, 1};
}
