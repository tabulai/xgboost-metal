// The GPU sort/scan/sample sequence follows CUDA XGBoost. The accumulation
// order and random stream preserve the pinned CPU reference's public behavior.
inline ulong sampling_random_state(uint seed, uint row) {
  constexpr ulong mask = 0x7ffffffffffffffful;
  ulong exponent = ulong(row) + 1, factor = 16807, state = seed ? seed : 1;
  while (exponent) {
    if (exponent & 1) state = (state * factor) & mask;
    factor = (factor * factor) & mask;
    exponent >>= 1;
  }
  return state;
}

inline bool sampling_uniform(ulong state, float probability) {
  // libc++'s bernoulli_distribution draws one float64 canonical value from
  // this 63-bit engine. Round its integer numerator to 53 bits explicitly;
  // the final denominator is exactly 2^63. No float64 Metal type is needed.
  ulong numerator = state - 1;
  uint bits = 64 - uint(clz(numerator));
  if (bits > 53) {
    uint shift = bits - 53;
    ulong lower = numerator & ((1ul << shift) - 1), midpoint = 1ul << (shift - 1);
    ulong upper = numerator >> shift;
    upper += ulong(lower > midpoint || (lower == midpoint && (upper & 1)));
    numerator = upper << shift;
  }
  float limit = probability * 0x1p63f;
  return limit < 0x1p24f ? float(numerator) < limit : numerator < ulong(limit);
}

kernel void sampling_magnitudes(device const Pair* gradients [[buffer(0)]],
                                  device float* magnitude [[buffer(1)]], device float2* sorted [[buffer(2)]],
                                  constant SamplingParams& p [[buffer(3)]],
                                  uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float values[256];
  uint row = block * 256 + tid;
  float value = INFINITY;
  if (row < p.rows) {
    float sum = 0;
    for (uint target = 0; target < max(p.padding, 1u); ++target) {
      auto pair = gradients[ulong(row) * p.groups + p.group + target];
      float grad_square = pair.g * pair.g, hess_square = pair.h * pair.h;
      sum += grad_square + 0.1f * hess_square;
    }
    value = sqrt(sum);
    magnitude[row] = value;
  }
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) for (uint j = k >> 1; j; j >>= 1) {
    float a = values[tid], b = values[tid ^ j];
    bool take_min = ((tid & k) == 0) == ((tid & j) == 0);
    float next = take_min ? min(a, b) : max(a, b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  sorted[row] = float2(values[tid], 0);
}

kernel void sampling_prefix(device const float2* sorted [[buffer(0)]], device float* prefix [[buffer(1)]],
                              constant SamplingParams& p [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) {
  // Replicate the ordered f32 accumulator across one SIMD group. Coalesced
  // loads and register broadcasts avoid a shared-memory load for every add.
  // The additions still occur in exactly the CPU partial_sum order.
  float total = 0;
  for (ulong begin = 0; begin < p.rows; begin += 256) {
    float values[8], results[8];
    #pragma clang loop unroll(full)
    for (uint tile = 0; tile < 8; ++tile) {
      ulong row = begin + tile * 32 + lane;
      values[tile] = row < p.rows ? sorted[row].x : 0;
    }
    #pragma clang loop unroll(full)
    for (uint tile = 0; tile < 8; ++tile) {
      float result = 0;
      #pragma clang loop unroll(full)
      for (uint source = 0; source < 32; ++source) {
        total += simd_broadcast(values[tile], source);
        result = select(result, total, lane == source);
      }
      results[tile] = result;
    }
    #pragma clang loop unroll(full)
    for (uint tile = 0; tile < 8; ++tile) {
      ulong row = begin + tile * 32 + lane;
      if (row < p.rows) prefix[row] = results[tile];
    }
  }
}

kernel void sampling_threshold(device const float2* sorted [[buffer(0)]], device const float* prefix [[buffer(1)]],
                                 device float* threshold [[buffer(2)]], constant SamplingParams& p [[buffer(3)]]) {
  long lo = 0, hi = long(p.rows) - 1;
  while (lo <= hi) {
    long index = lo + (hi - lo) / 2;
    float lower = sorted[index].x;
    float upper = index + 1 < p.rows ? sorted[index + 1].x : as_type<float>(0x7f7fffffu);
    float denominator = float(p.sample_rows) - float(ulong(p.rows) - ulong(index) - 1);
    if (denominator <= 0) { lo = index + 1; continue; }
    float value = prefix[index] / denominator;
    if (value > lower && value <= upper) { threshold[0] = value; return; }
    if (value <= lower) hi = index - 1;
    else lo = index + 1;
  }
  threshold[0] = prefix[p.rows - 1] / float(p.sample_rows);
}

kernel void sampling_apply(device const Pair* gradients [[buffer(0)]], device const float* magnitude [[buffer(1)]],
                             device const float* threshold [[buffer(2)]], device Pair* output [[buffer(3)]],
                             constant SamplingParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint targets = max(p.padding, 1u);
  if (p.method && !p.sample_rows) {
    for (uint target = 0; target < targets; ++target) output[ulong(row) * targets + target] = Pair{0, 0};
    return;
  }
  ulong state = sampling_random_state(p.seed, row);
  if (!p.method) {
    bool keep = sampling_uniform(state, p.subsample);
    for (uint target = 0; target < targets; ++target)
      output[ulong(row) * targets + target] = keep ? gradients[ulong(row) * p.groups + p.group + target] : Pair{0, 0};
    return;
  }
  float value = threshold[0];
  if (fabs(value) < 1.0e-6f) value = copysign(1.0e-6f, value);
  float probability = isinf(value) ? 0 : magnitude[row] / value;
  float random = float(state - 1) * 0x1p-63f;
  bool keep = probability >= 1 || (probability > 0 && random <= probability);
  float inverse = probability >= 1 ? 1 : 1.0f / probability;
  for (uint target = 0; target < targets; ++target) {
    auto pair = gradients[ulong(row) * p.groups + p.group + target];
    output[ulong(row) * targets + target] = probability >= 1 ? pair : keep ? Pair{pair.g * inverse, pair.h * inverse} : Pair{0, 0};
  }
}
