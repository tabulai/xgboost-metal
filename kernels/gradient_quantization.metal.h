// CUDA budgets 62 bits for reproducible signed gradient sums. Use an exact
// magnitude bound here: float32 magnitudes are integers in units of 2^-149,
// and ten limbs also cover UINT32_MAX rows of the largest finite float32.
kernel void gradient_partial_sums(device const Pair* gradient [[buffer(0)]],
                                   device uint* partial [[buffer(1)]],
                                   constant GradientParams& p [[buffer(2)]],
                                   uint2 block [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint mass[kGradientPartialWords];
  if (lane < kGradientPartialWords) atomic_store_explicit(mass + lane, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong begin = ulong(block.x) * kGradientRowsPerTile;
  ulong end = min(begin + kGradientRowsPerTile, ulong(p.rows));
  for (ulong row = begin + lane; row < end; row += 256) {
    auto value = gradient[row * p.groups + block.y];
    mass_atomic_bits(mass, as_type<uint>(value.g) & 0x7fffffffu);
    mass_atomic_bits(mass + kObjectiveMassWords, as_type<uint>(value.h) & 0x7fffffffu);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong tiles = (ulong(p.rows) + kGradientRowsPerTile - 1) / kGradientRowsPerTile;
  ulong offset = (ulong(block.y) * tiles + block.x) * kGradientPartialWords;
  if (lane < kGradientPartialWords)
    partial[offset + lane] = atomic_load_explicit(mass + lane, memory_order_relaxed);
}

kernel void gradient_scales(device const uint* partial [[buffer(0)]],
                             device float2* scales [[buffer(1)]],
                             constant GradientParams& p [[buffer(2)]],
                             uint2 block [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[256 * kObjectiveMassWords];
  ulong tiles = (ulong(p.rows) + kGradientRowsPerTile - 1) / kGradientRowsPerTile;
  auto total = mass_zero();
  for (ulong tile = lane; tile < tiles; tile += 256) {
    ulong offset = ((ulong(block.y) * tiles + tile) * 2 + block.x) * kObjectiveMassWords;
    total = mass_sum(total, mass_load(partial + offset));
  }
  for (uint word = 0; word < kObjectiveMassWords; ++word)
    sums[lane * kObjectiveMassWords + word] = total.word[word];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (lane < delta) {
      ObjectiveMass other;
      for (uint word = 0; word < kObjectiveMassWords; ++word)
        other.word[word] = sums[(lane + delta) * kObjectiveMassWords + word];
      total = mass_sum(total, other);
      for (uint word = 0; word < kObjectiveMassWords; ++word)
        sums[lane * kObjectiveMassWords + word] = total.word[word];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!lane) {
    int exponent = 0;
    for (uint word = kObjectiveMassWords; word-- > 0;) if (total.word[word]) {
      // Strict next power, as in CUDA CreateRoundingFactor; retain a minimum
      // magnitude bound of one. The scale exponent lies in [-98, 62].
      exponent = max(0, int(word * 32 + 32 - clz(total.word[word])) - 149);
      break;
    }
    // The scaled absolute sum is at most 2^62. Nearest-even quantization adds
    // at most rows/2, keeping every partial signed sum strictly below 2^63.
    reinterpret_cast<device float*>(scales)[block.y * 2 + block.x] =
        as_type<float>(uint(62 - exponent + 127) << 23);
    if (block.x == 0)
      reinterpret_cast<device GradCert*>(scales + p.groups)[block.y] = GradCert{0, 0, 0, 0, 0, 0, 0, 0};
  }
}
