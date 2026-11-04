// GPU row compaction for subsampled trees, monotone bound propagation, and
// exact MT19937 Fisher-Yates column sampling for depthwise GPU trees.

kernel void tree_rows_compact_sampled(device const long2* gradients [[buffer(0)]],
                                      device uint* rows [[buffer(1)]],
                                      device atomic_uint* sampled_count [[buffer(2)]],
                                      constant HistParams& p [[buffer(3)]],
                                      uint block [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint warp_sums[8];
  threadgroup uint tile_base;
  uint row = block * 256 + tid;
  bool keep = false;
  if (row < p.rows) {
    long2 q = gradients[ulong(row) * p.groups + p.group];
    keep = (q.x != 0 || q.y != 0);
  }
  uint prefix = simd_prefix_exclusive_sum(uint(keep));
  uint sum = simd_sum(uint(keep));
  if ((tid & 31) == 0) warp_sums[tid >> 5] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0;
  for (uint w = 0; w < 8; ++w) {
    if (w < (tid >> 5)) prefix += warp_sums[w];
    total += warp_sums[w];
  }
  if (tid == 0)
    tile_base = total ? atomic_fetch_add_explicit(sampled_count, total, memory_order_relaxed) : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (keep) rows[tile_base + prefix] = row;
}

kernel void tree_init_bounds(device float2* bank_bounds [[buffer(0)]],
                             device float2* history_bounds [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  float2 full(-FLT_MAX, FLT_MAX);
  bank_bounds[0] = full;
  history_bounds[0] = full;
}

kernel void tree_record_node_bounds(device const TreeDriverState* state [[buffer(0)]],
                                    device const uint* ids [[buffer(1)]],
                                    device const float2* current_bounds [[buffer(2)]],
                                    device float2* history_bounds [[buffer(3)]],
                                    uint slot [[thread_position_in_grid]]) {
  TreeDriverState s = state[0];
  if (s.error || slot >= s.frontier_count) return;
  history_bounds[ids[slot]] = current_bounds[slot];
}

kernel void tree_propagate_child_bounds(device const TreeDriverState* state [[buffer(0)]],
                                        device const uint* parent_indices [[buffer(1)]],
                                        device const Split* selected_splits [[buffer(2)]],
                                        device const int* monotone [[buffer(3)]],
                                        device const float2* parent_bounds [[buffer(4)]],
                                        device float2* child_bounds [[buffer(5)]],
                                        uint rank [[thread_position_in_grid]]) {
  TreeDriverState s = state[0];
  uint parents = s.error ? 0u : s.frontier_count / 2u;
  if (rank >= parents) return;
  uint slot = parent_indices[rank];
  Split split = selected_splits[rank];
  float2 lb = parent_bounds[slot], rb = lb;
  int constraint = split.feature >= 0 ? monotone[split.feature] : 0;
  if (constraint != 0) {
    float sum = scalar_float_add(split.left_weight, split.right_weight);
    float mid = as_type<float>(tree_emit_multiply(as_type<uint>(sum), 0x3f000000u));
    if (constraint > 0) {
      if (fp32_less(mid, lb.y)) lb.y = mid;
      if (fp32_less(rb.x, mid)) rb.x = mid;
    } else {
      if (fp32_less(lb.x, mid)) lb.x = mid;
      if (fp32_less(mid, rb.y)) rb.y = mid;
    }
  }
  child_bounds[2 * rank] = lb;
  child_bounds[2 * rank + 1] = rb;
}

inline void mt19937_seed(threadgroup uint* mt, uint seed) {
  mt[0] = seed;
  for (uint i = 1; i < 624; ++i)
    mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
}

// Exact libc++ std::mersenne_twister_engine::operator()() incremental twist.
inline uint mt19937_next(threadgroup uint* mt, thread uint& idx) {
  uint j = idx + 1 == 624 ? 0u : idx + 1;
  uint yp = (mt[idx] & 0x80000000u) | (mt[j] & 0x7fffffffu);
  uint k = idx + 397 >= 624 ? idx + 397 - 624 : idx + 397;
  uint x = mt[k] ^ (yp >> 1) ^ (0x9908b0dfu * (yp & 1u));
  mt[idx] = x;
  idx = j;
  uint z = x ^ (x >> 11);
  z ^= (z << 7) & 0x9d2c5680u;
  z ^= (z << 15) & 0xefc60000u;
  return z ^ (z >> 18);
}

// Exact libc++ std::shuffle prefix [0, keep) over `n` elements in `work`.
inline void mt19937_shuffle_prefix(threadgroup uint* mt, threadgroup uint* work,
                                   uint n, uint keep, uint seed) {
  mt19937_seed(mt, seed);
  uint idx = 0;
  for (uint pos = 0; pos < keep && pos + 1 < n; ++pos) {
    uint rp = n - pos;
    uint mask = 0xffffffffu >> clz(rp - 1u);
    uint u;
    do {
      u = mt19937_next(mt, idx) & mask;
    } while (u >= rp);
    if (u != 0) {
      uint tmp = work[pos];
      work[pos] = work[pos + u];
      work[pos + u] = tmp;
    }
  }
}

kernel void tree_sample_level(device const TreeDriverState* state [[buffer(0)]],
                              device const uint* seeds [[buffer(1)]],
                              device const uint* tree_features [[buffer(2)]],
                              device uint* level_features [[buffer(3)]],
                              device uint* rng_cursor [[buffer(4)]],
                              device uint* node_seed_base [[buffer(5)]],
                              device TreeDriverDispatch* sample_node_args [[buffer(6)]],
                              constant TreeSampleParams& p [[buffer(7)]],
                              uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint mt[624];
  threadgroup uint work[2048];
  threadgroup atomic_uint bits[64];
  threadgroup uint half_total;

  TreeDriverState s = state[0];
  uint nodes = s.error ? 0u : s.frontier_count;
  uint cursor = rng_cursor[0];
  if (lane == 0) {
    sample_node_args[0] = TreeDriverDispatch{nodes, 1u, 1u};
    if (nodes > 0) {
      node_seed_base[0] = cursor + (p.bylevel ? 1u : 0u);
      rng_cursor[0] = cursor + (p.bylevel ? 1u : 0u) + (p.bynode ? nodes : 0u);
    }
  }
  if (nodes == 0) return;

  if (!p.bylevel) {
    for (uint i = lane; i < p.tree_features; i += 32)
      level_features[i] = tree_features[i];
    return;
  }

  for (uint i = lane; i < p.tree_features; i += 32)
    work[i] = tree_features[i];
  for (uint w = lane; w < 64; w += 32)
    atomic_store_explicit(bits + w, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane == 0)
    mt19937_shuffle_prefix(mt, work, p.tree_features, p.level_features, seeds[cursor]);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lane; i < p.level_features; i += 32) {
    uint f = work[i];
    atomic_fetch_or_explicit(bits + (f >> 5), 1u << (f & 31u), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint word0 = atomic_load_explicit(bits + lane, memory_order_relaxed);
  uint c0 = popcount(word0);
  uint off0 = simd_prefix_exclusive_sum(c0);
  uint sum0 = simd_sum(c0);
  if (lane == 0) half_total = sum0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  while (word0 != 0u) {
    uint bit = ctz(word0);
    level_features[off0++] = (lane << 5) + bit;
    word0 &= word0 - 1u;
  }

  uint word1 = atomic_load_explicit(bits + 32 + lane, memory_order_relaxed);
  uint c1 = popcount(word1);
  uint off1 = half_total + simd_prefix_exclusive_sum(c1);
  while (word1 != 0u) {
    uint bit = ctz(word1);
    level_features[off1++] = ((32u + lane) << 5) + bit;
    word1 &= word1 - 1u;
  }
}

kernel void tree_sample_nodes(device const uint* seeds [[buffer(0)]],
                              device const uint* level_features [[buffer(1)]],
                              device const uint* node_seed_base [[buffer(2)]],
                              device uchar* allowed [[buffer(3)]],
                              constant TreeSampleParams& p [[buffer(4)]],
                              uint node [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]]) {
  threadgroup uint mt[624];
  threadgroup uint work[2048];

  device uchar* row = allowed + ulong(node) * p.cols;
  for (uint c = lane; c < p.cols; c += 32) row[c] = 0;

  if (!p.bynode) {
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = lane; i < p.level_features; i += 32)
      row[level_features[i]] = 1;
    return;
  }

  for (uint i = lane; i < p.level_features; i += 32)
    work[i] = level_features[i];
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  if (lane == 0) {
    uint seed = seeds[node_seed_base[0] + node];
    mt19937_shuffle_prefix(mt, work, p.level_features, p.node_features, seed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = lane; i < p.node_features; i += 32)
    row[work[i]] = 1;
}
