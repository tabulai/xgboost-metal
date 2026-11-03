// Host-streamlining kernels for GPU-driven trees. Row identities are written
// on the GPU instead of by a host loop, and asynchronous rounds emit the host
// replay's Node table so the next round's margin update needs no CPU model.
// The host still replays every tree from the same history and compares its
// Node bytes with this table before publishing the tree.

constant uint kTreeEmitThreads = 256;
constant uint kTreeEmitNodes = 2048;
constant uint kTreeEmitSplitWords = 24;
constant uint kTreeEmitNodeWords = 12;

kernel void tree_rows_init(device uint* rows [[buffer(0)]],
                           constant uint& count [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
  if (id < count) rows[id] = id;
}

// IEEE binary32 multiplication with round-to-nearest-even and gradual
// underflow, as AArch64 FMUL computes it with FPCR.FZ=0 and FPCR.DN=0. Metal
// arithmetic flushes subnormals, so the exact 48-bit product of the two
// 24-bit significands is rounded explicitly. NaN operands follow the AArch64
// order: first signaling NaN, else first quiet NaN, returned quieted.
inline uint tree_emit_multiply(uint a, uint b) {
  uint sign = (a ^ b) & 0x80000000u;
  uint power_a = (a >> 23) & 255, power_b = (b >> 23) & 255;
  uint fraction_a = a & 0x7fffffu, fraction_b = b & 0x7fffffu;
  bool nan_a = power_a == 255 && fraction_a, nan_b = power_b == 255 && fraction_b;
  if (nan_a || nan_b) {
    bool signal_a = nan_a && !(a & 0x400000u), signal_b = nan_b && !(b & 0x400000u);
    return (signal_a ? a : signal_b ? b : nan_a ? a : b) | 0x400000u;
  }
  bool zero_a = !power_a && !fraction_a, zero_b = !power_b && !fraction_b;
  if (power_a == 255 || power_b == 255) return zero_a || zero_b ? 0x7fc00000u : sign | 0x7f800000u;
  if (zero_a || zero_b) return sign;
  // value = significand x 2^(exponent - 150), significand in [2^23, 2^24).
  int exponent_a = int(power_a), exponent_b = int(power_b);
  uint significand_a = fraction_a | 0x800000u, significand_b = fraction_b | 0x800000u;
  if (!power_a) { uint shift = clz(fraction_a) - 8; significand_a = fraction_a << shift; exponent_a = 1 - int(shift); }
  if (!power_b) { uint shift = clz(fraction_b) - 8; significand_b = fraction_b << shift; exponent_b = 1 - int(shift); }
  // The exact product lies in [2^46, 2^48) units of 2^(exponents - 300).
  ulong product = ulong(significand_a) * ulong(significand_b);
  int lead = product >> 47 ? 47 : 46;
  int biased = lead + exponent_a + exponent_b - 173;
  // Subnormal results keep fewer significand bits; the units stay 2^-149.
  int shift = lead - 23 + max(0, 1 - biased);
  if (shift > 48) return sign;
  ulong kept = product >> shift, rest = product & ((1ul << shift) - 1), midpoint = 1ul << (shift - 1);
  if (rest > midpoint || (rest == midpoint && (kept & 1))) ++kept;
  if (biased <= 0) return sign | uint(kept);  // 2^23 encodes the smallest normal
  if (kept >> 24) { kept >>= 1; ++biased; }
  if (biased >= 255) return sign | 0x7f800000u;
  return sign | (uint(biased) << 23) | (uint(kept) & 0x7fffffu);
}

// Validation entry point for exhaustive leaf-scaling sweeps; training does
// not use it. Each thread scales one consecutive binary32 bit pattern.
kernel void tree_emit_multiply_sweep(device uint* output [[buffer(0)]],
                                     constant TreeEmitSweepParams& p [[buffer(1)]],
                                     uint id [[thread_position_in_grid]]) {
  if (id < p.count) output[id] = tree_emit_multiply(p.begin + id, as_type<uint>(p.eta));
}

// One threadgroup of at most kTreeEmitThreads threads emits the Node table
// that the host replay builds from TreeDriverState, flags and Split history.
// Fields are copied as raw words.
// Expansions occur in increasing node-ID order, so an expanded node's left
// child is 1 + 2 x the number of expanded IDs before it. Covers and leaves
// follow the replay: creation values from the parent split, then evaluated
// totals, then applied parent statistics for expanded nodes. An invalid
// driver state emits one zero leaf, keeping the consumer's traversal in
// bounds until the host rejects that round.
kernel void tree_emit_nodes(device const TreeDriverState* state [[buffer(0)]],
                            device const uint* flags [[buffer(1)]],
                            device const uint* history [[buffer(2)]],
                            device uint* nodes [[buffer(3)]],
                            device TreeMeta* meta [[buffer(4)]],
                            constant TreeEmitParams& p [[buffer(5)]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup ushort parent_of[kTreeEmitNodes];
  threadgroup ushort first_child[kTreeEmitNodes];
  threadgroup uint totals[kTreeEmitThreads / 32];
  TreeDriverState s = state[0];
  uint capacity = min(p.node_capacity, kTreeEmitNodes);
  if (s.error || !s.node_count || s.node_count > capacity) {
    if (tid < kTreeEmitNodeWords) nodes[tid] = tid < 4 ? 0xffffffffu : 0u;
    if (!tid) meta[0] = TreeMeta{0, 1, p.group, p.round, 0, 1, 0};
    return;
  }
  uint count = s.node_count;
  uint chunk = (count + threads - 1) / threads;
  uint begin = min(tid * chunk, count), end = min(begin + chunk, count);
  uint expanded = 0;
  for (uint id = begin; id < end; ++id) expanded += (flags[id] & kTreeDriverExpanded) != 0;
  uint prefix = simd_prefix_exclusive_sum(expanded);
  uint simd_total = simd_sum(expanded);
  if (simd_is_first()) totals[simd] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!simd) {
    uint total = lane < simds ? totals[lane] : 0;
    uint scanned = simd_prefix_exclusive_sum(total);
    if (lane < simds) totals[lane] = scanned;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint created = 1 + 2 * (totals[simd] + prefix);
  if (!tid) parent_of[0] = 0xffffu;
  for (uint id = begin; id < end; ++id) {
    if (!(flags[id] & kTreeDriverExpanded)) continue;
    uint left = created;
    created += 2;
    if (left + 1 >= count) continue;
    first_child[id] = ushort(left);
    parent_of[left] = parent_of[left + 1] = ushort(id);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint eta = as_type<uint>(p.eta);
  for (uint id = begin; id < end; ++id) {
    uint node[kTreeEmitNodeWords] = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0, 0, 0, 0, 0, 0, 0, 0};
    uint status = flags[id];
    if (id) {
      uint parent = parent_of[id];
      if (parent < id) {
        device const uint* split = history + ulong(parent) * kTreeEmitSplitWords;
        bool left = id & 1u;
        node[3] = parent;
        node[5] = tree_emit_multiply(split[left ? 8 : 9], eta);  // left/right weight
        node[6] = split[left ? 3 : 5];                           // left/right Hessian
      }
    } else if (status & kTreeDriverEvaluated) {
      node[5] = tree_emit_multiply(history[15], eta);            // parent weight
    }
    if (status & kTreeDriverEvaluated) {
      device const uint* split = history + ulong(id) * kTreeEmitSplitWords;
      node[6] = split[11];                                       // total Hessian
      if ((status & kTreeDriverExpanded) && first_child[id] + 1u < count) {
        uint left = first_child[id];
        node[0] = left;
        node[1] = left + 1;
        node[2] = split[6];                                      // feature
        node[4] = split[12] ? 1u : split[1];                     // threshold
        node[6] = split[4];                                      // applied parent Hessian
        node[7] = split[0];                                      // gain
        node[8] = split[7];                                      // default left
        node[9] = split[12];                                     // split type
      }
    }
    for (uint word = 0; word < kTreeEmitNodeWords; ++word)
      nodes[ulong(id) * kTreeEmitNodeWords + word] = node[word];
  }
  if (!tid) meta[0] = TreeMeta{0, count, p.group, p.round, 0, 1, 0};
}
