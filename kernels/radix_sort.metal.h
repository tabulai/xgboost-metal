// 4-pass 8-bit SIMDgroup LSD radix sort, parallel tile-segmented binary64
// ROC-AUC / PR-AUC reduction, GPU matrix/bin slicing, and fused classifier
// probability epilogue kernels.

inline void radix_multisplit_8bit(bool valid, uint digit, uint lane,
                                  thread uint& sg_rank, thread uint& sg_count) {
  uint active_mask = uint(simd_vote::vote_t(simd_ballot(valid)));
  uint match_mask = active_mask;
  #pragma clang loop unroll(full)
  for (uint b = 0; b < 8; ++b) {
    bool bit = (digit >> b) & 1u;
    uint ballot = uint(simd_vote::vote_t(simd_ballot(bit)));
    match_mask &= bit ? ballot : ~ballot;
  }
  uint lower_mask = lane ? ((1u << lane) - 1u) : 0u;
  sg_rank = uint(popcount(match_mask & lower_mask));
  sg_count = uint(popcount(match_mask));
}

// Pass 0 fused initialization and digit histogram for descending float32 order.
// ~fp32_order_key(pred) maps descending IEEE-754 order (including subnormals
// and canonicalized +-0.0f) onto ascending uint32 keys.
kernel void radix_init_desc_histogram(device const float* predictions [[buffer(0)]],
                                      device uint2* keys_out [[buffer(1)]],
                                      device uint* spine [[buffer(2)]],
                                      constant RadixParams& p [[buffer(3)]],
                                      uint2 block [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint local_counts[8][257];
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) local_counts[w][tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  ulong group_base = ulong(block.y) * p.rows;
  uint tile_base = block.x * kRadixTileRows;
  for (uint step = 0; step < 8; ++step) {
    uint row = tile_base + step * 256 + tid;
    bool valid = row < p.rows;
    uint key = 0;
    if (valid) {
      float pred = predictions[ulong(row) * p.groups + block.y];
      key = ~fp32_order_key(pred);
      keys_out[group_base + row] = uint2(key, row);
    }
    uint digit = valid ? (key & 255u) : 0u;
    uint sg_rank, sg_count;
    radix_multisplit_8bit(valid, digit, lane, sg_rank, sg_count);
    if (valid && sg_rank == 0) local_counts[sg][digit] += sg_count;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint total = 0;
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) total += local_counts[w][tid];
  spine[(ulong(block.y) * 256 + tid) * p.tiles + block.x] = total;
}

kernel void radix_histogram(device const uint2* keys_in [[buffer(0)]],
                            device uint* spine [[buffer(1)]],
                            constant RadixParams& p [[buffer(2)]],
                            uint2 block [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint local_counts[8][257];
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) local_counts[w][tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  ulong group_base = ulong(block.y) * p.rows;
  uint tile_base = block.x * kRadixTileRows;
  for (uint step = 0; step < 8; ++step) {
    uint row = tile_base + step * 256 + tid;
    bool valid = row < p.rows;
    uint key = valid ? keys_in[group_base + row].x : 0u;
    uint digit = valid ? ((key >> p.shift) & 255u) : 0u;
    uint sg_rank, sg_count;
    radix_multisplit_8bit(valid, digit, lane, sg_rank, sg_count);
    if (valid && sg_rank == 0) local_counts[sg][digit] += sg_count;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint total = 0;
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) total += local_counts[w][tid];
  spine[(ulong(block.y) * 256 + tid) * p.tiles + block.x] = total;
}

kernel void sampling_radix_init(device const Pair* gradients [[buffer(0)]],
                                device float* magnitude [[buffer(1)]],
                                device uint2* keys_out [[buffer(2)]],
                                device uint* spine [[buffer(3)]],
                                constant SamplingParams& sp [[buffer(4)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint local_counts[8][257];
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) local_counts[w][tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint tiles = (sp.rows + kRadixTileRows - 1) / kRadixTileRows;
  uint tile_base = block.x * kRadixTileRows;
  for (uint step = 0; step < 8; ++step) {
    uint row = tile_base + step * 256 + tid;
    bool valid = row < sp.rows;
    uint key = 0;
    if (valid) {
      float sum = 0;
      for (uint target = 0; target < max(sp.padding, 1u); ++target) {
        auto pair = gradients[ulong(row) * sp.groups + sp.group + target];
        sum += pair.g * pair.g + 0.1f * pair.h * pair.h;
      }
      float value = sqrt(sum);
      magnitude[row] = value;
      key = fp32_order_key(value);
      keys_out[row] = uint2(key, as_type<uint>(value));
    }
    uint digit = valid ? (key & 255u) : 0u;
    uint sg_rank, sg_count;
    radix_multisplit_8bit(valid, digit, lane, sg_rank, sg_count);
    if (valid && sg_rank == 0) local_counts[sg][digit] += sg_count;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint total = 0;
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) total += local_counts[w][tid];
  spine[ulong(tid) * tiles + block.x] = total;
}

kernel void sampling_radix_unpack(device const uint2* keys_in [[buffer(0)]],
                                  device float2* sorted [[buffer(1)]],
                                  constant RadixParams& p [[buffer(2)]],
                                  uint row [[thread_position_in_grid]]) {
  if (row < p.rows) sorted[row] = float2(as_type<float>(keys_in[row].y), 0.0f);
}

kernel void survival_radix_init(device const float* labels [[buffer(0)]],
                                device uint2* keys_out [[buffer(1)]],
                                device uint* spine [[buffer(2)]],
                                constant RadixParams& p [[buffer(3)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint local_counts[8][257];
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) local_counts[w][tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint tile_base = block.x * kRadixTileRows;
  for (uint step = 0; step < 8; ++step) {
    uint row = tile_base + step * 256 + tid;
    bool valid = row < p.rows;
    uint key = 0;
    if (valid) {
      key = as_type<uint>(labels[row]) & 0x7fffffffu;
      keys_out[row] = uint2(key, row);
    }
    uint digit = valid ? (key & 255u) : 0u;
    uint sg_rank, sg_count;
    radix_multisplit_8bit(valid, digit, lane, sg_rank, sg_count);
    if (valid && sg_rank == 0) local_counts[sg][digit] += sg_count;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint total = 0;
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) total += local_counts[w][tid];
  spine[ulong(tid) * p.tiles + block.x] = total;
}

kernel void radix_extract_payload(device const uint2* keys_in [[buffer(0)]],
                                  device uint* output [[buffer(1)]],
                                  constant RadixParams& p [[buffer(2)]],
                                  uint row [[thread_position_in_grid]]) {
  if (row < p.rows) output[row] = keys_in[row].y;
}

kernel void radix_scan_spine(device uint* spine [[buffer(0)]],
                             device uint* digit_totals [[buffer(1)]],
                             constant RadixParams& p [[buffer(2)]],
                             uint2 block [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint sg_sums[8];
  uint carry = 0;

  ulong base_idx = (ulong(block.y) * 256 + block.x) * p.tiles;
  for (uint base = 0; base < p.tiles; base += 256) {
    uint idx = base + tid;
    uint val = idx < p.tiles ? spine[base_idx + idx] : 0u;
    uint prefix = simd_prefix_exclusive_sum(val);
    uint sum = simd_sum(val);
    if (lane == 0) sg_sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint sg_offset = carry;
    for (uint w = 0; w < sg; ++w) sg_offset += sg_sums[w];
    if (idx < p.tiles) spine[base_idx + idx] = sg_offset + prefix;
    for (uint w = 0; w < 8; ++w) carry += sg_sums[w];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) digit_totals[ulong(block.y) * 256 + block.x] = carry;
}

kernel void radix_scatter(device const uint2* keys_in [[buffer(0)]],
                          device uint2* keys_out [[buffer(1)]],
                          device const uint* spine [[buffer(2)]],
                          device const uint* digit_totals [[buffer(3)]],
                          constant RadixParams& p [[buffer(4)]],
                          uint2 block [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint warp_hist[8][257];
  threadgroup uint bin_offset[256];
  threadgroup uint local_digit_base[256];
  threadgroup uint digit_global_start[256];
  threadgroup uint sg_sums[8];
  threadgroup uint2 stage_item[256];
  threadgroup uint stage_dst[256];

  uint d_total = digit_totals[ulong(block.y) * 256 + tid];
  uint tile_off = spine[(ulong(block.y) * 256 + tid) * p.tiles + block.x];
  uint d_prefix = simd_prefix_exclusive_sum(d_total);
  uint d_sum = simd_sum(d_total);
  if (lane == 0) sg_sums[sg] = d_sum;
  #pragma clang loop unroll(full)
  for (uint w = 0; w < 8; ++w) warp_hist[w][tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint d_sg_base = 0;
  for (uint w = 0; w < sg; ++w) d_sg_base += sg_sums[w];
  bin_offset[tid] = d_sg_base + d_prefix + tile_off;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  ulong group_base = ulong(block.y) * p.rows;
  uint tile_base = block.x * kRadixTileRows;
  for (uint step = 0; step < 8; ++step) {
    uint row = tile_base + step * 256 + tid;
    bool valid = row < p.rows;
    uint2 item = valid ? keys_in[group_base + row] : uint2(0);
    uint digit = valid ? ((item.x >> p.shift) & 255u) : 0u;
    uint sg_rank, sg_count;
    radix_multisplit_8bit(valid, digit, lane, sg_rank, sg_count);
    if (valid && sg_rank == 0) warp_hist[sg][digit] = sg_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint base_global = bin_offset[tid];
    uint step_count = 0;
    #pragma clang loop unroll(full)
    for (uint w = 0; w < 8; ++w) {
      uint c = warp_hist[w][tid];
      warp_hist[w][tid] = step_count;
      step_count += c;
    }
    bin_offset[tid] = base_global + step_count;

    uint pfx = simd_prefix_exclusive_sum(step_count);
    uint ssum = simd_sum(step_count);
    if (lane == 0) sg_sums[sg] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint sg_base = 0;
    for (uint w = 0; w < sg; ++w) sg_base += sg_sums[w];
    uint total_valid = sg_base;
    for (uint w = sg; w < 8; ++w) total_valid += sg_sums[w];
    local_digit_base[tid] = sg_base + pfx;
    digit_global_start[tid] = base_global;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
      uint rank_in_digit = warp_hist[sg][digit] + sg_rank;
      uint local_pos = local_digit_base[digit] + rank_in_digit;
      stage_item[local_pos] = item;
      stage_dst[local_pos] = digit_global_start[digit] + rank_in_digit;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < total_valid) keys_out[group_base + stage_dst[tid]] = stage_item[tid];
    #pragma clang loop unroll(full)
    for (uint w = 0; w < 8; ++w) warp_hist[w][tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Software binary64 pair operations for order-accurate, overflow-immune AUC/PR.
inline ulong2 auc_pair_add(ulong2 a, ulong2 b) {
  return ulong2(cox_add_bits(a.x, b.x), cox_add_bits(a.y, b.y));
}

inline ulong auc_u32_to_d64(uint bits) {
  uint power = (bits >> 23) & 255u, mantissa = bits & 0x7fffffu;
  ulong sign = ulong(bits & 0x80000000u) << 32;
  if (power == 255u) return sign | 0x7ff0000000000000ul | (ulong(mantissa) << 29);
  if (!power) {
    if (!mantissa) return sign;
    uint shift = uint(clz(mantissa)) - 8u;
    mantissa <<= shift;
    return sign | (ulong(897u - shift) << 52) | (ulong(mantissa & 0x7fffffu) << 29);
  }
  return sign | (ulong(power + 896u) << 52) | (ulong(mantissa) << 29);
}

inline ulong2 auc_load_tp_fp(device const float* labels, device const float* weights,
                             uint row, uint group, constant MetricParams& p) {
  float y = labels[ulong(row) * p.label_cols + min(group, p.label_cols - 1)];
  if (p.multiclass) y = float(uint(y) == group);
  float w = weights[row];
  return ulong2(auc_u32_to_d64(positive_product_bits(y, w)),
                auc_u32_to_d64(positive_product_bits(1.0f - y, w)));
}

inline ulong cox_atanh2_bits(ulong z) {
  constexpr ulong kOne = 0x3ff0000000000000ul;
  ulong z2 = cox_multiply_bits(z, z);
  ulong poly = cox_divide_bits(kOne, cox_float_bits(17.0f));
  for (int k = 15; k >= 3; k -= 2) {
    poly = cox_fma_bits(z2, poly, cox_divide_bits(kOne, cox_float_bits(float(k))));
  }
  poly = cox_fma_bits(z2, poly, kOne);
  return cox_scale_bits(cox_multiply_bits(z, poly), 1);
}

inline ulong cox_log_bits(ulong x) {
  constexpr ulong kOne = 0x3ff0000000000000ul;
  constexpr ulong kLn2Hi = 0x3fe62e42fefa39eful;
  constexpr ulong kLn2Lo = 0x3c7abc9e3b39803ful;
  int exp = 0;
  ulong sig = cox_finite_significand(x, exp);
  ulong m = (1023ul << 52) | (sig & kCoxFraction);
  if ((sig & kCoxFraction) >= 0x0006a09e667f3bcdul) {
    m = cox_scale_bits(m, -1);
    ++exp;
  }
  ulong z = cox_divide_bits(cox_add_bits(m, kOne ^ kCoxSign), cox_add_bits(m, kOne));
  ulong lnm = cox_atanh2_bits(z);
  if (!exp) return lnm;
  ulong exp_d64 = cox_float_bits(float(exp));
  ulong corr = cox_fma_bits(exp_d64, kLn2Lo, lnm);
  return cox_fma_bits(exp_d64, kLn2Hi, corr);
}

inline ulong cox_log1p_pos_bits(ulong u) {
  constexpr ulong kOne = 0x3ff0000000000000ul;
  constexpr ulong kTwo = 0x4000000000000000ul;
  if ((u & ~kCoxSign) <= 0x3fd8000000000000ul) {
    return cox_atanh2_bits(cox_divide_bits(u, cox_add_bits(kTwo, u)));
  }
  return cox_log_bits(cox_add_bits(kOne, u));
}

inline ulong auc_pr_segment_d64(ulong2 previous, ulong2 step, ulong total_positives) {
  constexpr ulong kOne = 0x3ff0000000000000ul;
  constexpr ulong kTwo = 0x4000000000000000ul;
  constexpr ulong kQuarter = 0x3fd0000000000000ul;
  if (!(step.x & ~kCoxSign) || !(total_positives & ~kCoxSign)) return 0ul;
  ulong mass_d64 = cox_add_bits(step.x, step.y);
  if (!(mass_d64 & ~kCoxSign)) return 0ul;
  ulong rise_d64 = cox_divide_bits(step.x, total_positives);
  ulong q_d64 = cox_divide_bits(step.x, mass_d64);
  ulong cumulative_d64 = cox_add_bits(previous.x, previous.y);
  if (!(cumulative_d64 & ~kCoxSign)) return cox_multiply_bits(rise_d64, q_d64);

  ulong r_d64 = cox_divide_bits(previous.x, cumulative_d64);
  ulong retained_d64, added_d64;
  if ((mass_d64 & ~kCoxSign) <= (cumulative_d64 & ~kCoxSign)) {
    ulong u = cox_divide_bits(mass_d64, cumulative_d64);
    if ((u & ~kCoxSign) <= kQuarter) {
      ulong z = cox_divide_bits(u, cox_add_bits(kTwo, u));
      ulong poly = cox_divide_bits(cox_float_bits(16.0f), cox_float_bits(17.0f));
      for (int k = 7; k >= 1; --k) {
        poly = cox_fma_bits(z, poly, kOne);
        ulong even_over_odd = cox_divide_bits(cox_float_bits(float(2 * k)), cox_float_bits(float(2 * k + 1)));
        poly = cox_fma_bits(z, poly, even_over_odd);
      }
      poly = cox_fma_bits(z, poly, kOne);
      added_d64 = cox_multiply_bits(cox_multiply_bits(z, cox_add_bits(kOne, z ^ kCoxSign)), poly);
      retained_d64 = cox_add_bits(kOne, added_d64 ^ kCoxSign);
    } else {
      retained_d64 = cox_divide_bits(cox_log1p_pos_bits(u), u);
      added_d64 = cox_add_bits(kOne, retained_d64 ^ kCoxSign);
    }
  } else {
    ulong v = cox_divide_bits(cumulative_d64, mass_d64);
    retained_d64 = !(v & ~kCoxSign) ? 0ul
                                    : cox_multiply_bits(v, cox_add_bits(cox_log1p_pos_bits(v), cox_log_bits(v) ^ kCoxSign));
    added_d64 = cox_add_bits(kOne, retained_d64 ^ kCoxSign);
  }
  ulong blend = cox_add_bits(cox_multiply_bits(r_d64, retained_d64), cox_multiply_bits(q_d64, added_d64));
  return cox_multiply_bits(rise_d64, blend);
}

inline ulong2 auc_simd_incl_scan(ulong2 v, uint lane, thread ulong2& sg_total) {
  #pragma clang loop unroll(full)
  for (ushort d = 1; d < 32; d <<= 1) {
    ulong2 prev = as_type<ulong2>(simd_shuffle_up(as_type<uint4>(v), d));
    if (lane >= d) v = auc_pair_add(prev, v);
  }
  sg_total = as_type<ulong2>(simd_broadcast(as_type<uint4>(v), 31));
  return v;
}

inline void auc_simd_suffix_carry(thread ulong2& seed, thread bool& cont, uint lane) {
  #pragma clang loop unroll(full)
  for (ushort d = 1; d < 32; d <<= 1) {
    ulong2 nxt_seed = as_type<ulong2>(simd_shuffle_down(as_type<uint4>(seed), d));
    uint nxt_cont = simd_shuffle_down(uint(cont), d);
    if (lane + d < 32) {
      if (cont) seed = auc_pair_add(seed, nxt_seed);
      cont = cont && (nxt_cont != 0);
    }
  }
}

kernel void auc_tile_stats(device const uint2* sorted_keys [[buffer(0)]],
                           device const float* labels [[buffer(1)]],
                           device const float* weights [[buffer(2)]],
                           device AucTileSummary* summaries [[buffer(3)]],
                           constant MetricParams& p [[buffer(4)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ulong2 sg_sums[8];
  threadgroup uint sh_first_key[256];
  threadgroup uint sh_last_key[256];
  threadgroup uchar sh_single[256];

  uint tile_begin = block.x * kAucTileRows;
  uint tile_valid = tile_begin < p.rows ? min(uint(kAucTileRows), p.rows - tile_begin) : 0u;
  uint tiles = (p.rows + kAucTileRows - 1) / kAucTileRows;
  ulong group_base = ulong(block.y) * p.rows;

  uint base = tid * 8;
  uint n = base < tile_valid ? min(8u, tile_valid - base) : 0u;
  ulong2 thr_total(0), thr_prefix(0), thr_suffix(0);
  uint first_k = 0, last_k = 0;
  bool single = true;
  if (n > 0) {
    device const uint2* src = sorted_keys + group_base + tile_begin + base;
    first_k = src[0].x;
    last_k = src[n - 1].x;
    bool in_prefix = true;
    uint prev_k = first_k;
    for (uint j = 0; j < n; ++j) {
      uint2 item = src[j];
      ulong2 val = auc_load_tp_fp(labels, weights, item.y, block.y, p);
      thr_total = auc_pair_add(thr_total, val);
      if (in_prefix && item.x == first_k) thr_prefix = auc_pair_add(thr_prefix, val);
      else { in_prefix = false; single = false; }
      if (j == 0 || item.x != prev_k) thr_suffix = val;
      else thr_suffix = auc_pair_add(thr_suffix, val);
      prev_k = item.x;
    }
  }

  ulong2 sg_tot(0);
  auc_simd_incl_scan(thr_total, lane, sg_tot);
  if (lane == 0) sg_sums[sg] = sg_tot;
  sh_first_key[tid] = first_k;
  sh_last_key[tid] = last_k;
  sh_single[tid] = uchar(single);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint active_threads = (tile_valid + 7) / 8;
  uint tile_fk = active_threads ? sh_first_key[0] : 0u;
  uint tile_lk = active_threads ? sh_last_key[active_threads - 1] : 0u;

  // Parallel check for how far the leading tie-run and trailing tie-run extend.
  bool break_prefix = (tid < active_threads) && (sh_first_key[tid] != tile_fk || !sh_single[tid]);
  threadgroup atomic_uint first_break_idx;
  threadgroup atomic_uint last_break_idx;
  if (tid == 0) {
    atomic_store_explicit(&first_break_idx, active_threads, memory_order_relaxed);
    atomic_store_explicit(&last_break_idx, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (break_prefix) atomic_fetch_min_explicit(&first_break_idx, tid, memory_order_relaxed);
  bool break_suffix = (tid < active_threads) && (sh_last_key[tid] != tile_lk || !sh_single[tid]);
  if (break_suffix) atomic_fetch_max_explicit(&last_break_idx, tid + 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint fb = atomic_load_explicit(&first_break_idx, memory_order_relaxed);
  bool all_same = (active_threads > 0) && (fb == active_threads);
  ulong2 p_contrib(0);
  if (tid < active_threads) {
    if (tid < fb) p_contrib = thr_total;
    else if (tid == fb && sh_first_key[tid] == tile_fk) p_contrib = thr_prefix;
  }
  ulong2 p_sg(0);
  auc_simd_incl_scan(p_contrib, lane, p_sg);

  uint lb_plus1 = atomic_load_explicit(&last_break_idx, memory_order_relaxed);
  ulong2 s_contrib(0);
  if (!all_same && tid < active_threads) {
    if (tid >= lb_plus1) s_contrib = thr_total;
    else if (tid + 1u == lb_plus1 && sh_last_key[tid] == tile_lk) s_contrib = thr_suffix;
  }
  ulong2 s_sg(0);
  auc_simd_incl_scan(s_contrib, lane, s_sg);

  threadgroup ulong2 p_sg_sums[8];
  threadgroup ulong2 s_sg_sums[8];
  if (lane == 0) {
    p_sg_sums[sg] = p_sg;
    s_sg_sums[sg] = s_sg;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    AucTileSummary out{};
    out.valid_rows = tile_valid;
    if (tile_valid > 0) {
      ulong2 tot(0), prun(0), srun(0);
      for (uint w = 0; w < 8; ++w) {
        tot = auc_pair_add(tot, sg_sums[w]);
        prun = auc_pair_add(prun, p_sg_sums[w]);
        srun = auc_pair_add(srun, s_sg_sums[w]);
      }
      out.total_tp = tot.x;
      out.total_fp = tot.y;
      out.first_key = tile_fk;
      out.last_key = tile_lk;
      out.prefix_tp = prun.x;
      out.prefix_fp = prun.y;
      out.single_run = uint(all_same);
      out.suffix_tp = all_same ? tot.x : srun.x;
      out.suffix_fp = all_same ? tot.y : srun.y;
    }
    summaries[ulong(block.y) * tiles + block.x] = out;
  }
}

kernel void auc_scan_tiles(device const AucTileSummary* summaries [[buffer(0)]],
                           device AucTilePrefix* prefixes [[buffer(1)]],
                           device ulong2* class_totals [[buffer(2)]],
                           constant MetricParams& p [[buffer(3)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ulong2 sg_sums[8];
  threadgroup uchar sg_cont[8];
  threadgroup ulong2 carry_shared;

  uint tiles = (p.rows + kAucTileRows - 1) / kAucTileRows;
  ulong base = ulong(block.y) * tiles;
  if (tid == 0) {
    carry_shared = ulong2(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Parallel forward exclusive scan across tiles in 256-tile chunks.
  ulong2 running(0);
  for (uint chunk = 0; chunk < tiles; chunk += 256) {
    uint m = chunk + tid;
    ulong2 val = m < tiles ? ulong2(summaries[base + m].total_tp, summaries[base + m].total_fp) : ulong2(0);
    ulong2 sg_tot(0);
    ulong2 incl = auc_simd_incl_scan(val, lane, sg_tot);
    ulong2 excl_in_sg = lane ? as_type<ulong2>(simd_shuffle_up(as_type<uint4>(incl), 1)) : ulong2(0);
    if (lane == 0) sg_sums[sg] = sg_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong2 sg_off = running;
    for (uint w = 0; w < sg; ++w) sg_off = auc_pair_add(sg_off, sg_sums[w]);
    if (m < tiles) {
      ulong2 excl = auc_pair_add(sg_off, excl_in_sg);
      prefixes[base + m].excl_tp = excl.x;
      prefixes[base + m].excl_fp = excl.y;
    }
    for (uint w = 0; w < 8; ++w) running = auc_pair_add(running, sg_sums[w]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) class_totals[block.y] = running;

  // Parallel backward segmented tie-carry scan across tiles in 256-tile chunks.
  uint num_chunks = (tiles + 255) / 256;
  for (int ci = int(num_chunks) - 1; ci >= 0; --ci) {
    uint chunk = uint(ci) * 256;
    uint chunk_len = min(256u, tiles - chunk);
    uint m = chunk + tid;
    ulong2 seed(0);
    bool cont = false;
    if (tid < chunk_len) {
      if (m + 1 < tiles) {
        AucTileSummary cur = summaries[base + m];
        AucTileSummary nxt = summaries[base + m + 1];
        if (cur.valid_rows > 0 && nxt.valid_rows > 0 && cur.last_key == nxt.first_key) {
          seed = ulong2(nxt.prefix_tp, nxt.prefix_fp);
          if (tid + 1 == chunk_len) {
            if (nxt.single_run) seed = auc_pair_add(seed, carry_shared);
            cont = false;
          } else {
            cont = (nxt.single_run != 0);
          }
        }
      }
    }
    auc_simd_suffix_carry(seed, cont, lane);
    if (lane == 0) {
      sg_sums[sg] = seed;
      sg_cont[sg] = uchar(cont);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ulong2 sg_tail(0);
    for (int w = 7; w > int(sg); --w) {
      sg_tail = sg_cont[w] ? auc_pair_add(sg_sums[w], sg_tail) : sg_sums[w];
    }
    ulong2 my_carry = cont ? auc_pair_add(seed, sg_tail) : seed;
    if (tid < chunk_len) {
      prefixes[base + m].carry_tp = my_carry.x;
      prefixes[base + m].carry_fp = my_carry.y;
    }
    if (tid == 0) carry_shared = my_carry;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void auc_tile_integrate(device const uint2* sorted_keys [[buffer(0)]],
                               device const float* labels [[buffer(1)]],
                               device const float* weights [[buffer(2)]],
                               device const AucTileSummary* summaries [[buffer(3)]],
                               device const AucTilePrefix* prefixes [[buffer(4)]],
                               device const ulong2* class_totals [[buffer(5)]],
                               device ulong* tile_areas [[buffer(6)]],
                               constant MetricParams& p [[buffer(7)]],
                               uint2 block [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup ulong2 sh_prefix[256];
  threadgroup ulong2 sg_sums[8];
  threadgroup ulong sg_areas[8];
  threadgroup uchar sg_cont[8];
  threadgroup uint sh_first_key[256];
  threadgroup uint sh_last_key[256];
  threadgroup uchar sh_single[256];

  uint tile_begin = block.x * kAucTileRows;
  uint tile_valid = tile_begin < p.rows ? min(uint(kAucTileRows), p.rows - tile_begin) : 0u;
  uint tiles = (p.rows + kAucTileRows - 1) / kAucTileRows;
  ulong group_row_base = ulong(block.y) * p.rows;
  ulong group_tile_base = ulong(block.y) * tiles;

  uint base = tid * 8;
  uint n = base < tile_valid ? min(8u, tile_valid - base) : 0u;
  uint keys[8];
  ulong2 vals[8];
  ulong2 thr_total(0), thr_prefix(0);
  bool single = true;
  if (n > 0) {
    device const uint2* src = sorted_keys + group_row_base + tile_begin + base;
    bool in_prefix = true;
    for (uint j = 0; j < n; ++j) {
      uint2 item = src[j];
      keys[j] = item.x;
      vals[j] = auc_load_tp_fp(labels, weights, item.y, block.y, p);
      thr_total = auc_pair_add(thr_total, vals[j]);
      if (in_prefix && keys[j] == keys[0]) thr_prefix = auc_pair_add(thr_prefix, vals[j]);
      else { in_prefix = false; single = false; }
    }
  }

  // Parallel exclusive scan for prev_sum across the 256 threads in this tile.
  ulong2 sg_tot(0);
  ulong2 incl = auc_simd_incl_scan(thr_total, lane, sg_tot);
  ulong2 excl_in_sg = lane ? as_type<ulong2>(simd_shuffle_up(as_type<uint4>(incl), 1)) : ulong2(0);
  if (lane == 0) sg_sums[sg] = sg_tot;
  sh_prefix[tid] = thr_prefix;
  sh_first_key[tid] = n > 0 ? keys[0] : 0u;
  sh_last_key[tid] = n > 0 ? keys[n - 1] : 0u;
  sh_single[tid] = uchar(single);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  AucTilePrefix tp = prefixes[group_tile_base + block.x];
  ulong2 prev_sum(tp.excl_tp, tp.excl_fp);
  for (uint w = 0; w < sg; ++w) prev_sum = auc_pair_add(prev_sum, sg_sums[w]);
  prev_sum = auc_pair_add(prev_sum, excl_in_sg);

  // Parallel backward segmented tie-carry scan across active_threads.
  uint active_threads = (tile_valid + 7) / 8;
  ulong2 seed(0);
  bool cont = false;
  if (tid < active_threads) {
    if (tid + 1 == active_threads) {
      seed = ulong2(tp.carry_tp, tp.carry_fp);
      cont = false;
    } else if (sh_last_key[tid] == sh_first_key[tid + 1]) {
      seed = sh_prefix[tid + 1];
      cont = (sh_single[tid + 1] != 0);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auc_simd_suffix_carry(seed, cont, lane);
  if (lane == 0) {
    sg_sums[sg] = seed;
    sg_cont[sg] = uchar(cont);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  ulong2 sg_tail(0);
  for (int w = 7; w > int(sg); --w) {
    sg_tail = sg_cont[w] ? auc_pair_add(sg_sums[w], sg_tail) : sg_sums[w];
  }
  ulong2 my_carry = cont ? auc_pair_add(seed, sg_tail) : seed;

  ulong local_area = 0ul;
  if (n > 0) {
    ulong2 total_class = class_totals[block.y];
    ulong2 steps[8];
    ulong2 run_acc = my_carry;
    for (int j = int(n) - 1; j >= 0; --j) {
      if (uint(j + 1) < n && keys[j] != keys[j + 1]) run_acc = vals[j];
      else run_acc = auc_pair_add(vals[j], run_acc);
      steps[j] = run_acc;
    }

    bool has_prev = (tid > 0) || (block.x > 0);
    uint prev_key = tid > 0 ? sh_last_key[tid - 1]
                            : (block.x > 0 ? summaries[group_tile_base + block.x - 1].last_key : 0u);
    for (uint j = 0; j < n; ++j) {
      bool is_head = !has_prev || (keys[j] != prev_key);
      if (is_head) {
        ulong2 step = steps[j];
        ulong delta = 0ul;
        if (p.type == 13) {
          ulong two_prev_tp = cox_scale_bits(prev_sum.x, 1);
          ulong height = cox_add_bits(two_prev_tp, step.x);
          delta = cox_scale_bits(cox_multiply_bits(step.y, height), -1);
        } else {
          delta = auc_pr_segment_d64(prev_sum, step, total_class.x);
        }
        local_area = cox_add_bits(local_area, delta);
      }
      prev_sum = auc_pair_add(prev_sum, vals[j]);
      prev_key = keys[j];
      has_prev = true;
    }
  }

  #pragma clang loop unroll(full)
  for (ushort d = 16; d > 0; d >>= 1) {
    local_area = cox_add_bits(local_area, as_type<ulong>(simd_shuffle_down(as_type<uint2>(local_area), d)));
  }
  if (lane == 0) sg_areas[sg] = local_area;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong tot_area = 0ul;
    for (uint w = 0; w < 8; ++w) tot_area = cox_add_bits(tot_area, sg_areas[w]);
    tile_areas[group_tile_base + block.x] = tot_area;
  }
}

kernel void auc_final_reduce(device const ulong* tile_areas [[buffer(0)]],
                             device const ulong2* class_totals [[buffer(1)]],
                             device ulong4* output [[buffer(2)]],
                             constant MetricParams& p [[buffer(3)]],
                             uint2 block [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup ulong sh_area[256];
  uint tiles = (p.rows + kAucTileRows - 1) / kAucTileRows;
  ulong base = ulong(block.y) * tiles;
  ulong sum = 0ul;
  for (uint m = tid; m < tiles; m += 256) sum = cox_add_bits(sum, tile_areas[base + m]);
  sh_area[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint d = 128; d > 0; d >>= 1) {
    if (tid < d) sh_area[tid] = cox_add_bits(sh_area[tid], sh_area[tid + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    ulong2 tot = class_totals[block.y];
    output[block.y] = ulong4(sh_area[0], tot.x, tot.y, 0ul);
  }
}

// Bit-exact GPU row slicing for dense float32 matrices (uint32 word copy
// preserves NaN payloads and subnormals) with bounds checking.
kernel void slice_rows_dense(device const uint* source [[buffer(0)]],
                             device const int* indices [[buffer(1)]],
                             device uint* destination [[buffer(2)]],
                             device atomic_uint* invalid [[buffer(3)]],
                             constant SliceParams& p [[buffer(4)]],
                             uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.slice_rows) return;
  int src_row = indices[id.y];
  if (src_row < 0 || uint(src_row) >= p.parent_rows) {
    if (id.x == 0) atomic_store_explicit(invalid, 1u, memory_order_relaxed);
    return;
  }
  if (id.x < p.cols) {
    destination[ulong(id.y) * p.cols + id.x] = source[ulong(uint(src_row)) * p.cols + id.x];
  }
}

kernel void slice_bins_u16(device const ushort* source [[buffer(0)]],
                           device const int* indices [[buffer(1)]],
                           device ushort* destination [[buffer(2)]],
                           device atomic_uint* has_missing [[buffer(3)]],
                           constant SliceParams& p [[buffer(4)]],
                           uint2 id [[thread_position_in_grid]]) {
  bool valid = id.x < p.slice_rows && id.y < p.cols;
  bool miss = false;
  if (valid) {
    uint src_row = uint(indices[id.x]);
    ushort code = source[ulong(id.y) * p.parent_rows + src_row];
    destination[ulong(id.y) * p.slice_rows + id.x] = code;
    miss = (code == 0xffffu);
  }
  bool any_miss = simd_any(miss);
  if (!(id.x & 31u) && any_miss) atomic_store_explicit(has_missing, 1u, memory_order_relaxed);
}

kernel void slice_bins_u32(device const uint* source [[buffer(0)]],
                           device const int* indices [[buffer(1)]],
                           device uint* destination [[buffer(2)]],
                           device atomic_uint* has_missing [[buffer(3)]],
                           constant SliceParams& p [[buffer(4)]],
                           uint2 id [[thread_position_in_grid]]) {
  bool valid = id.x < p.slice_rows && id.y < p.cols;
  bool miss = false;
  if (valid) {
    uint src_row = uint(indices[id.x]);
    uint code = source[ulong(id.y) * p.parent_rows + src_row];
    destination[ulong(id.y) * p.slice_rows + id.x] = code;
    miss = (code == 0xffffffffu);
  }
  bool any_miss = simd_any(miss);
  if (!(id.x & 31u) && any_miss) atomic_store_explicit(has_missing, 1u, memory_order_relaxed);
}

// Fused binary classification probability expansion: computes [1 - p, p] in a
// single GPU pass from raw margins (sigmoid for logistic/logitraw, step for hinge).
kernel void expand_binary_proba(device const float* input [[buffer(0)]],
                                device float2* output [[buffer(1)]],
                                constant PredictParams& p [[buffer(2)]],
                                uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  float x = input[row];
  float p1 = (p.objective == 1 || p.objective == 2) ? sigmoid(x) : (p.objective == 10 ? float(x > 0) : x);
  output[row] = float2(1.0f - p1, p1);
}
