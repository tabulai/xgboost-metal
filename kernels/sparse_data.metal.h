// Native Metal 3.1 Sparse Data Pipeline: CSR/CSC/COO canonicalization,
// GPU prefix sums, CSR->CSC transpose, CSC quantile input queues, and CSR slicing.

kernel void sparse_prefix_tiles(device const uint* counts [[buffer(0)]],
                                device uint* offsets [[buffer(1)]],
                                device uint* tile_sums [[buffer(2)]],
                                constant uint& n [[buffer(3)]],
                                uint block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint warp_sums[8];
  uint base = block * 1024;
  uint vals[4];
  uint thread_sum = 0;
  for (uint i = 0; i < 4; ++i) {
    uint idx = base + tid * 4 + i;
    vals[i] = idx < n ? counts[idx] : 0u;
    thread_sum += vals[i];
  }
  uint warp_excl = simd_prefix_exclusive_sum(thread_sum);
  uint warp_tot = simd_sum(thread_sum);
  if ((tid & 31u) == 0u) warp_sums[tid >> 5] = warp_tot;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint block_excl = 0;
  uint block_tot = 0;
  for (uint w = 0; w < 8; ++w) {
    if (w < (tid >> 5)) block_excl += warp_sums[w];
    block_tot += warp_sums[w];
  }
  uint running = block_excl + warp_excl;
  for (uint i = 0; i < 4; ++i) {
    uint idx = base + tid * 4 + i;
    if (idx < n) offsets[idx] = running;
    running += vals[i];
  }
  if (tid == 0) {
    tile_sums[block] = block_tot;
    if (n <= 1024 && block == 0) offsets[n] = block_tot;
  }
}

kernel void sparse_prefix_add(device uint* offsets [[buffer(0)]],
                              device const uint* tile_offsets [[buffer(1)]],
                              device const uint* tile_totals [[buffer(2)]],
                              constant uint& n [[buffer(3)]],
                              uint block [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  uint add = tile_offsets[block];
  uint base = block * 1024;
  for (uint i = 0; i < 4; ++i) {
    uint idx = base + tid * 4 + i;
    if (idx < n) offsets[idx] += add;
  }
  uint tiles = (n + 1023u) / 1024u;
  if (block + 1 == tiles && tid == 0) {
    offsets[n] = add + tile_totals[block];
  }
}

kernel void sparse_count_csr(device const uint* in_row_ptr [[buffer(0)]],
                             device const uint* in_col_idx [[buffer(1)]],
                             device const float* in_val [[buffer(2)]],
                             device uint* row_counts [[buffer(3)]],
                             device uchar* row_dirty [[buffer(4)]],
                             device atomic_uint* flags [[buffer(5)]],
                             constant SparseParams& p [[buffer(6)]],
                             uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint begin = in_row_ptr[row], end = in_row_ptr[row + 1];
  if (begin > end || end > p.nnz) {
    atomic_fetch_or_explicit(flags, 1u, memory_order_relaxed);
    row_counts[row] = 0;
    row_dirty[row] = 0;
    return;
  }
  uint valid = 0;
  bool dirty = false;
  bool have_prev = false;
  uint prev_col = 0;
  for (uint i = begin; i < end; ++i) {
    uint col = in_col_idx[i];
    if (col >= p.cols) {
      atomic_fetch_or_explicit(flags, 2u, memory_order_relaxed);
      continue;
    }
    float val = in_val[i];
    if (!isnan(val) && !fp32_equal(val, p.missing)) {
      if (isinf(val)) atomic_fetch_or_explicit(flags, 4u, memory_order_relaxed);
      if (have_prev && col <= prev_col) dirty = true;
      prev_col = col;
      have_prev = true;
      ++valid;
    }
  }
  row_counts[row] = valid;
  row_dirty[row] = uchar(dirty);
  if (dirty) atomic_fetch_or_explicit(flags + 1, 1u, memory_order_relaxed);
}

kernel void sparse_compact_csr(device const uint* in_row_ptr [[buffer(0)]],
                               device const uint* in_col_idx [[buffer(1)]],
                               device const float* in_val [[buffer(2)]],
                               device const uint* out_row_ptr [[buffer(3)]],
                               device uint* out_col_idx [[buffer(4)]],
                               device float* out_val [[buffer(5)]],
                               constant SparseParams& p [[buffer(6)]],
                               uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint begin = in_row_ptr[row], end = in_row_ptr[row + 1];
  uint dst = out_row_ptr[row];
  for (uint i = begin; i < end; ++i) {
    float val = in_val[i];
    if (!isnan(val) && !fp32_equal(val, p.missing)) {
      out_col_idx[dst] = in_col_idx[i];
      out_val[dst] = val;
      ++dst;
    }
  }
}

kernel void sparse_sort_dedup_rows(device const uint* row_ptr [[buffer(0)]],
                                   device const uchar* row_dirty [[buffer(1)]],
                                   device uint* col_idx [[buffer(2)]],
                                   device float* val [[buffer(3)]],
                                   device uint* orig_pos [[buffer(4)]],
                                   device uint* new_counts [[buffer(5)]],
                                   device atomic_uint* has_dups [[buffer(6)]],
                                   constant SparseParams& p [[buffer(7)]],
                                   uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint begin = row_ptr[row], end = row_ptr[row + 1];
  uint len = end - begin;
  if (!row_dirty[row] || len <= 1) {
    new_counts[row] = len;
    return;
  }
  for (uint i = 0; i < len; ++i) orig_pos[begin + i] = i;
  // Shell sort by (col_idx, orig_pos) so duplicate columns preserve exact input order.
  for (uint gap = len >> 1; gap > 0; gap >>= 1) {
    for (uint i = gap; i < len; ++i) {
      uint kc = col_idx[begin + i];
      uint kp = orig_pos[begin + i];
      float kv = val[begin + i];
      uint j = i;
      while (j >= gap) {
        uint pc = col_idx[begin + j - gap];
        uint pp = orig_pos[begin + j - gap];
        if (pc < kc || (pc == kc && pp < kp)) break;
        col_idx[begin + j] = pc;
        orig_pos[begin + j] = pp;
        val[begin + j] = val[begin + j - gap];
        j -= gap;
      }
      col_idx[begin + j] = kc;
      orig_pos[begin + j] = kp;
      val[begin + j] = kv;
    }
  }
  // Deduplicate keeping the LAST occurrence per column (matching dense last-write-wins).
  uint write = 0;
  for (uint i = 0; i < len; ++i) {
    if (i + 1 == len || col_idx[begin + i] != col_idx[begin + i + 1]) {
      col_idx[begin + write] = col_idx[begin + i];
      val[begin + write] = val[begin + i];
      ++write;
    }
  }
  new_counts[row] = write;
  if (write < len) atomic_store_explicit(has_dups, 1u, memory_order_relaxed);
}

kernel void sparse_recompact_csr(device const uint* old_row_ptr [[buffer(0)]],
                                 device const uint* old_col_idx [[buffer(1)]],
                                 device const float* old_val [[buffer(2)]],
                                 device const uint* new_row_ptr [[buffer(3)]],
                                 device uint* new_col_idx [[buffer(4)]],
                                 device float* new_val [[buffer(5)]],
                                 constant SparseParams& p [[buffer(6)]],
                                 uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint src = old_row_ptr[row];
  uint dst = new_row_ptr[row];
  uint len = new_row_ptr[row + 1] - dst;
  for (uint i = 0; i < len; ++i) {
    new_col_idx[dst + i] = old_col_idx[src + i];
    new_val[dst + i] = old_val[src + i];
  }
}

// Direct GPU COO validation, row counting, scatter, and duplicate summation into CSR.
kernel void sparse_coo_count_rows(device const uint* coo_row [[buffer(0)]],
                                  device const uint* coo_col [[buffer(1)]],
                                  device const float* coo_val [[buffer(2)]],
                                  device atomic_uint* row_counts [[buffer(3)]],
                                  device atomic_uint* flags [[buffer(4)]],
                                  constant SparseParams& p [[buffer(5)]],
                                  uint idx [[thread_position_in_grid]]) {
  if (idx >= p.nnz) return;
  uint r = coo_row[idx], c = coo_col[idx];
  if (r >= p.rows || c >= p.cols) {
    atomic_fetch_or_explicit(flags, 2u, memory_order_relaxed);
    return;
  }
  (void)coo_val;
  atomic_fetch_add_explicit(row_counts + r, 1u, memory_order_relaxed);
}

kernel void sparse_coo_scatter_csr(device const uint* coo_row [[buffer(0)]],
                                   device const uint* coo_col [[buffer(1)]],
                                   device const float* coo_val [[buffer(2)]],
                                   device atomic_uint* cursors [[buffer(3)]],
                                   device uint* csr_col [[buffer(4)]],
                                   device float* csr_val [[buffer(5)]],
                                   device uint* csr_orig [[buffer(6)]],
                                   constant SparseParams& p [[buffer(7)]],
                                   uint idx [[thread_position_in_grid]]) {
  if (idx >= p.nnz) return;
  uint r = coo_row[idx], c = coo_col[idx];
  uint pos = atomic_fetch_add_explicit(cursors + r, 1u, memory_order_relaxed);
  csr_col[pos] = c;
  csr_val[pos] = coo_val[idx];
  csr_orig[pos] = idx;
}

kernel void sparse_coo_sort_dedup_rows(device const uint* row_ptr [[buffer(0)]],
                                       device uint* col_idx [[buffer(1)]],
                                       device float* val [[buffer(2)]],
                                       device uint* orig_pos [[buffer(3)]],
                                       device uint* new_counts [[buffer(4)]],
                                       device atomic_uint* has_dups [[buffer(5)]],
                                       constant SparseParams& p [[buffer(6)]],
                                       uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint begin = row_ptr[row], end = row_ptr[row + 1];
  uint len = end - begin;
  if (len == 0) {
    new_counts[row] = 0;
    return;
  }
  for (uint gap = len >> 1; gap > 0; gap >>= 1) {
    for (uint i = gap; i < len; ++i) {
      uint kc = col_idx[begin + i];
      uint kp = orig_pos[begin + i];
      float kv = val[begin + i];
      uint j = i;
      while (j >= gap) {
        uint pc = col_idx[begin + j - gap];
        uint pp = orig_pos[begin + j - gap];
        if (pc < kc || (pc == kc && pp < kp)) break;
        col_idx[begin + j] = pc;
        orig_pos[begin + j] = pp;
        val[begin + j] = val[begin + j - gap];
        j -= gap;
      }
      col_idx[begin + j] = kc;
      orig_pos[begin + j] = kp;
      val[begin + j] = kv;
    }
  }
  uint write = 0;
  float acc = 0.0f;
  bool has_acc = false;
  for (uint i = 0; i < len; ++i) {
    acc = has_acc ? (acc + val[begin + i]) : val[begin + i];
    has_acc = true;
    if (i + 1 == len || col_idx[begin + i] != col_idx[begin + i + 1]) {
      if (!isnan(acc) && !fp32_equal(acc, p.missing)) {
        if (isinf(acc)) atomic_fetch_or_explicit(has_dups, 4u, memory_order_relaxed);
        col_idx[begin + write] = col_idx[begin + i];
        val[begin + write] = acc;
        ++write;
      }
      has_acc = false;
    }
  }
  new_counts[row] = write;
  if (write < len) atomic_fetch_or_explicit(has_dups, 1u, memory_order_relaxed);
}

// GPU CSR -> CSC Transpose (with deterministic ascending row order per column).
kernel void csr_to_csc_counts(device const uint* csr_col_idx [[buffer(0)]],
                              device atomic_uint* col_counts [[buffer(1)]],
                              constant SparseParams& p [[buffer(2)]],
                              uint e [[thread_position_in_grid]]) {
  if (e >= p.nnz) return;
  atomic_fetch_add_explicit(col_counts + csr_col_idx[e], 1u, memory_order_relaxed);
}

kernel void csr_to_csc_init_cursors(device const uint* csc_col_ptr [[buffer(0)]],
                                    device uint* cursors [[buffer(1)]],
                                    device atomic_uint* max_col_len [[buffer(2)]],
                                    constant uint& cols [[buffer(3)]],
                                    uint col [[thread_position_in_grid]]) {
  if (col >= cols) return;
  uint begin = csc_col_ptr[col], end = csc_col_ptr[col + 1];
  cursors[col] = begin;
  uint len = end - begin;
  uint m = simd_max(len);
  if (simd_is_first() && m > 0) atomic_fetch_max_explicit(max_col_len, m, memory_order_relaxed);
}

kernel void csr_to_csc_scatter(device const uint* csr_row_ptr [[buffer(0)]],
                               device const uint* csr_col_idx [[buffer(1)]],
                               device const float* csr_val [[buffer(2)]],
                               device atomic_uint* cursors [[buffer(3)]],
                               device uint* csc_row_idx [[buffer(4)]],
                               device float* csc_val [[buffer(5)]],
                               constant SparseParams& p [[buffer(6)]],
                               uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint begin = csr_row_ptr[row], end = csr_row_ptr[row + 1];
  for (uint e = begin; e < end; ++e) {
    uint col = csr_col_idx[e];
    uint pos = atomic_fetch_add_explicit(cursors + col, 1u, memory_order_relaxed);
    csc_row_idx[pos] = row;
    csc_val[pos] = csr_val[e];
  }
}

kernel void csc_sort_small_columns(device const uint* csc_col_ptr [[buffer(0)]],
                                   device uint* csc_row_idx [[buffer(1)]],
                                   device float* csc_val [[buffer(2)]],
                                   constant uint& cols [[buffer(3)]],
                                   uint col [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint s_row[1024];
  threadgroup float s_val[1024];
  if (col >= cols) return;
  uint begin = csc_col_ptr[col], end = csc_col_ptr[col + 1];
  uint len = end - begin;
  if (len <= 1 || len > 1024) return;
  uint n2 = 1u << (32u - clz(len - 1u));
  for (uint i = tid; i < n2; i += 256) {
    s_row[i] = i < len ? csc_row_idx[begin + i] : 0xffffffffu;
    s_val[i] = i < len ? csc_val[begin + i] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= n2; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      for (uint i = tid; i < n2; i += 256) {
        uint ixj = i ^ j;
        if (ixj > i) {
          bool asc = (i & k) == 0;
          bool swap_pair = asc ? (s_row[i] > s_row[ixj]) : (s_row[i] < s_row[ixj]);
          if (swap_pair) {
            uint tr = s_row[i]; s_row[i] = s_row[ixj]; s_row[ixj] = tr;
            float tv = s_val[i]; s_val[i] = s_val[ixj]; s_val[ixj] = tv;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = tid; i < len; i += 256) {
    csc_row_idx[begin + i] = s_row[i];
    csc_val[begin + i] = s_val[i];
  }
}

struct CscMergeParams { uint begin, len, width, padding; };

kernel void csc_sort_blocks_256(device uint* row_idx [[buffer(0)]],
                                device float* val [[buffer(1)]],
                                constant CscMergeParams& p [[buffer(2)]],
                                uint block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint s_row[256];
  threadgroup float s_val[256];
  uint idx = block * 256 + tid;
  s_row[tid] = idx < p.len ? row_idx[p.begin + idx] : 0xffffffffu;
  s_val[tid] = idx < p.len ? val[p.begin + idx] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) {
    for (uint j = k >> 1; j > 0; j >>= 1) {
      uint ixj = tid ^ j;
      if (ixj > tid) {
        bool asc = (tid & k) == 0;
        bool swap_pair = asc ? (s_row[tid] > s_row[ixj]) : (s_row[tid] < s_row[ixj]);
        if (swap_pair) {
          uint tr = s_row[tid]; s_row[tid] = s_row[ixj]; s_row[ixj] = tr;
          float tv = s_val[tid]; s_val[tid] = s_val[ixj]; s_val[ixj] = tv;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  if (idx < p.len) {
    row_idx[p.begin + idx] = s_row[tid];
    val[p.begin + idx] = s_val[tid];
  }
}

kernel void csc_merge_pass(device const uint* in_row [[buffer(0)]],
                           device const float* in_val [[buffer(1)]],
                           device uint* out_row [[buffer(2)]],
                           device float* out_val [[buffer(3)]],
                           constant CscMergeParams& p [[buffer(4)]],
                           uint pos [[thread_position_in_grid]]) {
  if (pos >= p.len) return;
  uint pair_base = (pos / (2 * p.width)) * (2 * p.width);
  uint mid = min(pair_base + p.width, p.len);
  uint end = min(pair_base + 2 * p.width, p.len);
  if (mid >= end) {
    out_row[pos] = in_row[p.begin + pos];
    out_val[pos] = in_val[p.begin + pos];
    return;
  }
  uint local = pos - pair_base;
  uint len_a = mid - pair_base, len_b = end - mid;
  uint lo = local > len_b ? local - len_b : 0u;
  uint hi = min(local, len_a);
  while (lo < hi) {
    uint i = lo + ((hi - lo) >> 1);
    uint j = local - i;
    if (j > 0 && i < len_a && in_row[p.begin + mid + j - 1] > in_row[p.begin + pair_base + i]) {
      lo = i + 1;
    } else {
      hi = i;
    }
  }
  uint i = lo, j = local - lo;
  bool take_a = j >= len_b || (i < len_a && in_row[p.begin + pair_base + i] < in_row[p.begin + mid + j]);
  uint src = p.begin + (take_a ? pair_base + i : mid + j);
  out_row[pos] = in_row[src];
  out_val[pos] = in_val[src];
}

kernel void csc_copy_back(device const uint* src_row [[buffer(0)]],
                          device const float* src_val [[buffer(1)]],
                          device uint* dst_row [[buffer(2)]],
                          device float* dst_val [[buffer(3)]],
                          constant CscMergeParams& p [[buffer(4)]],
                          uint pos [[thread_position_in_grid]]) {
  if (pos >= p.len) return;
  dst_row[p.begin + pos] = src_row[pos];
  dst_val[p.begin + pos] = src_val[pos];
}

// Categorical validation & bin counts on CSR nonzeros.
kernel void category_statistics_csr(device const uint* csr_col_idx [[buffer(0)]],
                                    device const float* csr_val [[buffer(1)]],
                                    device const uchar* categorical [[buffer(2)]],
                                    device atomic_uint* counts [[buffer(3)]],
                                    device atomic_uint* invalid [[buffer(4)]],
                                    constant SparseParams& p [[buffer(5)]],
                                    uint e [[thread_position_in_grid]]) {
  if (e >= p.nnz) return;
  uint col = csr_col_idx[e];
  if (!categorical[col]) return;
  float value = csr_val[e];
  bool bad = !isnan(value) && (fp32_less(value, 0) || value >= 16777216.0f);
  if (bad) {
    atomic_store_explicit(invalid, 1u, memory_order_relaxed);
  } else if (!isnan(value)) {
    atomic_fetch_max_explicit(counts + col, uint(value) + 1u, memory_order_relaxed);
  }
}

// Quantize CSR values directly into compact csr_bins (nnz entries, no rows * cols dense allocation).
kernel void sparse_quantize_csr(device const uint* csr_col_idx [[buffer(0)]],
                                device const float* csr_val [[buffer(1)]],
                                device const float* cuts [[buffer(2)]],
                                device const uint* counts [[buffer(3)]],
                                device ushort* csr_bins [[buffer(4)]],
                                device const uchar* categorical [[buffer(5)]],
                                constant SparseParams& p [[buffer(6)]],
                                uint e [[thread_position_in_grid]]) {
  if (e >= p.nnz) return;
  uint col = csr_col_idx[e];
  float value = csr_val[e];
  if (categorical[col]) {
    uint bin = 0xffffffffu;
    if (!isnan(value) && counts[col]) {
      float code = float(int(value));
      auto column = cuts + ulong(col) * p.bins;
      uint candidate = uint(max(code, 0.0f));
      if (candidate < counts[col] && fp32_equal(column[candidate], code)) bin = candidate;
      else {
        uint lo = 0, hi = counts[col];
        while (lo < hi) {
          uint mid = lo + (hi - lo) / 2;
          if (fp32_less(column[mid], code)) lo = mid + 1; else hi = mid;
        }
        bin = min(lo, counts[col] - 1);
      }
    }
    store_bin(csr_bins, e, bin, p.bins);
    return;
  }
  uint lo = 0, hi = counts[col];
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (fp32_greater_equal(value, cuts[ulong(col) * p.bins + mid])) lo = mid + 1; else hi = mid;
  }
  store_bin(csr_bins, e, isnan(value) || counts[col] == 0 ? 0xffffffffu : min(lo, counts[col] - 1), p.bins);
}

// Decode quantized CSR bins back into CSR values (for QuantileDMatrix on sparse storage).
kernel void sparse_decode_quantiles_csr(device const uint* csr_col_idx [[buffer(0)]],
                                        device const ushort* csr_bins [[buffer(1)]],
                                        device const float* cuts [[buffer(2)]],
                                        device const uchar* categorical [[buffer(3)]],
                                        device float* out_val [[buffer(4)]],
                                        constant QuantileValueParams& p [[buffer(5)]],
                                        uint e [[thread_position_in_grid]]) {
  // Note: p.rows carries nnz in this call.
  if (e >= p.rows) return;
  uint column = csr_col_idx[e];
  uint bin = load_bin(csr_bins, e, p.bins);
  float value = NAN;
  if (bin != 0xffffffffu) {
    auto feature_cuts = cuts + ulong(column) * p.bins;
    if (categorical[column] && !p.numeric_only) value = feature_cuts[bin];
    else if (bin) value = feature_cuts[bin - 1];
    else value = p.representative ? fp32_next_down(feature_cuts[0]) : -INFINITY;
  }
  out_val[e] = value;
}

// Lazy fallback materialization (ONLY for legacy callers like refresh updater).
kernel void sparse_fill_nan(device float* dense [[buffer(0)]],
                            constant ulong& total [[buffer(1)]],
                            uint idx [[thread_position_in_grid]]) {
  if (ulong(idx) < total) dense[idx] = NAN;
}

kernel void sparse_scatter_dense(device const uint* csr_row_ptr [[buffer(0)]],
                                 device const uint* csr_col_idx [[buffer(1)]],
                                 device const float* csr_val [[buffer(2)]],
                                 device float* dense [[buffer(3)]],
                                 constant SparseParams& p [[buffer(4)]],
                                 uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  ulong base = ulong(row) * p.cols;
  for (uint e = csr_row_ptr[row]; e < csr_row_ptr[row + 1]; ++e) {
    dense[base + csr_col_idx[e]] = csr_val[e];
  }
}

kernel void sparse_fill_missing_bins(device ushort* bins [[buffer(0)]],
                                     constant SparseParams& p [[buffer(1)]],
                                     uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.cols) return;
  store_bin(bins, ulong(id.y) * p.rows + id.x, 0xffffffffu, p.bins);
}

kernel void sparse_scatter_dense_bins(device const uint* csr_row_ptr [[buffer(0)]],
                                      device const uint* csr_col_idx [[buffer(1)]],
                                      device const ushort* csr_bins [[buffer(2)]],
                                      device ushort* dense_bins [[buffer(3)]],
                                      constant SparseParams& p [[buffer(4)]],
                                      uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  for (uint e = csr_row_ptr[row]; e < csr_row_ptr[row + 1]; ++e) {
    uint col = csr_col_idx[e];
    uint bin = load_bin(csr_bins, e, p.bins);
    store_bin(dense_bins, ulong(col) * p.rows + row, bin, p.bins);
  }
}

// GPU CSR row slicing (preserves CSR values and optional quantized csr_bins).
kernel void sparse_slice_counts(device const uint* parent_row_ptr [[buffer(0)]],
                                device const int* slice_indices [[buffer(1)]],
                                device uint* out_counts [[buffer(2)]],
                                constant uint& slice_rows [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
  if (i >= slice_rows) return;
  uint src = uint(slice_indices[i]);
  out_counts[i] = parent_row_ptr[src + 1] - parent_row_ptr[src];
}

kernel void sparse_slice_scatter(device const uint* parent_row_ptr [[buffer(0)]],
                                 device const uint* parent_col_idx [[buffer(1)]],
                                 device const float* parent_val [[buffer(2)]],
                                 device const ushort* parent_bins [[buffer(3)]],
                                 device const int* slice_indices [[buffer(4)]],
                                 device const uint* out_row_ptr [[buffer(5)]],
                                 device uint* out_col_idx [[buffer(6)]],
                                 device float* out_val [[buffer(7)]],
                                 device ushort* out_bins [[buffer(8)]],
                                 constant SparseParams& p [[buffer(9)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i >= p.rows) return;
  uint src = uint(slice_indices[i]);
  uint src_begin = parent_row_ptr[src];
  uint dst_begin = out_row_ptr[i];
  uint len = out_row_ptr[i + 1] - dst_begin;
  bool copy_bins = (p.flags & 1u) != 0u;
  for (uint k = 0; k < len; ++k) {
    out_col_idx[dst_begin + k] = parent_col_idx[src_begin + k];
    out_val[dst_begin + k] = parent_val[src_begin + k];
    if (copy_bins) {
      uint bin = load_bin(parent_bins, src_begin + k, p.bins);
      store_bin(out_bins, dst_begin + k, bin, p.bins);
    }
  }
}

// CSC Quantile Sketching Input Queue Builder (O(nnz) instead of O(rows * cols)).
kernel void sketch_csc_tile_counts(device const uint* csc_col_ptr [[buffer(0)]],
                                   device const uchar* categorical [[buffer(1)]],
                                   device uint* tile_counts [[buffer(2)]],
                                   constant uint& cols [[buffer(3)]],
                                   uint col [[thread_position_in_grid]]) {
  if (col >= cols) return;
  if (categorical[col]) {
    tile_counts[col] = 0;
    return;
  }
  uint len = csc_col_ptr[col + 1] - csc_col_ptr[col];
  tile_counts[col] = (len + 127u) / 128u;
}

kernel void sketch_csc_populate_tiles(device const uint* col_tile_offsets [[buffer(0)]],
                                      device uint2* tile_map [[buffer(1)]],
                                      constant uint& cols [[buffer(2)]],
                                      uint col [[thread_position_in_grid]]) {
  if (col >= cols) return;
  uint begin = col_tile_offsets[col], end = col_tile_offsets[col + 1];
  for (uint t = begin; t < end; ++t) {
    tile_map[t] = uint2(col, t - begin);
  }
}

kernel void sketch_csc_blocks(device const uint* csc_col_ptr [[buffer(0)]],
                              device const uint* csc_row_idx [[buffer(1)]],
                              device const float* csc_val [[buffer(2)]],
                              device const float* weights [[buffer(3)]],
                              device const uint2* tile_map [[buffer(4)]],
                              device float2* temp_values [[buffer(5)]],
                              device SketchInputBlock* blocks [[buffer(6)]],
                              constant SketchCscParams& p [[buffer(7)]],
                              uint block_idx [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  uint tile = block_idx * 8 + (tid >> 5);
  uint lane = tid & 31u;
  if (tile >= p.total_tiles) return;
  uint2 info = tile_map[tile];
  uint feature = info.x, local_tile = info.y;
  uint col_begin = csc_col_ptr[feature], col_end = csc_col_ptr[feature + 1];
  uint col_len = col_end - col_begin;

  float2 local[4];
  bool valid[4];
  uint count = 0, nonmissing = 0;
  float first = 0, last = 0;
  for (uint i = 0; i < 4; ++i) {
    uint k = local_tile * 128 + lane * 4 + i;
    float value = k < col_len ? csc_val[col_begin + k] : NAN;
    float weight = k < col_len ? weights[csc_row_idx[col_begin + k]] : 0.0f;
    nonmissing += !isnan(value);
    valid[i] = !isnan(value) && (as_type<uint>(weight) & 0x7fffffffu) != 0;
    local[i] = float2(value, weight);
    if (valid[i]) {
      if (!count) first = value;
      last = value;
      ++count;
    }
  }
  uint mask = simd_sum(uint(count != 0) << lane);
  uint before = mask & ((1u << lane) - 1u);
  uint previous_lane = before ? 31u - clz(before) : 0u;
  float previous = simd_shuffle(last, previous_lane);
  bool have_previous = before != 0;
  uint offset = simd_prefix_exclusive_sum(count), runs = 0, element = 0;
  ulong output = ulong(tile) * 128 + offset;
  for (uint i = 0; i < 4; ++i) if (valid[i]) {
    temp_values[output + element++] = local[i];
    runs += !have_previous || !fp32_equal(local[i].x, previous);
    previous = local[i].x;
    have_previous = true;
  }
  uint total = simd_sum(count), total_nonmissing = simd_sum(nonmissing), total_runs = simd_sum(runs);
  uint first_lane = mask ? 31u - clz(mask & (~mask + 1u)) : 0u;
  uint last_lane = mask ? 31u - clz(mask) : 0u;
  float first_value = simd_shuffle(first, first_lane), last_value = simd_shuffle(last, last_lane);
  if (!lane) blocks[tile] = SketchInputBlock{total, total_nonmissing, total_runs, 0, first_value, last_value, 0, 0};
}

kernel void sketch_csc_offsets(device const uint* col_tile_offsets [[buffer(0)]],
                               device SketchInputBlock* blocks [[buffer(1)]],
                               device SketchStats* stats [[buffer(2)]],
                               constant SketchCscParams& p [[buffer(3)]],
                               uint feature [[thread_position_in_grid]]) {
  if (feature >= p.cols) return;
  uint t_begin = col_tile_offsets[feature], t_end = col_tile_offsets[feature + 1];
  SketchStats result{};
  bool have_last = false;
  float last = 0;
  for (uint tile = t_begin; tile < t_end; ++tile) {
    auto block = blocks[tile];
    block.offset = result.elements;
    block.run_offset = result.runs;
    block.continued = block.elements && have_last && fp32_equal(block.first, last);
    result.elements += block.elements;
    result.nonmissing += block.nonmissing;
    result.runs += block.runs - block.continued;
    if (block.elements) { have_last = true; last = block.last; }
    blocks[tile] = block;
  }
  stats[feature] = result;
}

kernel void sketch_csc_compact(device const uint* csc_col_ptr [[buffer(0)]],
                               device const uint2* tile_map [[buffer(1)]],
                               device const float2* temp_values [[buffer(2)]],
                               device const SketchInputBlock* blocks [[buffer(3)]],
                               device float2* compact [[buffer(4)]],
                               device uint* starts [[buffer(5)]],
                               constant SketchCscParams& p [[buffer(6)]],
                               uint block_idx [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  uint tile = block_idx * 8 + (tid >> 5);
  uint lane = tid & 31u;
  if (tile >= p.total_tiles) return;
  auto block = blocks[tile];
  if (!block.elements) return;
  uint feature = tile_map[tile].x;
  ulong column = ulong(csc_col_ptr[feature]);
  auto source = temp_values + ulong(tile) * 128;
  bool new_run[4];
  float2 local[4];
  uint runs = 0;
  for (uint i = 0; i < 4; ++i) {
    uint index = lane * 4 + i;
    local[i] = index < block.elements ? source[index] : float2(0);
    new_run[i] = index < block.elements && (index ? !fp32_equal(local[i].x, source[index - 1].x) : !block.continued);
    runs += new_run[i];
  }
  uint run_offset = block.run_offset + simd_prefix_exclusive_sum(runs);
  for (uint i = 0; i < 4; ++i) {
    uint index = lane * 4 + i;
    if (index >= block.elements) continue;
    compact[column + block.offset + index] = local[i];
    if (new_run[i]) starts[column + run_offset++] = block.offset + index;
  }
}

kernel void sketch_csc_runs(device const uint* csc_col_ptr [[buffer(0)]],
                            device const uint* col_tile_offsets [[buffer(1)]],
                            device const uint2* tile_map [[buffer(2)]],
                            device const float2* compact [[buffer(3)]],
                            device const uint* starts [[buffer(4)]],
                            device const SketchStats* stats [[buffer(5)]],
                            device float2* queue [[buffer(6)]],
                            constant SketchCscParams& p [[buffer(7)]],
                            uint gid [[thread_position_in_grid]]) {
  uint tile = gid / 128;
  uint local_run = gid % 128;
  if (tile >= p.total_tiles) return;
  uint2 info = tile_map[tile];
  uint feature = info.x;
  uint run_idx = info.y * 128 + local_run;
  uint total_runs = stats[feature].runs;
  if (run_idx >= total_runs) return;
  ulong column = ulong(csc_col_ptr[feature]);
  uint begin = starts[column + run_idx];
  uint end = run_idx + 1 < total_runs ? starts[column + run_idx + 1] : stats[feature].elements;
  float2 value = compact[column + begin];
  if (p.unit_weights) value.y = float(min(end - begin, 1u << 24));
  else for (uint i = begin + 1; i < end; ++i) value.y = sketch_add(value.y, compact[column + i].y);
  queue[column + run_idx] = value;
}
