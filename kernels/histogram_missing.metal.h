// Missing values share one histogram bin. Accumulate them privately, then
// issue one exact 64-bit update per SIMD group instead of one per missing row.
// Unsigned addition preserves the existing atomics' modulo-2^64 arithmetic,
// including carries and cancellation between positive and negative gradients.
inline ulong2 histogram_missing_reduce(ulong2 value, uint lane) {
  for (ushort distance = 16; distance; distance >>= 1) {
    ulong2 other = as_type<ulong2>(simd_shuffle_down(as_type<uint4>(value), distance));
    if (lane + distance < 32) value += other;
  }
  return value;
}

kernel void histogram_missing(device const ushort* bins [[buffer(0)]],
                               device const uint* row_ids [[buffer(1)]],
                               device const long2* gradients [[buffer(2)]],
                               device const float2* scales [[buffer(3)]],
                               device const HistTask* tasks [[buffer(4)]],
                               device atomic_uint* hist [[buffer(5)]],
                               constant HistParams& p [[buffer(6)]],
                               uint2 block [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint local[4 * 513];
  HistTask task = tasks[block.x];
  uint length = (p.bins + 1) * 4;
  if (p.bins <= 512) {
    for (uint i = tid; i < p.bins * 4; i += 256)
      atomic_store_explicit(local + i, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  ulong offset = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * length;
  ulong2 missing(0);
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    uint bin = load_bin(bins, ulong(task.feature) * p.rows + row, p.bins);
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    if (bin == 0xffffffffu) {
      missing += as_type<ulong2>(value);
    } else if (p.bins <= 512) {
      add64(local + bin * 4, value.x); add64(local + bin * 4 + 2, value.y);
    } else {
      add64(hist + offset + bin * 4, value.x); add64(hist + offset + bin * 4 + 2, value.y);
    }
  }
  uint lane = tid & 31;
  missing = histogram_missing_reduce(missing, lane);
  if (lane == 0 && (missing.x != 0 || missing.y != 0)) {
    add64(hist + offset + p.bins * 4, as_type<long>(missing.x));
    add64(hist + offset + p.bins * 4 + 2, as_type<long>(missing.y));
  }
  if (p.bins <= 512) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bin = tid; bin < p.bins; bin += 256) {
      long2 value(load64(local + bin * 4), load64(local + bin * 4 + 2));
      if (value.x != 0 || value.y != 0) {
        add64(hist + offset + bin * 4, value.x);
        add64(hist + offset + bin * 4 + 2, value.y);
      }
    }
  }
}

kernel void histogram_grouped_missing(device const ushort* bins [[buffer(0)]],
                                       device const uint* row_ids [[buffer(1)]],
                                       device const long2* gradients [[buffer(2)]],
                                       device const float2* scales [[buffer(3)]],
                                       device const HistTask* tasks [[buffer(4)]],
                                       device atomic_uint* hist [[buffer(5)]],
                                       constant HistParams& p [[buffer(6)]],
                                       uint2 block [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  // No extra threadgroup scratch: shader validation doubles this allocation.
  threadgroup atomic_uint local[4 * 1024];
  HistTask task = tasks[block.x];
  uint features = min(p.padding0, p.cols - task.feature);
  uint bins_per_feature = p.bins + 1;
  uint length = features * p.bins;
  ulong global_begin = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * bins_per_feature * 4;
  for (uint i = tid; i < length * 4; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong2 missing[4] = {ulong2(0), ulong2(0), ulong2(0), ulong2(0)};
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    for (uint f = 0; f < features; ++f) {
      uint bin = load_bin(bins, ulong(task.feature + f) * p.rows + row, p.bins);
      if (bin == 0xffffffffu) {
        missing[f] += as_type<ulong2>(value);
      } else {
        uint offset = (f * p.bins + bin) * 4;
        add64(local + offset, value.x); add64(local + offset + 2, value.y);
      }
    }
  }
  uint lane = tid & 31;
  for (uint f = 0; f < features; ++f) {
    ulong2 value = histogram_missing_reduce(missing[f], lane);
    if (lane == 0 && (value.x != 0 || value.y != 0)) {
      ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
      add64(hist + offset, as_type<long>(value.x));
      add64(hist + offset + 2, as_type<long>(value.y));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bin = tid; bin < length; bin += 256) {
    ulong offset = global_begin + (bin / p.bins * bins_per_feature + bin % p.bins) * 4;
    long2 value(load64(local + bin * 4), load64(local + bin * 4 + 2));
    if (value.x != 0 || value.y != 0) {
      add64(hist + offset, value.x);
      add64(hist + offset + 2, value.y);
    }
  }
}
