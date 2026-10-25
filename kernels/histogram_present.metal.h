// Nonroot scalar GPU-driver nodes inherit exact integer totals from the parent.
// Their split search reconstructs missing = inherited_total - present, so the
// stored missing histogram bin is never consumed as a statistic. Skip missing
// rows before reading gradients and leave the already-cleared missing bin alone.
// Derived siblings may carry a placeholder missing bin through subtraction;
// all present bins and their exact integer addition remain unchanged.
// This must not be used for root, categorical/vector, or non-inherited scoring.
kernel void histogram_present(device const ushort* bins [[buffer(0)]],
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
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    uint bin = load_bin(bins, ulong(task.feature) * p.rows + row, p.bins);
    if (bin == 0xffffffffu) continue;
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    if (p.bins <= 512) {
      add64(local + bin * 4, value.x); add64(local + bin * 4 + 2, value.y);
    } else {
      add64(hist + offset + bin * 4, value.x); add64(hist + offset + bin * 4 + 2, value.y);
    }
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

kernel void histogram_grouped_present(device const ushort* bins [[buffer(0)]],
                                       device const uint* row_ids [[buffer(1)]],
                                       device const long2* gradients [[buffer(2)]],
                                       device const float2* scales [[buffer(3)]],
                                       device const HistTask* tasks [[buffer(4)]],
                                       device atomic_uint* hist [[buffer(5)]],
                                       constant HistParams& p [[buffer(6)]],
                                       uint2 block [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  // Same 16 KiB scratch budget as histogram_grouped_missing. Per-row bin IDs
  // are private; even validation's extra scratch stays within the device limit.
  threadgroup atomic_uint local[4 * 1024];
  HistTask task = tasks[block.x];
  uint features = min(p.padding0, p.cols - task.feature);
  uint bins_per_feature = p.bins + 1;
  uint length = features * p.bins;
  ulong global_begin = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * bins_per_feature * 4;
  for (uint i = tid; i < length * 4; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    uint row_bins[4];
    bool any_present = false;
    for (uint f = 0; f < features; ++f) {
      row_bins[f] = load_bin(bins, ulong(task.feature + f) * p.rows + row, p.bins);
      any_present = any_present || row_bins[f] != 0xffffffffu;
    }
    if (!any_present) continue;
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    for (uint f = 0; f < features; ++f) {
      uint bin = row_bins[f];
      if (bin == 0xffffffffu) continue;
      uint offset = (f * p.bins + bin) * 4;
      add64(local + offset, value.x); add64(local + offset + 2, value.y);
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
