// Bounds-specialized copies of the original histogram arithmetic. The host
// selects these only for scalar GPU-driven trees with at most 256 bins and
// one or two features per task. Smaller threadgroup allocations leave room
// for more resident groups without changing row order or integer updates.
template <bool Missing>
inline void histogram_compact_single(
    device const ushort* bins, device const uint* row_ids,
    device const long2* gradients, device const HistTask* tasks,
    device atomic_uint* hist, constant HistParams& p, uint2 block, uint tid,
    threadgroup atomic_uint* local) {
  HistTask task = tasks[block.x];
  uint length = (p.bins + 1) * 4;
  uint local_length = (p.bins + uint(!Missing)) * 4;
  for (uint i = tid; i < local_length; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong offset = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * length;
  ulong2 missing(0);
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    uint bin = load_bin(bins, ulong(task.feature) * p.rows + row, p.bins);
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    if (Missing && bin == 0xffffffffu) {
      missing += as_type<ulong2>(value);
    } else {
      if (bin == 0xffffffffu) bin = p.bins;
      add64(local + bin * 4, value.x); add64(local + bin * 4 + 2, value.y);
    }
  }
  if (Missing) {
    uint lane = tid & 31;
    missing = histogram_missing_reduce(missing, lane);
    if (lane == 0 && (missing.x != 0 || missing.y != 0)) {
      add64(hist + offset + p.bins * 4, as_type<long>(missing.x));
      add64(hist + offset + p.bins * 4 + 2, as_type<long>(missing.y));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bin = tid; bin < p.bins + uint(!Missing); bin += 256) {
    long2 value(load64(local + bin * 4), load64(local + bin * 4 + 2));
    if (!Missing || value.x != 0 || value.y != 0) {
      add64(hist + offset + bin * 4, value.x);
      add64(hist + offset + bin * 4 + 2, value.y);
    }
  }
}

template <bool Missing>
inline void histogram_compact_pair(
    device const ushort* bins, device const uint* row_ids,
    device const long2* gradients, device const HistTask* tasks,
    device atomic_uint* hist, constant HistParams& p, uint2 block, uint tid,
    threadgroup atomic_uint* local) {
  HistTask task = tasks[block.x];
  uint features = min(p.padding0, p.cols - task.feature);
  uint bins_per_feature = p.bins + 1;
  uint length = features * p.bins;
  ulong global_begin = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * bins_per_feature * 4;
  for (uint i = tid; i < length * 4; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong2 missing[2] = {ulong2(0), ulong2(0)};
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    for (uint f = 0; f < features; ++f) {
      uint bin = load_bin(bins, ulong(task.feature + f) * p.rows + row, p.bins);
      if (bin == 0xffffffffu) {
        if (Missing) missing[f] += as_type<ulong2>(value);
        else {
          ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
          add64(hist + offset, value.x); add64(hist + offset + 2, value.y);
        }
      } else {
        uint offset = (f * p.bins + bin) * 4;
        add64(local + offset, value.x); add64(local + offset + 2, value.y);
      }
    }
  }
  if (Missing) {
    uint lane = tid & 31;
    for (uint f = 0; f < features; ++f) {
      ulong2 value = histogram_missing_reduce(missing[f], lane);
      if (lane == 0 && (value.x != 0 || value.y != 0)) {
        ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
        add64(hist + offset, as_type<long>(value.x));
        add64(hist + offset + 2, as_type<long>(value.y));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bin = tid; bin < length; bin += 256) {
    ulong offset = global_begin + (bin / p.bins * bins_per_feature + bin % p.bins) * 4;
    long2 value(load64(local + bin * 4), load64(local + bin * 4 + 2));
    if (!Missing || value.x != 0 || value.y != 0) {
      add64(hist + offset, value.x); add64(hist + offset + 2, value.y);
    }
  }
}

#define COMPACT_HISTOGRAM_ARGUMENTS \
    device const ushort* bins [[buffer(0)]], device const uint* row_ids [[buffer(1)]], \
    device const long2* gradients [[buffer(2)]], device const float2* scales [[buffer(3)]], \
    device const HistTask* tasks [[buffer(4)]], device atomic_uint* hist [[buffer(5)]], \
    constant HistParams& p [[buffer(6)]], \
    uint2 block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]

kernel void histogram_compact(COMPACT_HISTOGRAM_ARGUMENTS) {
  threadgroup atomic_uint local[4 * 257];
  histogram_compact_single<false>(bins, row_ids, gradients, tasks, hist, p, block, tid, local);
}

kernel void histogram_missing_compact(COMPACT_HISTOGRAM_ARGUMENTS) {
  threadgroup atomic_uint local[4 * 256];
  histogram_compact_single<true>(bins, row_ids, gradients, tasks, hist, p, block, tid, local);
}

kernel void histogram_grouped_compact(COMPACT_HISTOGRAM_ARGUMENTS) {
  threadgroup atomic_uint local[4 * 512];
  histogram_compact_pair<false>(bins, row_ids, gradients, tasks, hist, p, block, tid, local);
}

kernel void histogram_grouped_missing_compact(COMPACT_HISTOGRAM_ARGUMENTS) {
  threadgroup atomic_uint local[4 * 512];
  histogram_compact_pair<true>(bins, row_ids, gradients, tasks, hist, p, block, tid, local);
}
#undef COMPACT_HISTOGRAM_ARGUMENTS

// Subtraction fully materializes the derived sibling before scoring. Only its
// built sibling needs zero initialization, including a zero-row built node
// that emits no histogram tasks. Reuse the numeric evaluator's feature/node
// dispatch grid; the host keeps root clearing and unsupported paths unchanged.
kernel void fill_histogram_built(device uint4* histogram [[buffer(0)]],
                                 device const uint* counts [[buffer(1)]],
                                 constant HistogramPlanParams& plan [[buffer(2)]],
                                 constant HistParams& p [[buffer(3)]],
                                 uint2 block [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  if (block.x >= p.cols || block.y >= plan.nodes) return;
  if (plan.subtract) {
    uint first = block.y & ~1u;
    uint built = counts[first] <= counts[first + 1] ? first : first + 1;
    if (block.y != built) return;
  }
  ulong offset = (ulong(block.y) * p.cols + block.x) * (p.bins + 1);
  for (uint bin = tid; bin <= p.bins; bin += 256) histogram[offset + bin] = uint4(0);
}
