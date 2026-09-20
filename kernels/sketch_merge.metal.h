// CUDA-inspired parallel merge and prune of weighted rank summaries. A
// threadgroup owns a feature's level stack, retaining the CPU sketch's exact
// carry order while evaluating independent output entries in parallel.
inline uint sketch_prefix_count(uint value, uint tid, threadgroup uint* storage) {
  uint lane = tid & 31, warp = tid / 32;
  uint local = simd_prefix_exclusive_sum(value), total = simd_sum(value);
  if (!lane) storage[warp] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!warp) {
    uint count = lane < 8 ? storage[lane] : 0;
    uint prefix = simd_prefix_exclusive_sum(count), all = simd_sum(count);
    if (lane < 8) storage[lane] = prefix;
    if (!lane) storage[8] = all;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local + storage[warp];
}
inline float sketch_prefix_max(float value, uint tid, threadgroup float* storage) {
  uint lane = tid & 31, warp = tid / 32;
  float prefix = value;
  for (uint shift = 1; shift < 32; shift <<= 1) {
    float other = simd_shuffle_up(prefix, shift);
    if (lane >= shift) prefix = sketch_max(prefix, other);
  }
  float local = simd_shuffle_up(prefix, 1);
  if (!lane) local = 0;
  if (lane == 31) storage[warp] = prefix;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!warp) {
    float across = lane < 8 ? storage[lane] : 0;
    for (uint shift = 1; shift < 32; shift <<= 1) {
      float other = simd_shuffle_up(across, shift);
      if (lane >= shift) across = sketch_max(across, other);
    }
    float previous = simd_shuffle_up(across, 1);
    if (lane < 8) storage[lane] = lane ? previous : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return sketch_max(local, storage[warp]);
}

inline uint sketch_prune_index(device const float4* entries, uint size, uint capacity, uint k) {
  if (!k) return 0;
  if (k + 1 == capacity) return size - 1;
  float begin = entries[0].y, range = sketch_subtract(entries[size - 1].x, begin);
  float rank = sketch_multiply(2, sketch_add(sketch_divide(sketch_multiply(float(k), range), float(capacity - 1)), begin));
  uint lo = 2, hi = size;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (fp32_greater_equal(rank, sketch_add(entries[mid].y, entries[mid].x))) lo = mid + 1;
    else hi = mid;
  }
  uint left = lo - 1;
  if (left == size - 1) return left;
  return left + uint(!(fp32_less(rank, sketch_add(sketch_min_next(entries[left]), sketch_max_previous(entries[left + 1])))));
}

inline uint sketch_prune_parallel(device float4* entries, uint size, uint capacity,
                                  device float4* scratch, uint tid, threadgroup uint* scan) {
  if (size <= capacity) return size;
  if (capacity < 2) return capacity;
  if (!isfinite(entries[0].y) || !isfinite(entries[size - 1].y)) {
    if (!tid) scan[8] = sketch_prune_serial(entries, size, capacity);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return scan[8];
  }
  for (uint i = tid; i < capacity; i += 256) scratch[i] = entries[sketch_prune_index(entries, size, capacity, i)];
  threadgroup_barrier(mem_flags::mem_device);
  uint chunk = (capacity + 255) / 256, begin = min(tid * chunk, capacity), end = min(begin + chunk, capacity);
  uint count = 0;
  for (uint i = begin; i < end; ++i) count += !i || !fp32_equal(scratch[i].w, scratch[i - 1].w);
  uint output = sketch_prefix_count(count, tid, scan);
  for (uint i = begin; i < end; ++i)
    if (!i || !fp32_equal(scratch[i].w, scratch[i - 1].w)) entries[output++] = scratch[i];
  threadgroup_barrier(mem_flags::mem_device);
  return scan[8];
}

inline uint sketch_lower_bound(device const float4* entries, uint size, float value) {
  uint lo = 0, hi = size;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (fp32_less(entries[mid].w, value)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

inline uint sketch_combine_parallel(device float4* current, uint n,
                                    device const float4* other, uint m,
                                    device float4* scratch, uint tid,
                                    threadgroup uint* counts, threadgroup float* scan) {
  if (!m) return n;
  if (!n) {
    for (uint i = tid; i < m; i += 256) current[i] = other[i];
    threadgroup_barrier(mem_flags::mem_device);
    return m;
  }
  uint size = n + m, bad = 0;
  for (uint i = tid; i < size; i += 256) {
    bool from_left = i < n;
    uint index = from_left ? i : i - n;
    auto opposite = from_left ? other : current;
    uint opposite_size = from_left ? m : n;
    float4 value = from_left ? current[index] : other[index];
    uint position = sketch_lower_bound(opposite, opposite_size, value.w);
    bool equal = position < opposite_size && fp32_equal(opposite[position].w, value.w);
    float4 entry;
    if (equal) {
      float4 left = from_left ? value : opposite[position];
      float4 right = from_left ? opposite[position] : value;
      entry = float4(sketch_add(left.x, right.x), sketch_add(left.y, right.y), sketch_add(left.z, right.z), left.w);
    } else {
      float lower = position ? sketch_min_next(opposite[position - 1]) : 0;
      float upper = position == opposite_size ? opposite[opposite_size - 1].y : sketch_max_previous(opposite[position]);
      entry = float4(sketch_add(value.x, lower), sketch_add(value.y, upper), value.z, value.w);
    }
    scratch[index + position + uint(!from_left && equal)] = entry;
    bad |= !isfinite(entry.x) || !isfinite(entry.y) || !isfinite(entry.z);
  }
  threadgroup_barrier(mem_flags::mem_device);
  sketch_prefix_count(bad, tid, counts);
  if (counts[8]) {
    // Preserve ordered IEEE comparisons for overflowed rank summaries.
    if (!tid) counts[8] = sketch_combine_serial(current, n, other, m, scratch);
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    return counts[8];
  }
  uint chunk = (size + 255) / 256, begin = min(tid * chunk, size), end = min(begin + chunk, size);
  float local_min = 0;
  uint unique = 0;
  for (uint i = begin; i < end; ++i) {
    local_min = sketch_max(local_min, scratch[i].x);
    unique += !i || !fp32_equal(scratch[i].w, scratch[i - 1].w);
  }
  float previous_min = sketch_prefix_max(local_min, tid, scan), local_max = 0;
  for (uint i = begin; i < end; ++i) {
    auto value = scratch[i];
    previous_min = sketch_max(previous_min, value.x);
    value.x = previous_min;
    value.y = sketch_max(value.y, sketch_min_next(value));
    local_max = sketch_max(local_max, value.y);
    scratch[i] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);
  float previous_max = sketch_prefix_max(local_max, tid, scan);
  uint output = sketch_prefix_count(unique, tid, counts);
  for (uint i = begin; i < end; ++i) {
    auto value = scratch[i];
    previous_max = sketch_max(previous_max, value.y);
    value.y = previous_max;
    if (!i || !fp32_equal(value.w, scratch[i - 1].w)) current[output++] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);
  return counts[8];
}

kernel void sketch_reduce_parallel(device const float4* summaries [[buffer(0)]],
                                   device const uint* sizes [[buffer(1)]],
                                   device const SketchBatch* batches [[buffer(2)]],
                                   device const SketchFeature* features [[buffer(3)]],
                                   device float4* levels [[buffer(4)]],
                                   device float4* workspace [[buffer(5)]],
                                   device SketchStats* stats [[buffer(6)]],
                                   constant SketchParams& p [[buffer(7)]],
                                   uint feature [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  if (feature >= p.features) return;
  auto column = features[feature];
  auto current = workspace + column.work_offset;
  auto scratch = current + 2 * column.limit;
  auto resident = levels + column.level_offset;
  threadgroup uint counts[64], integer_scan[9];
  threadgroup float float_scan[9];
  if (tid < 64) counts[tid] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint batch = column.batch_begin; batch < column.batch_begin + column.batches; ++batch) {
    uint n = sizes[batch], level = 0;
    auto source = summaries + batches[batch].offset;
    for (uint i = tid; i < n; i += 256) current[i] = source[i];
    threadgroup_barrier(mem_flags::mem_device);
    while (true) {
      n = sketch_prune_parallel(current, n, column.limit, scratch, tid, integer_scan);
      uint m = counts[level];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      n = sketch_combine_parallel(current, n, resident + ulong(level) * column.limit, m, scratch,
                                  tid, integer_scan, float_scan);
      if (!tid) counts[level] = 0;
      if (n <= column.limit) break;
      ++level;
    }
    if (!tid) counts[level] = n;
    for (uint i = tid; i < n; i += 256) resident[ulong(level) * column.limit + i] = current[i];
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  uint n = 0;
  for (uint level = 0; level < column.levels; ++level) {
    n = sketch_combine_parallel(current, n, resident + ulong(level) * column.limit, counts[level], scratch,
                                tid, integer_scan, float_scan);
    n = sketch_prune_parallel(current, n, max(column.budget, column.limit), scratch, tid, integer_scan);
  }
  n = sketch_prune_parallel(current, n, column.budget, scratch, tid, integer_scan);
  if (!tid) stats[feature].summary = n;
}
