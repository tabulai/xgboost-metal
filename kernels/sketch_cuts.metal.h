// Independent rank queries follow CUDA's segmented pruning layout. The CPU
// cut rule additionally forces each selected value past its predecessor:
// s_i = max(q_i, s_(i-1) + 1) = i + max(0, max(q_j-j, j <= i)).
// Clamp before unsigned subtraction, and widen before the final addition.
inline uint sketch_cut_prefix_max(uint value, uint tid, threadgroup uint* storage, bool exclusive) {
  uint lane = tid & 31, warp = tid / 32;
  uint prefix = value;
  for (uint shift = 1; shift < 32; shift <<= 1) {
    uint other = simd_shuffle_up(prefix, shift);
    if (lane >= shift) prefix = max(prefix, other);
  }
  uint previous = simd_shuffle_up(prefix, 1);
  uint local = exclusive ? (lane ? previous : 0) : prefix;
  if (lane == 31) storage[warp] = prefix;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!warp) {
    uint across = lane < 8 ? storage[lane] : 0;
    for (uint shift = 1; shift < 32; shift <<= 1) {
      uint other = simd_shuffle_up(across, shift);
      if (lane >= shift) across = max(across, other);
    }
    if (lane < 8) storage[lane] = across;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return max(local, warp ? storage[warp - 1] : 0);
}

inline uint sketch_cut_query_index(device const float4* entries, uint size, uint bins, uint i) {
  ulong total = cox_float_bits(entries[size - 1].y), denominator = sketch_integer_bits(bins);
  ulong rank = cox_scale_bits(cox_divide_bits(cox_multiply_bits(sketch_integer_bits(i), total), denominator), 1);
  uint lo = 1, hi = size - 1;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (rank >= cox_float_bits(sketch_add(entries[mid].x, entries[mid].y))) lo = mid + 1;
    else hi = mid;
  }
  uint cursor = lo - 1;
  float boundary = sketch_add(sketch_min_next(entries[cursor]), sketch_max_previous(entries[cursor + 1]));
  return cursor + uint(isnan(boundary) || rank >= cox_float_bits(boundary));
}

kernel void sketch_cut_queries(device float4* workspace [[buffer(0)]],
                             device const SketchFeature* features [[buffer(1)]],
                             device const SketchStats* stats [[buffer(2)]],
                             device const uchar* categorical [[buffer(3)]],
                             device uint* tiles [[buffer(4)]],
                             constant SketchCutParams& p [[buffer(5)]],
                             uint2 block [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  uint feature = block.y, size = stats[feature].summary;
  auto plan = features[feature];
  auto entries = workspace + plan.work_offset;
  if (categorical[feature] || size <= p.matrix.numeric_bins || !isfinite(entries[size - 1].y)) return;
  uint i = block.x * 256 + tid + 1, value = 0;
  if (i < p.capacity) {
    uint queried = sketch_cut_query_index(entries, size, p.matrix.numeric_bins, i);
    value = queried > i ? queried - i : 0;
  }
  threadgroup uint scan[8];
  uint prefix = sketch_cut_prefix_max(value, tid, scan, false);
  if (i < p.capacity) {
    auto scratch = reinterpret_cast<device uint*>(entries + 2ul * plan.limit);
    scratch[i - 1] = prefix;
  }
  if (tid == 255) tiles[ulong(feature) * p.tiles + block.x] = prefix;
}

kernel void sketch_cut_carries(device float4* workspace [[buffer(0)]],
                             device const SketchFeature* features [[buffer(1)]],
                             device const SketchStats* stats [[buffer(2)]],
                             device float* cuts [[buffer(3)]],
                             device uint* counts [[buffer(4)]],
                             device float* minima [[buffer(5)]],
                             device const uchar* categorical [[buffer(6)]],
                             device uint* tiles [[buffer(7)]],
                             constant SketchCutParams& p [[buffer(8)]],
                             uint feature [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  uint size = stats[feature].summary;
  auto plan = features[feature];
  auto entries = workspace + plan.work_offset;
  auto output = cuts + ulong(feature) * p.matrix.bins;
  if (categorical[feature] || !size ||
      (size > p.matrix.numeric_bins && !isfinite(entries[size - 1].y))) {
    if (!tid) sketch_cuts_serial(workspace, features, stats, cuts, counts, minima, categorical, p.matrix, feature);
    return;
  }
  if (!tid) minima[feature] = -INFINITY;
  if (size <= p.matrix.numeric_bins) {
    if (!tid) {
      float tail = entries[size - 1].w;
      output[size - 1] = tail + (fabs(tail) + 1e-5f);
      counts[feature] = size;
    }
    return;
  }
  auto blocks = tiles + ulong(feature) * p.tiles;
  uint chunk = (p.tiles + 255) / 256, begin = min(tid * chunk, p.tiles), end = min(begin + chunk, p.tiles);
  uint largest = 0;
  for (uint i = begin; i < end; ++i) largest = max(largest, blocks[i]);
  threadgroup uint scan[8];
  uint carry = sketch_cut_prefix_max(largest, tid, scan, true);
  for (uint i = begin; i < end; ++i) {
    uint value = blocks[i];
    blocks[i] = carry;
    carry = max(carry, value);
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (!tid) {
    auto scratch = reinterpret_cast<device uint*>(entries + 2ul * plan.limit);
    // Selected indices strictly increase. Find the first exhausted query;
    // the final sentinel occupies that query's output slot.
    uint lo = 1, hi = p.matrix.numeric_bins;
    while (lo < hi) {
      uint mid = lo + (hi - lo) / 2;
      ulong selected = ulong(mid) + max(scratch[mid - 1], blocks[(mid - 1) / 256]);
      if (selected < size) lo = mid + 1;
      else hi = mid;
    }
    float tail = entries[size - 1].w;
    output[lo - 1] = tail + (fabs(tail) + 1e-5f);
    counts[feature] = lo;
  }
}

kernel void sketch_cut_gather(device const float4* workspace [[buffer(0)]],
                            device const SketchFeature* features [[buffer(1)]],
                            device const SketchStats* stats [[buffer(2)]],
                            device float* cuts [[buffer(3)]],
                            device const uint* counts [[buffer(4)]],
                            device const uchar* categorical [[buffer(5)]],
                            device const uint* tiles [[buffer(6)]],
                            constant SketchCutParams& p [[buffer(7)]],
                            uint2 id [[thread_position_in_grid]]) {
  uint feature = id.y, i = id.x + 1, size = stats[feature].summary;
  auto plan = features[feature];
  auto entries = workspace + plan.work_offset;
  if (categorical[feature] || i >= counts[feature] ||
      (size > p.matrix.numeric_bins && !isfinite(entries[size - 1].y))) return;
  ulong selected = i;
  if (size > p.matrix.numeric_bins) {
    auto scratch = reinterpret_cast<device const uint*>(entries + 2ul * plan.limit);
    selected += max(scratch[i - 1], tiles[ulong(feature) * p.tiles + (i - 1) / 256]);
  }
  cuts[ulong(feature) * p.matrix.bins + i - 1] = entries[selected].w;
}
