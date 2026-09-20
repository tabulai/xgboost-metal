// Exact positive-integer conversion, including bin counts above float32's
// consecutive-integer range. A uint's significand fits entirely in binary64.
inline ulong sketch_integer_bits(uint value) {
  if (!value) return 0;
  uint power = 31 - clz(value);
  return (ulong(power + 1023) << 52) | ((ulong(value) << (52 - power)) & kCoxFraction);
}

// Weighted rank summaries follow the pinned XGBoost sketch. Sorting is batched
// on Metal; CUDA's segmented merge/prune design motivates the device layout.
// Entry components are (minimum rank, maximum rank, minimum weight, value).
#include "sketch_sort.metal.h"
#include "sketch_arithmetic.metal.h"
inline float sketch_min_next(float4 entry) { return sketch_add(entry.x, entry.z); }
inline float sketch_max_previous(float4 entry) { return sketch_subtract(entry.y, entry.z); }

// Eight independent feature segments per threadgroup. Each SIMD lane handles
// four consecutive rows, preserving the order of missing/zero-weight removal.
kernel void sketch_input_blocks(device const float* data [[buffer(0)]],
                                 device const float* weights [[buffer(1)]],
                                 device const uchar* categorical [[buffer(2)]],
                                 device float2* values [[buffer(3)]],
                                 device SketchInputBlock* blocks [[buffer(4)]],
                                 constant SketchInputParams& p [[buffer(5)]],
                                 uint2 block [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  uint feature = block.y * 8 + tid / 32, lane = tid & 31;
  if (feature >= p.cols) return;
  uint tiles = (p.rows + 127) / 128;
  ulong index = ulong(feature) * tiles + block.x;
  if (categorical[feature]) {
    if (!lane) blocks[index] = SketchInputBlock{};
    return;
  }
  float2 local[4];
  bool valid[4];
  uint count = 0, nonmissing = 0;
  float first = 0, last = 0;
  for (uint i = 0; i < 4; ++i) {
    uint row = block.x * 128 + lane * 4 + i;
    float value = row < p.rows ? data[ulong(row) * p.cols + feature] : NAN;
    float weight = row < p.rows ? weights[row] : 0;
    nonmissing += !isnan(value);
    valid[i] = !isnan(value) && (as_type<uint>(weight) & 0x7fffffffu) != 0;
    local[i] = float2(value, weight);
    if (valid[i]) {
      if (!count) first = value;
      last = value; ++count;
    }
  }
  uint mask = simd_sum(uint(count != 0) << lane);
  uint before = mask & ((1u << lane) - 1);
  uint previous_lane = before ? 31u - clz(before) : 0;
  float previous = simd_shuffle(last, previous_lane);
  bool have_previous = before != 0;
  uint offset = simd_prefix_exclusive_sum(count), runs = 0, element = 0;
  ulong output = ulong(feature) * p.stride + block.x * 128 + offset;
  for (uint i = 0; i < 4; ++i) if (valid[i]) {
    values[output + element++] = local[i];
    runs += !have_previous || !fp32_equal(local[i].x, previous);
    previous = local[i].x; have_previous = true;
  }
  uint total = simd_sum(count), total_nonmissing = simd_sum(nonmissing), total_runs = simd_sum(runs);
  uint first_lane = mask ? 31u - clz(mask & (~mask + 1)) : 0;
  uint last_lane = mask ? 31u - clz(mask) : 0;
  float first_value = simd_shuffle(first, first_lane), last_value = simd_shuffle(last, last_lane);
  if (!lane) blocks[index] = SketchInputBlock{total, total_nonmissing, total_runs, 0, first_value, last_value, 0, 0};
}

kernel void sketch_input_offsets(device SketchInputBlock* blocks [[buffer(0)]],
                                  device SketchStats* stats [[buffer(1)]],
                                  constant SketchInputParams& p [[buffer(2)]],
                                  uint feature [[thread_position_in_grid]]) {
  if (feature >= p.cols) return;
  uint tiles = (p.rows + 127) / 128;
  SketchStats result{};
  bool have_last = false;
  float last = 0;
  for (uint tile = 0; tile < tiles; ++tile) {
    ulong index = ulong(feature) * tiles + tile;
    auto block = blocks[index];
    block.offset = result.elements;
    block.run_offset = result.runs;
    block.continued = block.elements && have_last && fp32_equal(block.first, last);
    result.elements += block.elements;
    result.nonmissing += block.nonmissing;
    result.runs += block.runs - block.continued;
    if (block.elements) { have_last = true; last = block.last; }
    blocks[index] = block;
  }
  stats[feature] = result;
}

kernel void sketch_input_compact(device const float2* values [[buffer(0)]],
                                  device const SketchInputBlock* blocks [[buffer(1)]],
                                  device float2* compact [[buffer(2)]],
                                  device uint* starts [[buffer(3)]],
                                  constant SketchInputParams& p [[buffer(4)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
  uint feature = group.y * 8 + tid / 32, lane = tid & 31;
  if (feature >= p.cols) return;
  uint tiles = (p.rows + 127) / 128;
  auto block = blocks[ulong(feature) * tiles + group.x];
  if (!block.elements) return;
  ulong column = ulong(feature) * p.stride;
  auto source = values + column + group.x * 128;
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

kernel void sketch_input_runs(device const float2* compact [[buffer(0)]],
                               device const uint* starts [[buffer(1)]],
                               device const SketchStats* stats [[buffer(2)]],
                               device float2* queue [[buffer(3)]],
                               constant SketchInputParams& p [[buffer(4)]],
                               uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.cols || id.x >= stats[id.y].runs) return;
  ulong column = ulong(id.y) * p.stride;
  uint begin = starts[column + id.x];
  uint end = id.x + 1 < stats[id.y].runs ? starts[column + id.x + 1] : stats[id.y].elements;
  float2 value = compact[column + begin];
  if (p.unit_weights) value.y = float(min(end - begin, 1u << 24));
  else for (uint i = begin + 1; i < end; ++i) value.y = sketch_add(value.y, compact[column + i].y);
  queue[column + id.x] = value;
}

kernel void sketch_sort_blocks(device const float2* input [[buffer(0)]],
                                device const SketchBatch* batches [[buffer(1)]],
                                device float2* output [[buffer(2)]],
                                constant SketchParams& p [[buffer(3)]],
                                uint2 block [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]]) {
  if (block.y >= p.tasks) return;
  auto task = batches[block.y];
  if (block.x * 256 >= task.stride) return;
  threadgroup float2 values[256];
  uint index = block.x * 256 + tid;
  values[tid] = index < task.size ? input[task.input + index] : float2(INFINITY, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) {
    for (uint j = k >> 1; j; j >>= 1) {
      float2 a = values[tid], b = values[tid ^ j];
      bool take_min = ((tid & k) == 0) == ((tid & j) == 0);
      float2 next = take_min ? (fp32_less_equal(a.x, b.x) ? a : b) : (fp32_greater_equal(a.x, b.x) ? a : b);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      values[tid] = next;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  output[task.offset + index] = values[tid];
}

kernel void sketch_merge_sorted(device const float2* input [[buffer(0)]],
                                 device const SketchBatch* batches [[buffer(1)]],
                                 device float2* output [[buffer(2)]],
                                 constant SketchParams& p [[buffer(3)]],
                                 uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.tasks) return;
  auto task = batches[id.y];
  if (id.x >= task.stride) return;
  uint begin = (id.x / (2 * p.width)) * (2 * p.width);
  uint middle = min(begin + p.width, task.stride), end = min(begin + 2 * p.width, task.stride);
  bool left = id.x < middle;
  uint own = left ? begin : middle;
  uint other = left ? middle : begin;
  uint lo = other, hi = left ? end : middle;
  float2 value = input[task.offset + id.x];
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    float candidate = input[task.offset + mid].x;
    if (fp32_less(candidate, value.x) || (!left && fp32_equal(candidate, value.x))) lo = mid + 1;
    else hi = mid;
  }
  output[task.offset + begin + (id.x - own) + (lo - other)] = value;
}

inline uint sketch_prune_serial(device float4* entries, uint size, uint capacity) {
  if (size <= capacity) return size;
  if (!capacity) return 0;
  if (capacity == 1) return 1;
  float begin = entries[0].y, range = sketch_subtract(entries[size - 1].x, begin);
  uint n = capacity - 1, count = 1, i = 1, last = 0;
  float4 tail = entries[size - 1], left = entries[1], right = entries[2];
  for (uint k = 1; k < n; ++k) {
    float rank = sketch_multiply(2, sketch_add(sketch_divide(sketch_multiply(float(k), range), float(n)), begin));
    while (i < size - 1 && fp32_greater_equal(rank, sketch_add(right.y, right.x))) {
      ++i; left = right;
      if (i < size - 1) right = entries[i + 1];
    }
    if (i == size - 1) break;
    if (fp32_less(rank, sketch_add(sketch_min_next(left), sketch_max_previous(right)))) {
      if (i != last) { entries[count++] = left; last = i; }
    } else if (i + 1 != last) {
      entries[count++] = right; last = i + 1;
    }
  }
  if (last != size - 1) entries[count++] = tail;
  return count;
}

inline uint sketch_summarize_ordered(device float2* source, SketchBatch task,
                                      device float4* output, device const float2* queue) {
  bool repair = false;
  for (uint i = 1; i < task.size; ++i) {
    float2 left = source[i - 1], right = source[i];
    if (fp32_equal(left.x, right.x) &&
        (as_type<uint>(left.x) != as_type<uint>(right.x) ||
         (!task.reserved && !fp32_equal(left.y, right.y)))) { repair = true; break; }
  }
  if (repair) {
    for (uint i = 0; i < task.size; ++i) source[i] = queue[task.input + i];
    sketch_tie_sort(source, task.size);
  }
  uint size = 0;
  float total = 0;
  for (uint i = 0; i < task.size;) {
    float2 value = source[i];
    uint j = i + 1;
    while (j < task.size && fp32_equal(source[j].x, value.x)) value.y = sketch_add(value.y, source[j++].y);
    output[size++] = float4(total, sketch_add(total, value.y), value.y, value.x);
    total = sketch_add(total, value.y);
    i = j;
  }
  return sketch_prune_serial(output, size, task.limit);
}

kernel void sketch_summarize(device float2* input [[buffer(0)]],
                             device const SketchBatch* batches [[buffer(1)]],
                             device float4* summaries [[buffer(2)]],
                             device uint* sizes [[buffer(3)]],
                             device const float2* queue [[buffer(4)]],
                             constant SketchParams& p [[buffer(5)]],
                             uint batch [[thread_position_in_grid]]) {
  if (batch >= p.tasks) return;
  auto task = batches[batch];
  sizes[batch] = sketch_summarize_ordered(input + task.offset, task, summaries + task.offset, queue);
}

kernel void sketch_summarize_pending(device float2* input [[buffer(0)]],
                                     device const SketchBatch* batches [[buffer(1)]],
                                     device float4* summaries [[buffer(2)]],
                                     device uint* sizes [[buffer(3)]],
                                     device const float2* queue [[buffer(4)]],
                                     constant SketchParams& p [[buffer(5)]],
                                     uint batch [[thread_position_in_grid]]) {
  if (batch >= p.tasks || sizes[batch] != 0xffffffffu) return;
  auto task = batches[batch];
  sizes[batch] = sketch_summarize_ordered(input + task.offset, task, summaries + task.offset, queue);
}

inline uint sketch_combine_serial(device float4* current, uint n,
                                  device const float4* other, uint m,
                                  device float4* scratch) {
  if (!m) return n;
  if (!n) {
    for (uint i = 0; i < m; ++i) current[i] = other[i];
    return m;
  }
  uint a = 0, b = 0, count = 0;
  float previous_a = 0, previous_b = 0;
  while (a < n && b < m) {
    float4 x = current[a], y = other[b], result;
    if (fp32_equal(x.w, y.w)) {
      result = float4(sketch_add(x.x, y.x), sketch_add(x.y, y.y), sketch_add(x.z, y.z), x.w);
      previous_a = sketch_min_next(x); previous_b = sketch_min_next(y);
      ++a; ++b;
    } else if (fp32_less(x.w, y.w)) {
      result = float4(sketch_add(x.x, previous_b), sketch_add(x.y, sketch_max_previous(y)), x.z, x.w);
      previous_a = sketch_min_next(x); ++a;
    } else {
      result = float4(sketch_add(y.x, previous_a), sketch_add(y.y, sketch_max_previous(x)), y.z, y.w);
      previous_b = sketch_min_next(y); ++b;
    }
    scratch[count++] = result;
  }
  float tail_a = current[n - 1].y, tail_b = other[m - 1].y;
  while (a < n) {
    auto x = current[a++];
    scratch[count++] = float4(sketch_add(x.x, previous_b), sketch_add(x.y, tail_b), x.z, x.w);
  }
  while (b < m) {
    auto y = other[b++];
    scratch[count++] = float4(sketch_add(y.x, previous_a), sketch_add(y.y, tail_a), y.z, y.w);
  }
  float previous_min = 0, previous_max = 0;
  for (uint i = 0; i < count; ++i) {
    auto entry = scratch[i];
    if (fp32_less(entry.x, previous_min)) entry.x = previous_min;
    else previous_min = entry.x;
    if (fp32_less(entry.y, previous_max)) entry.y = previous_max;
    if (fp32_less(entry.y, sketch_min_next(entry))) entry.y = sketch_min_next(entry);
    previous_max = entry.y;
    current[i] = entry;
  }
  return count;
}

#include "sketch_merge.metal.h"
#include "sketch_summary.metal.h"
#include "sketch_weighted.metal.h"

kernel void sketch_reduce_serial(device const float4* summaries [[buffer(0)]],
                                 device const uint* sizes [[buffer(1)]],
                                 device const SketchBatch* batches [[buffer(2)]],
                                 device const SketchFeature* features [[buffer(3)]],
                                 device float4* levels [[buffer(4)]],
                                 device float4* workspace [[buffer(5)]],
                                 device SketchStats* stats [[buffer(6)]],
                                 constant SketchParams& p [[buffer(7)]],
                                 uint feature [[thread_position_in_grid]]) {
  if (feature >= p.features) return;
  auto column = features[feature];
  auto current = workspace + column.work_offset;
  auto scratch = current + 2 * column.limit;
  auto resident = levels + column.level_offset;
  uint counts[64] = {};
  for (uint batch = column.batch_begin; batch < column.batch_begin + column.batches; ++batch) {
    uint n = sizes[batch], level = 0;
    auto source = summaries + batches[batch].offset;
    for (uint i = 0; i < n; ++i) current[i] = source[i];
    while (true) {
      n = sketch_prune_serial(current, n, column.limit);
      n = sketch_combine_serial(current, n, resident + ulong(level) * column.limit, counts[level], scratch);
      counts[level] = 0;
      if (n <= column.limit) break;
      ++level;
    }
    counts[level] = n;
    for (uint i = 0; i < n; ++i) resident[ulong(level) * column.limit + i] = current[i];
  }
  uint n = 0;
  for (uint level = 0; level < column.levels; ++level) {
    n = sketch_combine_serial(current, n, resident + ulong(level) * column.limit, counts[level], scratch);
    n = sketch_prune_serial(current, n, max(column.budget, column.limit));
  }
  stats[feature].summary = sketch_prune_serial(current, n, column.budget);
}

inline void sketch_cuts_serial(device const float4* workspace,
                                device const SketchFeature* features,
                                device const SketchStats* stats,
                                device float* cuts, device uint* counts,
                                device float* minima, device const uchar* categorical,
                                constant MatrixParams& p, uint feature) {
  if (feature >= p.cols) return;
  auto output = cuts + ulong(feature) * p.bins;
  if (categorical[feature]) {
    for (uint i = 0; i < counts[feature]; ++i) output[i] = float(i);
    minima[feature] = 0;
    return;
  }
  minima[feature] = -INFINITY;
  uint size = stats[feature].summary, count = 0;
  if (!size) { output[0] = 1e-5f; counts[feature] = 1; return; }
  auto entries = workspace + features[feature].work_offset;
  if (size <= p.numeric_bins) {
    float last = entries[0].w;
    for (uint i = 1; i < size; ++i) {
      float value = entries[i].w;
      if (fp32_less(last, value)) { output[count++] = value; last = value; }
    }
  } else {
    float last = entries[0].w;
    uint next = 1, cursor = 0;
    ulong total = cox_float_bits(entries[size - 1].y), denominator = sketch_integer_bits(p.numeric_bins);
    for (uint i = 1; i < p.numeric_bins; ++i) {
      ulong rank = cox_scale_bits(cox_divide_bits(cox_multiply_bits(sketch_integer_bits(i), total), denominator), 1);
      while (cursor < size - 2 && rank >= cox_float_bits(sketch_add(entries[cursor + 1].x, entries[cursor + 1].y))) ++cursor;
      float boundary = sketch_add(sketch_min_next(entries[cursor]), sketch_max_previous(entries[cursor + 1]));
      float value = entries[cursor + uint(isnan(boundary) || rank >= cox_float_bits(boundary))].w;
      if (fp32_less_equal(value, last)) {
        while (next < size && fp32_less_equal(entries[next].w, last)) ++next;
        if (next == size) break;
        value = entries[next].w;
      } else if (next < size && fp32_less_equal(entries[next].w, value)) {
        do { ++next; } while (next < size && fp32_less_equal(entries[next].w, value));
      }
      output[count++] = value;
      last = value;
    }
  }
  float tail = entries[size - 1].w;
  output[count++] = tail + (fabs(tail) + 1e-5f);
  counts[feature] = count;
}

kernel void sketch_cuts(device const float4* workspace [[buffer(0)]],
                        device const SketchFeature* features [[buffer(1)]],
                        device const SketchStats* stats [[buffer(2)]],
                        device float* cuts [[buffer(3)]],
                        device uint* counts [[buffer(4)]],
                        device float* minima [[buffer(5)]],
                        device const uchar* categorical [[buffer(6)]],
                        constant MatrixParams& p [[buffer(7)]],
                        uint feature [[thread_position_in_grid]]) {
  sketch_cuts_serial(workspace, features, stats, cuts, counts, minima, categorical, p, feature);
}

#include "sketch_cuts.metal.h"
