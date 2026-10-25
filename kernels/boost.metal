#include "metalgb/shared.h"
#include "float_order.metal.h"

// Vector stores keep zeroing within a serial compute encoder. Grid-stride
// iteration also supports buffers larger than a single 32-bit thread grid.
kernel void fill_zero(device uint4* output [[buffer(0)]],
                       constant ulong& bytes [[buffer(1)]],
                       uint3 id [[thread_position_in_grid]],
                       uint3 threads [[threads_per_grid]]) {
  ulong vectors = bytes / 16;
  for (ulong i = id.x; i < vectors; i += threads.x) output[i] = uint4(0);
  if (id.x == 0) {
    auto tail = reinterpret_cast<device uchar*>(output);
    for (ulong i = vectors * 16; i < bytes; ++i) tail[i] = 0;
  }
}

// Fixed-point histogram sums are independent of thread scheduling. Two 32-bit
// atomics implement an exact 64-bit sum on every supported Apple GPU.
inline void add64(threadgroup atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint old = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
  atomic_fetch_add_explicit(dst + 1, uint(bits >> 32) + uint(old + low < old), memory_order_relaxed);
}
inline void add64(device atomic_uint* dst, long value) {
  ulong bits = as_type<ulong>(value);
  uint low = uint(bits);
  uint old = atomic_fetch_add_explicit(dst, low, memory_order_relaxed);
  atomic_fetch_add_explicit(dst + 1, uint(bits >> 32) + uint(old + low < old), memory_order_relaxed);
}
inline long load64(threadgroup atomic_uint* src) {
  uint lo = atomic_load_explicit(src, memory_order_relaxed);
  uint hi = atomic_load_explicit(src + 1, memory_order_relaxed);
  return as_type<long>(uint2(lo, hi));
}
inline float2 load_hist(device const uint* hist, ulong offset, float2 scale) {
  ulong g = ulong(hist[offset]) | (ulong(hist[offset + 1]) << 32);
  ulong h = ulong(hist[offset + 2]) | (ulong(hist[offset + 3]) << 32);
  return float2(float(as_type<long>(g)), float(as_type<long>(h))) / scale;
}
inline long2 load_hist_int(device const uint* hist, ulong offset) {
  ulong g = ulong(hist[offset]) | (ulong(hist[offset + 1]) << 32);
  ulong h = ulong(hist[offset + 2]) | (ulong(hist[offset + 3]) << 32);
  return long2(as_type<long>(g), as_type<long>(h));
}
inline uint hash32(uint x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float log1p_stable(float x) {
  float y = 1 + x;
  return y == 1 ? x : log(y) * x / (y - 1);
}
inline float huber_curvature(float slope, float residual) {
  float scale = max(fabs(slope), fabs(residual));
  float a = slope / scale, b = residual / scale;
  return fabs(a) / sqrt(a * a + b * b);
}

#include "objectives.metal.h"
#include "distributions.metal.h"
#include "survival.metal.h"
#include "ranking.metal.h"
#include "linear.metal.h"
#include "sampling.metal.h"

kernel void normalize_data(device float* data [[buffer(0)]],
                           device atomic_uint* invalid [[buffer(1)]],
                           constant NormalizeParams& p [[buffer(2)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
  ulong index = (ulong(block.y) * p.block_stride + block.x) * 256 + tid;
  if (index >= ulong(p.rows) * p.cols) return;
  float value = data[index];
  if (fp32_equal(value, p.missing)) data[index] = NAN;
  else if (isinf(value)) atomic_store_explicit(invalid, 1, memory_order_relaxed);
}

kernel void transpose_columns(device const float* columns [[buffer(0)]],
                               device float* data [[buffer(1)]],
                               device atomic_uint* invalid [[buffer(2)]],
                               constant NormalizeParams& p [[buffer(3)]],
                               uint2 block [[threadgroup_position_in_grid]],
                               uint2 lane [[thread_position_in_threadgroup]]) {
  threadgroup float tile[32][33];
  ulong tile_index = ulong(block.y) * p.block_stride + block.x;
  uint row_tiles = uint((ulong(p.rows) + 31) / 32);
  uint row_begin = uint(tile_index % row_tiles) * 32;
  ulong column_begin = (tile_index / row_tiles) * 32;
  if (column_begin >= p.cols) return;
  for (uint j = 0; j < 32; j += 8) {
    uint row = row_begin + lane.x;
    ulong column = column_begin + lane.y + j;
    tile[lane.y + j][lane.x] = row < p.rows && column < p.cols ? columns[column * p.rows + row] : NAN;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 32; j += 8) {
    uint row = row_begin + lane.y + j;
    ulong column = column_begin + lane.x;
    if (row < p.rows && column < p.cols) {
      float value = tile[lane.x][lane.y + j];
      if (fp32_equal(value, p.missing)) value = NAN;
      else if (isinf(value)) atomic_store_explicit(invalid, 1, memory_order_relaxed);
      data[ulong(row) * p.cols + column] = value;
    }
  }
}

kernel void recode_categories(device const float* input [[buffer(0)]],
                              device const uint2* offsets [[buffer(1)]],
                              device const uint* mapping [[buffer(2)]],
                              device float* output [[buffer(3)]],
                              device atomic_uint* invalid [[buffer(4)]],
                              constant CategoryRecodeParams& p [[buffer(5)]],
                              uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.cols) return;
  ulong index = ulong(id.x) * p.cols + id.y;
  float value = input[p.column_major ? ulong(id.y) * p.rows + id.x : index];
  uint2 segment = offsets[id.y];
  if (p.construction && segment.x != 0xffffffffu) {
    // The pinned Apple CPU CatAccessor casts NaN to code zero. It maps before
    // the matrix's missing-value filter, even for identical category names.
    if (segment.y && (isnan(value) || (value > -1.0f && value < float(segment.y)))) {
      uint code = isnan(value) ? 0u : uint(max(value, 0.0f));
      value = float(mapping[segment.x + code]);
    }
  } else if (!isnan(value) && segment.x != 0xffffffffu) {
    if (fp32_less(value, 0) || value >= float(segment.y) || !fp32_equal(floor(value), value)) {
      atomic_store_explicit(invalid, 1, memory_order_relaxed);
      value = NAN;
    } else value = float(mapping[segment.x + uint(value)]);
  }
  output[index] = value;
}

kernel void category_statistics(device const float* data [[buffer(0)]],
                                device const uchar* categorical [[buffer(1)]],
                                device atomic_uint* counts [[buffer(2)]],
                                device atomic_uint* invalid [[buffer(3)]],
                                constant MatrixParams& p [[buffer(4)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.cols || !categorical[id.y]) return;
  float value = id.x < p.rows ? data[ulong(id.x) * p.cols + id.y] : NAN;
  bool bad = !isnan(value) && (fp32_less(value, 0) || value >= 16777216.0f);
  uint count = !isnan(value) && !bad ? uint(value) + 1 : 0;
  uint maximum = simd_max(count), errors = simd_sum(uint(bad));
  if (simd_is_first()) {
    atomic_fetch_max_explicit(counts + id.y, maximum, memory_order_relaxed);
    if (errors) atomic_store_explicit(invalid, 1, memory_order_relaxed);
  }
}

kernel void base_partial(device const float* labels [[buffer(0)]],
                          device const float* weights [[buffer(1)]],
                          device SurvivalSum* partial [[buffer(2)]],
                          constant GradientParams& p [[buffer(3)]],
                          uint2 block [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum values[256];
  uint row = block.x * 256 + tid;
  SurvivalSum pair{sr(0), sr(0)};
  if (row < p.rows) {
    float y = labels[ulong(row) * p.label_cols + min(block.y, p.label_cols - 1)];
    float w = p.has_weight ? weights[row] : 1;
    if (p.objective == 1 || p.objective == 2) w *= y == 1 ? p.scale_pos_weight : 1;
    if (p.objective == 3 || p.objective == 4) y = float(uint(y) == block.y);
    if (p.objective == 6) y = log1p_stable(y);
    if (p.objective == 9) {
      float h = huber_curvature(p.parameter, y);
      pair = SurvivalSum{sr(y * h * w), sr(h * w)};
    } else {
      // GLM intercepts reduce weighted labels with wider intermediates in
      // both reference backends. Keep low bits and range through the mean.
      auto product = sr(y) * sr(w);
      // The reweighted logistic intercept first emits float32 gradient pairs.
      if ((p.objective == 1 || p.objective == 2) && fabs(p.scale_pos_weight - 1) > 1.0e-6f)
        product = sr(y * w);
      pair = SurvivalSum{product, sr(w)};
    }
  }
  values[tid] = pair;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (tid < delta) values[tid] = values[tid] + values[tid + delta];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) partial[ulong(block.y) * ((p.rows + 255) / 256) + block.x] = values[0];
}
kernel void base_reduce(device const SurvivalSum* partial [[buffer(0)]],
                        device float* values [[buffer(1)]],
                        constant GradientParams& p [[buffer(2)]],
                        uint group [[thread_position_in_grid]]) {
  if (group >= p.groups) return;
  SurvivalSum sum{sr(0), sr(0)};
  uint blocks = (p.rows + 255) / 256;
  for (uint i = 0; i < blocks; ++i) sum = sum + partial[ulong(group) * blocks + i];
  float value = survival_less(sr(0), sum.weight) ? extended_result(sum.value / sum.weight) :
                (p.objective == 0 && p.rows == 0 ? 0.0f : 0.5f);
  if (p.objective == 6) value = exp(value) - 1;
  values[group] = value;
}
kernel void base_finalize(device const float* values [[buffer(0)]],
                          device float* scores [[buffer(1)]],
                          device float* margins [[buffer(2)]],
                          constant GradientParams& p [[buffer(3)]],
                          uint group [[thread_position_in_grid]]) {
  if (group >= p.groups) return;
  float score = values[group], margin = score;
  if (p.objective == 3 || p.objective == 4) {
    float mean = 0;
    for (uint k = 0; k < p.groups; ++k) mean += log(max(values[k], 1.0e-6f));
    margin = log(max(score, 1.0e-6f)) - mean / float(p.groups);
    score = margin;
  } else if (p.objective == 1 || p.objective == 2) {
    float v = clamp(score, 1.0e-6f, 1.0f - 1.0e-6f);
    margin = log(v / (1 - v));
    if (p.objective == 2) score = margin;
  } else if (p.objective == 5 || p.objective == 7 || p.objective == 8) margin = log(max(score, 1.0e-16f));
  scores[group] = score; margins[group] = margin;
}

kernel void merge_sorted(device const float2* source [[buffer(0)]],
                         device float2* dest [[buffer(1)]],
                         constant SortParams& p [[buffer(2)]],
                         uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.stride || id.y >= p.cols) return;
  uint pair_begin = (id.x / (2 * p.width)) * (2 * p.width);
  uint middle = min(pair_begin + p.width, p.stride);
  uint end = min(pair_begin + 2 * p.width, p.stride);
  bool left = id.x < middle;
  uint own_begin = left ? pair_begin : middle;
  uint other_begin = left ? middle : pair_begin;
  uint other_end = left ? end : middle;
  ulong col = ulong(id.y) * p.stride;
  float2 value = source[col + id.x];
  uint lo = other_begin, hi = other_end;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    float other = source[col + mid].x;
    if (other < value.x || (!left && other == value.x)) lo = mid + 1;
    else hi = mid;
  }
  dest[col + pair_begin + (id.x - own_begin) + (lo - other_begin)] = value;
}

#include "quantile.metal.h"

inline void store_bin(device ushort* output, ulong index, uint value, uint capacity) {
  if (capacity <= 65534) output[index] = ushort(value);
  else reinterpret_cast<device uint*>(output)[index] = value;
}
inline uint load_bin(device const ushort* input, ulong index, uint capacity) {
  if (capacity > 65534) return reinterpret_cast<device const uint*>(input)[index];
  uint value = input[index];
  return value == 65535 ? 0xffffffffu : value;
}

kernel void quantize(device const float* data [[buffer(0)]],
                     device const float* cuts [[buffer(1)]],
                     device const uint* counts [[buffer(2)]],
                     device ushort* bins [[buffer(3)]],
                     device const uchar* categorical [[buffer(4)]],
                     device atomic_uint* missing [[buffer(5)]],
                     constant MatrixParams& p [[buffer(6)]],
                     uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.cols) return;
  float value = data[ulong(id.x) * p.cols + id.y];
  bool has_missing = simd_any(isnan(value));
  if (!(id.x & 31) && has_missing) atomic_store_explicit(missing, 1, memory_order_relaxed);
  if (categorical[id.y]) {
    uint bin = 0xffffffffu;
    if (!isnan(value) && counts[id.y]) {
      float code = float(int(value));
      auto column = cuts + ulong(id.y) * p.bins;
      uint candidate = uint(max(code, 0.0f));
      if (candidate < counts[id.y] && fp32_equal(column[candidate], code)) bin = candidate;
      else {
        uint lo = 0, hi = counts[id.y];
        while (lo < hi) {
          uint mid = lo + (hi - lo) / 2;
          if (fp32_less(column[mid], code)) lo = mid + 1; else hi = mid;
        }
        bin = min(lo, counts[id.y] - 1);
      }
    }
    store_bin(bins, ulong(id.y) * p.rows + id.x, bin, p.bins);
    return;
  }
  uint lo = 0, hi = counts[id.y];
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (fp32_greater_equal(value, cuts[ulong(id.y) * p.bins + mid])) lo = mid + 1; else hi = mid;
  }
  store_bin(bins, ulong(id.y) * p.rows + id.x,
            isnan(value) || counts[id.y] == 0 ? 0xffffffffu : min(lo, counts[id.y] - 1), p.bins);
}

// Like CUDA's EllpackLoader, prediction reads numerical bin lower bounds.
// CSR export instead uses NumericBinValue: one float below the first cut.
// A tiled transpose coalesces feature-major bin reads and row-major stores.
kernel void quantile_values(device const ushort* bins [[buffer(0)]],
                            device const float* cuts [[buffer(1)]],
                            device const uchar* categorical [[buffer(2)]],
                            device float* output [[buffer(3)]],
                            constant QuantileValueParams& p [[buffer(4)]],
                            uint2 block [[threadgroup_position_in_grid]],
                            uint2 lane [[thread_position_in_threadgroup]]) {
  threadgroup float tile[32][33];
  ulong row_begin = ulong(block.x) * 32, column_begin = ulong(block.y) * 32;
  for (uint j = 0; j < 32; j += 8) {
    ulong row = row_begin + lane.x, column = column_begin + lane.y + j;
    float value = NAN;
    if (row < p.rows && column < p.cols) {
      uint bin = load_bin(bins, column * p.rows + row, p.bins);
      if (bin != 0xffffffffu) {
        auto feature_cuts = cuts + column * p.bins;
        if (categorical[column] && !p.numeric_only) value = feature_cuts[bin];
        else if (bin) value = feature_cuts[bin - 1];
        else value = p.representative ? fp32_next_down(feature_cuts[0]) : -INFINITY;
      }
    }
    tile[lane.y + j][lane.x] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 32; j += 8) {
    ulong row = row_begin + lane.y + j, column = column_begin + lane.x;
    if (row < p.rows && column < p.cols) output[row * p.cols + column] = tile[lane.x][lane.y + j];
  }
}

kernel void quantile_category_bounds(device const float* cuts [[buffer(0)]],
                                     device const uint* counts [[buffer(1)]],
                                     device float* output [[buffer(2)]],
                                     constant MatrixParams& p [[buffer(3)]],
                                     uint feature [[thread_position_in_grid]]) {
  if (feature >= p.cols) return;
  // A numeric-only cut cache keeps numeric row partitioning after a type
  // change. CommonRowPartitioner searches for the category node's bit-1
  // split condition and uses bin -1 when that value is absent from the cuts.
  float condition = as_type<float>(1u);
  auto column = cuts + ulong(feature) * p.bins;
  uint lo = 0, hi = counts[feature];
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (fp32_less(column[mid], condition)) lo = mid + 1; else hi = mid;
  }
  output[feature] = lo < counts[feature] && fp32_equal(column[lo], condition) ? condition : -INFINITY;
}

kernel void gradients(device const float* predictions [[buffer(0)]],
                      device const float* labels [[buffer(1)]],
                      device const float* weights [[buffer(2)]],
                      device Pair* output [[buffer(3)]],
                      constant GradientParams& p [[buffer(4)]],
                      uint2 id [[thread_position_in_grid]]) {
  uint row = id.x, group = id.y;
  if (row >= p.rows || group >= p.groups) return;
  ulong idx = ulong(row) * p.groups + group;
  float pred = predictions[idx];
  float label = labels[ulong(row) * p.label_cols + min(group, p.label_cols - 1)];
  float weight = p.has_weight ? weights[row] : 1.0f;
  float g = pred - label, h = 1.0f;
  switch (p.objective) {
    case 1: case 2: {
      float prob = sigmoid(pred);
      weight *= label == 1.0f ? p.scale_pos_weight : 1.0f;
      g = prob - label; h = max(prob * (1.0f - prob), 1.0e-16f); break;
    }
    case 3: case 4: {
      float max_value = 0x1p-126f;
      float2 denom = 0;
      for (uint k = 0; k < p.groups; ++k) max_value = max(max_value, predictions[ulong(row) * p.groups + k]);
      for (uint k = 0; k < p.groups; ++k)
        denom = accurate_add(denom, float2(extended_result(survival_exp(sr(predictions[ulong(row) * p.groups + k] - max_value))), 0));
      float prob = extended_result(survival_exp(sr(pred - max_value))) / (denom.x + denom.y);
      g = prob - float(uint(label) == group);
      output[idx] = Pair{g * weight, max(fabs(g) * weight, 1.0e-16f)};
      return;
    }
    case 5: { float mu = exp(pred); g = mu - label; h = (2 * mu + label) / 3; break; }
    case 6: {
      pred = max(pred, -1.0f + 1.0e-6f);
      float diff = log1p_stable(pred) - log1p_stable(label);
      g = diff / (pred + 1); h = max((1 - diff) / ((pred + 1) * (pred + 1)), 1.0e-6f); break;
    }
    case 7: { float ratio = label / exp(pred); g = 1 - ratio; h = (2 * ratio + 1) / 3; break; }
    case 8: {
      float a = exp((1 - p.parameter) * pred), b = exp((2 - p.parameter) * pred);
      g = -label * a + b; h = (p.parameter * label * a + (3 - p.parameter) * b) / 3; break;
    }
    case 9: {
      h = huber_curvature(p.parameter, pred - label);
      g = (pred - label) * h; break;
    }
    case 10: g = pred * (2 * label - 1) < 1 ? -(2 * label - 1) : 0; h = g == 0 ? 1.0e-16f : 1; break;
  }
  output[idx] = Pair{g * weight, h * weight};
}

#include "gradient_quantization.metal.h"

kernel void quantize_gradients(device const Pair* gradient [[buffer(0)]],
                               device const float2* scales [[buffer(1)]],
                               device long2* output [[buffer(2)]],
                               constant GradientParams& p [[buffer(3)]],
                               uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  ulong index = ulong(id.x) * p.groups + id.y;
  Pair value = gradient[index];
  float2 scale = scales[id.y];
  output[index] = long2(long(rint(value.g * scale.x)), long(rint(value.h * scale.y)));
}

// Generate work from GPU-resident row ranges, so a level can partition and build
// child histograms without copying counts to the CPU between the two stages.
kernel void plan_histograms(device const uint* counts [[buffer(0)]],
                            device const uint* offsets [[buffer(1)]],
                            device const uint* parents [[buffer(2)]],
                            device HistTask* tasks [[buffer(3)]],
                            device HistSubtract* subtraction [[buffer(4)]],
                            device atomic_uint* arguments [[buffer(5)]],
                            constant HistogramPlanParams& p [[buffer(6)]],
                            uint node [[thread_position_in_grid]]) {
  if (node >= p.nodes) return;
  if (p.subtract) {
    uint first = node & ~1u;
    uint small = counts[first] <= counts[first + 1] ? first : first + 1;
    if (node == first) subtraction[first / 2] = HistSubtract{parents[first / 2], small, small ^ 1u, 0};
    if (node != small) return;
  }
  uint count = counts[node];
  uint tiles = count / p.rows_per_tile + uint(count % p.rows_per_tile != 0);
  uint groups = (p.cols - 1) / p.features + 1;
  uint cursor = atomic_fetch_add_explicit(arguments, tiles * groups, memory_order_relaxed);
  for (uint f = 0; f < groups; ++f) for (uint tile = 0; tile < tiles; ++tile) {
    uint row = tile * p.rows_per_tile;
    uint begin = offsets[node] + row, end = begin + min(p.rows_per_tile, count - row);
    tasks[cursor++] = HistTask{begin, end, node, f * p.features};
  }
}

kernel void histogram(device const ushort* bins [[buffer(0)]],
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
    for (uint i = tid; i < length; i += 256) atomic_store_explicit(local + i, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  ulong offset = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * length;
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    uint bin = load_bin(bins, ulong(task.feature) * p.rows + row, p.bins);
    if (bin == 0xffffffffu) bin = p.bins;
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    long g = value.x, h = value.y;
    if (p.bins <= 512) { add64(local + bin * 4, g); add64(local + bin * 4 + 2, h); }
    else { add64(hist + offset + bin * 4, g); add64(hist + offset + bin * 4 + 2, h); }
  }
  if (p.bins <= 512) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bin = tid; bin <= p.bins; bin += 256) {
      add64(hist + offset + bin * 4, load64(local + bin * 4));
      add64(hist + offset + bin * 4 + 2, load64(local + bin * 4 + 2));
    }
  }
}

kernel void histogram_grouped(device const ushort* bins [[buffer(0)]],
                              device const uint* row_ids [[buffer(1)]],
                              device const long2* gradients [[buffer(2)]],
                              device const float2* scales [[buffer(3)]],
                              device const HistTask* tasks [[buffer(4)]],
                              device atomic_uint* hist [[buffer(5)]],
                              constant HistParams& p [[buffer(6)]],
                              uint2 block [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]]) {
  // Stay within 16 KiB, including under shader validation's shadow allocation.
  threadgroup atomic_uint local[4 * 1024];
  HistTask task = tasks[block.x];
  uint features = min(p.padding0, p.cols - task.feature);
  uint bins_per_feature = p.bins + 1;
  uint length = features * p.bins;
  ulong global_begin = ((ulong(task.node) * max(p.padding1, 1u) + block.y) * p.cols + task.feature) * bins_per_feature * 4;
  for (uint i = tid; i < length * 4; i += 256) atomic_store_explicit(local + i, 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = task.begin + tid; i < task.end; i += 256) {
    uint row = row_ids[i];
    long2 value = gradients[ulong(row) * p.groups + p.group + block.y];
    for (uint f = 0; f < features; ++f) {
      uint bin = load_bin(bins, ulong(task.feature + f) * p.rows + row, p.bins);
      if (bin == 0xffffffffu) {
        ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
        add64(hist + offset, value.x); add64(hist + offset + 2, value.y);
      } else {
        uint offset = (f * p.bins + bin) * 4;
        add64(local + offset, value.x); add64(local + offset + 2, value.y);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint bin = tid; bin < length; bin += 256) {
    ulong offset = global_begin + (bin / p.bins * bins_per_feature + bin % p.bins) * 4;
    add64(hist + offset, load64(local + bin * 4));
    add64(hist + offset + 2, load64(local + bin * 4 + 2));
  }
}

#include "histogram_missing.metal.h"
#include "histogram_present.metal.h"
#include "histogram_compact.metal.h"

kernel void subtract_histograms(device const uint* parent [[buffer(0)]],
                                device const HistSubtract* tasks [[buffer(1)]],
                                device uint* histogram [[buffer(2)]],
                                constant HistParams& p [[buffer(3)]],
                                uint2 id [[thread_position_in_grid]]) {
  ulong entries = ulong(max(p.padding1, 1u)) * p.cols * (p.bins + 1);
  if (id.x >= entries) return;
  HistSubtract task = tasks[id.y];
  long2 value = load_hist_int(parent, (ulong(task.parent) * entries + id.x) * 4) -
                load_hist_int(histogram, (ulong(task.built) * entries + id.x) * 4);
  ulong g = as_type<ulong>(value.x), h = as_type<ulong>(value.y);
  ulong offset = (ulong(task.other) * entries + id.x) * 4;
  histogram[offset] = uint(g); histogram[offset + 1] = uint(g >> 32);
  histogram[offset + 2] = uint(h); histogram[offset + 3] = uint(h >> 32);
}

kernel void copy_histograms(device const uint* source [[buffer(0)]],
                            device const uint2* mapping [[buffer(1)]],
                            device uint* destination [[buffer(2)]],
                            constant HistParams& p [[buffer(3)]],
                            uint2 id [[thread_position_in_grid]]) {
  ulong words = ulong(max(p.padding1, 1u)) * p.cols * (p.bins + 1) * 4;
  if (id.x >= words) return;
  uint2 nodes = mapping[id.y];
  destination[ulong(nodes.y) * words + id.x] = source[ulong(nodes.x) * words + id.x];
}

#include "split_statistics.metal.h"
#include "scalar_gain.metal.h"
#include "scalar_weight_fast.metal.h"
#include "scalar_gain_fast.metal.h"
#include "scalar_statistics_fast.metal.h"
#include "scalar_score_fused.metal.h"

inline SurvivalReal histogram_value(long integer, float scale) {
  auto value = ranking_integer(integer);
  // The histogram scale is a power of two. Retain the low bits of the integer
  // sum through weight and gain evaluation instead of rounding each child to f32.
  int exponent = 0;
  frexp(scale, exponent);
  value.exponent -= exponent - 1;
  return value;
}
inline SurvivalSum histogram_sum(long2 value, float2 scale) {
  return SurvivalSum{histogram_value(value.x, scale.x), histogram_value(value.y, scale.y)};
}
inline bool histogram_fast_value(SurvivalReal value) {
  return value.significand.x == 0 || (isfinite(value.significand.x) && value.exponent >= -30 && value.exponent <= 30);
}
inline bool histogram_fast_scalar(float value) {
  return value == 0 || (fabs(value) >= 0x1p-30f && fabs(value) <= 0x1p30f);
}
inline bool histogram_fast_sum(SurvivalSum value, constant SplitParams& p) {
  return histogram_fast_value(value.value) && histogram_fast_value(value.weight) &&
         histogram_fast_scalar(p.reg_lambda) && histogram_fast_scalar(p.reg_alpha);
}
inline float2 histogram_significand(SurvivalReal value) {
  return ldexp(value.significand, int2(value.exponent));
}
inline float leaf_weight(SurvivalSum value, constant SplitParams& p, float2 bounds) {
  if (!survival_less(sr(0), value.weight)) return 0;
  if (histogram_fast_sum(value, p)) {
    auto gradient = histogram_significand(value.value);
    auto positive = accurate_add(gradient, float2(-p.reg_alpha, 0));
    auto negative = accurate_add(gradient, float2(p.reg_alpha, 0));
    gradient = positive.x > 0 ? positive : negative.x < 0 ? negative : float2(0);
    auto denominator = accurate_add(histogram_significand(value.weight), float2(p.reg_lambda, 0));
    auto weight = accurate_divide(-gradient, denominator);
    float w = gradient.x == 0 ? -0.0f : weight.x + weight.y;
    if (p.max_delta_step != 0) w = clamp(w, -p.max_delta_step, p.max_delta_step);
    return clamp(w, bounds.x, bounds.y);
  }
  auto gradient = sr(0);
  if (survival_less(sr(p.reg_alpha), value.value)) gradient = value.value - sr(p.reg_alpha);
  else if (survival_less(value.value, sr(-p.reg_alpha))) gradient = value.value + sr(p.reg_alpha);
  float w = gradient.significand.x == 0 ? -0.0f : extended_result(-gradient / (value.weight + sr(p.reg_lambda)));
  if (p.max_delta_step != 0) w = clamp(w, -p.max_delta_step, p.max_delta_step);
  return clamp(w, bounds.x, bounds.y);
}
// As in CUDA XGBoost, sort categorical bins by their leaf weight. Stable bin-ID
// ties make split selection reproducible; merges handle features above 256 bins.
kernel void category_sort_blocks(device const uint* histogram [[buffer(0)]],
                                 device const float2* scales [[buffer(1)]],
                                 device const uint* counts [[buffer(2)]],
                                 device const uchar* categorical [[buffer(3)]],
                                 device float2* output [[buffer(4)]],
                                 constant SplitParams& p [[buffer(5)]],
                                 uint2 block [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float2 values[256];
  uint bin = block.x * 256 + tid, feature = block.y % p.cols;
  float weight = INFINITY;
  if (categorical[feature] && bin < counts[feature]) {
    auto sum = scalar_histogram_sum(load_hist_int(histogram, (ulong(block.y) * (p.bins + 1) + bin) * 4), scales[p.group]);
    // The pinned reference applies min_child_weight to candidate children, not
    // to individual bins when ordering categories.
    weight = scalar_weight(sum, p, float2(-INFINITY, INFINITY));
  }
  values[tid] = float2(weight, float(bin));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) for (uint j = k >> 1; j; j >>= 1) {
    float2 a = values[tid], b = values[tid ^ j];
    bool take_min = ((tid & k) == 0) == ((tid & j) == 0);
    bool ordered = a.x < b.x || (a.x == b.x && a.y <= b.y);
    float2 next = take_min == ordered ? a : b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  output[ulong(block.y) * p.sort_stride + bin] = values[tid];
}

kernel void category_sort_ranks(device const float2* sorted [[buffer(0)]],
                                device const uint* counts [[buffer(1)]],
                                device const uchar* categorical [[buffer(2)]],
                                device uint* ranks [[buffer(3)]],
                                constant SplitParams& p [[buffer(4)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.y >= p.nodes * p.cols || !categorical[id.y % p.cols] || id.x >= counts[id.y % p.cols]) return;
  uint bin = uint(sorted[ulong(id.y) * p.sort_stride + id.x].y);
  ranks[ulong(id.y) * p.bins + bin] = id.x;
}
inline void consider_split(thread Split& best, long2 left_int, long2 right_int, float2 scale, float threshold,
                           uint default_left, uint feature, int monotone,
                           float2 bounds, float parent_gain, constant SplitParams& p) {
  auto left = scalar_histogram_sum(left_int, scale), right = scalar_histogram_sum(right_int, scale);
  if (!scalar_valid_hessian(left.y, p.min_child_weight) ||
      !scalar_valid_hessian(right.y, p.min_child_weight)) return;
  float lw = scalar_weight(left, p, bounds), rw = scalar_weight(right, p, bounds);
  if ((monotone > 0 && !fp32_less_equal(lw, rw)) || (monotone < 0 && !fp32_greater_equal(lw, rw))) return;
  float gain = scalar_float_add(scalar_float_add(scalar_gain(left, lw, p), scalar_gain(right, rw, p)), -parent_gain);
  if (gain > best.gain) {
    best.gain = gain; best.threshold = threshold; best.feature = feature;
    best.default_left = default_left;
    best.left_h = cox_float_result(left.y);
    best.right_h = cox_float_result(right.y);
    best.left_weight = lw; best.right_weight = rw;
    best.left_sum = IntPair{left_int.x, left_int.y};
    best.right_sum = IntPair{right_int.x, right_int.y};
  }
}
inline void evaluate_feature(device const uint* histogram, device const float2* scales,
                              device const float* cuts, device const uint* counts, device const float* minima,
                              device const int* monotone, device const float2* bounds, device const uchar* allowed,
                              device Split* output, device const uchar* categorical, device const float2* sorted,
                              device const uint* ranks, device uint* categories, device const long2* node_sums,
                              constant SplitParams& p, uint2 id) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  uint feature = id.x, node = id.y, count = counts[feature];
  ulong offset = (ulong(node) * p.cols + feature) * (p.bins + 1) * 4;
  float2 scale = scales[p.group];
  long2 missing = load_hist_int(histogram, offset + p.bins * 4);
  long2 present = 0;
  for (uint b = 0; b < count; ++b) present += load_hist_int(histogram, offset + b * 4);
  long2 total_int = p.use_parent_sums ? node_sums[node] : present + missing;
  missing = total_int - present;
  auto total = scalar_histogram_sum(total_int, scale);
  float w = scalar_weight(total, p, bounds[node]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, cox_float_result(total.x), cox_float_result(total.y)};
  best.parent_weight = w;
  if (allowed[ulong(node) * p.cols + feature]) {
    float parent_gain = scalar_gain(total, w, p);
    if (p.has_categorical && categorical[feature]) {
      ulong cat_offset = (ulong(node) * p.cols + feature) * p.bins;
      if (count < p.max_cat_to_onehot) {
        uint chosen = 0;
        for (uint b = 0; b < count; ++b) {
          long2 right = load_hist_int(histogram, offset + b * 4);
          float previous = best.gain;
          consider_split(best, total_int - right, right, scale, 0, 1,
                         feature, monotone[feature], bounds[node], parent_gain, p);
          if (best.gain > previous) { chosen = b; best.category_count = 1; }
          right += missing;
          previous = best.gain;
          consider_split(best, total_int - right, right, scale, 0, 0,
                         feature, monotone[feature], bounds[node], parent_gain, p);
          if (best.gain > previous) { chosen = b; best.category_count = 1; }
        }
        if (best.category_count) categories[cat_offset] = uint(cuts[ulong(feature) * p.bins + chosen]);
      } else {
        uint limit = min(count, p.max_cat_threshold);
        ulong order = (ulong(node) * p.cols + feature) * p.sort_stride;
        long2 right = 0, left = 0;
        for (uint i = 0; i + 1 < limit; ++i) {
          uint b = uint(sorted[order + i].y);
          right += load_hist_int(histogram, offset + b * 4);
          float previous = best.gain;
          consider_split(best, total_int - right, right, scale, 0, 1,
                         feature, monotone[feature], bounds[node], parent_gain, p);
          if (best.gain > previous) best.category_count = i + 1;
        }
        for (uint step = 1; step < limit; ++step) {
          uint i = count - step, b = uint(sorted[order + i].y);
          left += load_hist_int(histogram, offset + b * 4);
          float previous = best.gain;
          consider_split(best, left, total_int - left, scale, 0, 0,
                         feature, monotone[feature], bounds[node], parent_gain, p);
          if (best.gain > previous) best.category_count = i;
        }
        uint written = 0;
        for (uint b = 0; b < count; ++b) if (ranks[cat_offset + b] < best.category_count) {
          uint code = uint(cuts[ulong(feature) * p.bins + b]);
          if (!written || categories[cat_offset + written - 1] != code) categories[cat_offset + written++] = code;
        }
        best.category_count = written;
      }
      best.split_type = 1; best.category_begin = uint(cat_offset);
      output[ulong(node) * p.cols + feature] = best;
      return;
    }
    long2 left = 0;
    for (uint b = 0; b < count; ++b) {
      left += load_hist_int(histogram, offset + b * 4);
      consider_split(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + b], 0,
                     feature, monotone[feature], bounds[node], parent_gain, p);
    }
    long2 right = 0;
    for (int b = (missing.x != 0 || missing.y != 0) ? int(count) - 1 : -1; b >= 0; --b) {
      right += load_hist_int(histogram, offset + uint(b) * 4);
      float threshold = b == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + b - 1];
      consider_split(best, total_int - right, right, scale, threshold, 1,
                     feature, monotone[feature], bounds[node], parent_gain, p);
    }
  }
  output[ulong(node) * p.cols + feature] = best;
}

kernel void evaluate_splits(device const uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
                            device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
                            device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
                            device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
                            device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
                            device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
                            device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
                            constant SplitParams& p [[buffer(14)]],
                            uint2 id [[thread_position_in_grid]]) {
  evaluate_feature(histogram, scales, cuts, counts, minima, monotone, bounds, allowed, output,
                     categorical, sorted, ranks, categories, node_sums, p, id);
}

inline long2 histogram_prefix(long2 value, uint lane) {
  for (uint delta = 1; delta < 32; delta <<= 1) {
    auto preceding = as_type<long2>(simd_shuffle_up(as_type<uint4>(value), delta));
    if (lane >= delta) value += preceding;
  }
  return value;
}
inline long2 histogram_broadcast(long2 value, uint lane) {
  return as_type<long2>(simd_broadcast(as_type<uint4>(value), lane));
}

// CUDA's evaluator assigns one warp to a feature. A Metal SIMD group likewise
// scans integer histogram bins in parallel, retaining the serial implementation's
// forward-then-backward candidate order when gains tie.
kernel void evaluate_splits_simd(device const uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
                                 device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
                                 device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
                                 device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
                                 device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
                                 device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
                                 device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
                                 constant SplitParams& p [[buffer(14)]],
                                 uint2 id [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  if (p.has_categorical && categorical[id.x]) {
    if (!lane) evaluate_feature(histogram, scales, cuts, counts, minima, monotone, bounds, allowed, output,
                                  categorical, sorted, ranks, categories, node_sums, p, id);
    return;
  }
  uint feature = id.x, node = id.y, count = counts[feature];
  ulong offset = (ulong(node) * p.cols + feature) * (p.bins + 1) * 4;
  float2 scale = scales[p.group];
  long2 missing = load_hist_int(histogram, offset + p.bins * 4), local = 0;
  for (uint b = lane; b < count; b += 32) local += load_hist_int(histogram, offset + b * 4);
  long2 present = histogram_broadcast(histogram_prefix(local, lane), 31);
  long2 total_int = p.use_parent_sums ? node_sums[node] : present + missing;
  missing = total_int - present;
  auto total = scalar_histogram_sum(total_int, scale);
  float w = scalar_weight(total, p, bounds[node]);
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, cox_float_result(total.x), cox_float_result(total.y)};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (allowed[ulong(node) * p.cols + feature]) {
    float parent_gain = scalar_gain(total, w, p);
    long2 preceding = 0;
    for (uint begin = 0; begin < count; begin += 32) {
      uint bin = begin + lane;
      long2 value = bin < count ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto left = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (bin < count) {
        float previous = best.gain;
        consider_split(best, left, total_int - left, scale, cuts[ulong(feature) * p.bins + bin], 0,
                       feature, monotone[feature], bounds[node], parent_gain, p);
        if (best.gain > previous) order = bin;
      }
    }
    preceding = 0;
    for (uint begin = 0; begin < ((missing.x != 0 || missing.y != 0) ? count : 0); begin += 32) {
      uint step = begin + lane;
      uint bin = count - min(step + 1, count);
      long2 value = step < count ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto right = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (step < count) {
        float previous = best.gain;
        float threshold = bin == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + bin - 1];
        consider_split(best, total_int - right, right, scale, threshold, 1,
                       feature, monotone[feature], bounds[node], parent_gain, p);
        if (best.gain > previous) order = count + step;
      }
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) output[ulong(node) * p.cols + feature] = best;
}
#include "scalar_sparse.metal.h"
#include "scalar_fused.metal.h"
#include "scalar_parallel.metal.h"
#include "scalar_compact.metal.h"
#include "scalar_root_parallel.metal.h"

// Split long numeric features across SIMD groups. Integer carries and the
// original forward/backward candidate positions preserve the scalar tie order.
kernel void scalar_tile_sums(device const uint* histogram [[buffer(0)]],
                              device const uint* counts [[buffer(1)]],
                              device const uchar* categorical [[buffer(2)]],
                              device long2* sums [[buffer(3)]],
                              constant ScalarTileParams& p [[buffer(4)]],
                              uint3 block [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  uint feature = block.x, node = block.y;
  if (p.split.has_categorical && categorical[feature]) return;
  auto index = ulong(node) * p.split.cols + feature;
  auto histogram_begin = index * (p.split.bins + 1) * 4;
  uint begin = block.z * p.bin_tile, end = min(counts[feature], begin + p.bin_tile);
  long2 sum = 0;
  for (uint bin = begin + lane; bin < end; bin += 32)
    sum += load_hist_int(histogram, histogram_begin + bin * 4);
  auto total = histogram_prefix(sum, lane);
  if (lane == 31) sums[index * p.bin_tiles + block.z] = total;
}

kernel void scalar_tile_carries(device const long2* sums [[buffer(0)]],
                                 device const uchar* categorical [[buffer(1)]],
                                 device long2* carries [[buffer(2)]],
                                 constant ScalarTileParams& p [[buffer(3)]],
                                 uint2 block [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  if (p.split.has_categorical && categorical[block.x]) return;
  auto index = ulong(block.y) * p.split.cols + block.x;
  auto source = index * p.bin_tiles, destination = index * (p.bin_tiles + 1);
  long2 preceding = 0;
  for (uint begin = 0; begin < p.bin_tiles; begin += 32) {
    uint tile = begin + lane;
    auto value = tile < p.bin_tiles ? sums[source + tile] : long2(0);
    auto prefix = histogram_prefix(value, lane);
    if (tile < p.bin_tiles) carries[destination + tile] = preceding + prefix - value;
    preceding += histogram_broadcast(prefix, 31);
  }
  if (!lane) carries[destination + p.bin_tiles] = preceding;
}

kernel void evaluate_scalar_tiles(device const uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]],
                                   device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]],
                                   device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]],
                                   device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]],
                                   device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]],
                                   device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]],
                                   device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]],
                                   device const long2* carries [[buffer(14)]], device ScalarTileChoice* tiles [[buffer(15)]],
                                   constant ScalarTileParams& p [[buffer(16)]],
                                   uint3 block [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constant SplitParams& sp = p.split;
  uint feature = block.x, node = block.y, count = counts[feature];
  if (sp.has_categorical && categorical[feature]) {
    if (!block.z && !lane)
      evaluate_feature(histogram, scales, cuts, counts, minima, monotone, bounds, allowed, output,
                       categorical, sorted, ranks, categories, node_sums, sp, uint2(feature, node));
    return;
  }
  auto index = ulong(node) * sp.cols + feature;
  auto offset = index * (sp.bins + 1) * 4, carry = index * (p.bin_tiles + 1);
  uint begin = block.z * p.bin_tile, end = min(count, begin + p.bin_tile);
  auto present = carries[carry + p.bin_tiles];
  auto total_int = sp.use_parent_sums ? node_sums[node] : present + load_hist_int(histogram, offset + sp.bins * 4);
  auto missing = total_int - present;
  float2 scale = scales[sp.group];
  auto total = scalar_histogram_sum(total_int, scale);
  float w = scalar_weight(total, sp, bounds[node]);
  Split best{-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, cox_float_result(total.x), cox_float_result(total.y)};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  if (begin < end && allowed[index]) {
    float parent_gain = scalar_gain(total, w, sp);
    long2 preceding = carries[carry + block.z];
    for (uint first = begin; first < end; first += 32) {
      uint bin = first + lane;
      auto value = bin < end ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto left = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (bin < end) {
        float previous = best.gain;
        consider_split(best, left, total_int - left, scale, cuts[ulong(feature) * sp.bins + bin], 0,
                       feature, monotone[feature], bounds[node], parent_gain, sp);
        if (best.gain > previous) order = bin;
      }
    }
    // The suffix beyond this physical tile is present minus its inclusive carry.
    preceding = present - carries[carry + block.z + 1];
    uint length = end - begin;
    for (uint first = 0; first < ((missing.x != 0 || missing.y != 0) ? length : 0); first += 32) {
      uint step = first + lane, bin = end - min(step + 1, length);
      auto value = step < length ? load_hist_int(histogram, offset + bin * 4) : long2(0);
      auto prefix = histogram_prefix(value, lane);
      auto right = preceding + prefix;
      preceding += histogram_broadcast(prefix, 31);
      if (step < length) {
        float previous = best.gain;
        float threshold = bin == 0 ? minima[feature] : cuts[ulong(feature) * sp.bins + bin - 1];
        consider_split(best, total_int - right, right, scale, threshold, 1,
                       feature, monotone[feature], bounds[node], parent_gain, sp);
        if (best.gain > previous) order = count + (count - 1 - bin);
      }
    }
  }
  float maximum = simd_max(best.gain);
  uint first = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first ? lane : 0xffffffffu);
  if (lane == winner) tiles[index * p.bin_tiles + block.z] = ScalarTileChoice{best, order, 0, 0, 0};
}

kernel void reduce_scalar_tiles(device const ScalarTileChoice* tiles [[buffer(0)]],
                                 device const uchar* categorical [[buffer(1)]],
                                 device Split* output [[buffer(2)]],
                                 constant ScalarTileParams& p [[buffer(3)]],
                                 uint2 block [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  if (p.split.has_categorical && categorical[block.x]) return;
  auto index = ulong(block.y) * p.split.cols + block.x;
  float best_gain = -INFINITY;
  uint best_order = 0xffffffffu, best_tile = 0xffffffffu;
  for (uint tile = lane; tile < p.bin_tiles; tile += 32) {
    auto candidate = index * p.bin_tiles + tile;
    float gain = tiles[candidate].split.gain;
    uint order = tiles[candidate].order;
    if (gain > best_gain || (gain == best_gain && (order < best_order ||
        (order == best_order && tile < best_tile)))) {
      best_gain = gain; best_order = order; best_tile = tile;
    }
  }
  float maximum = simd_max(best_gain);
  uint order = simd_min(best_gain == maximum ? best_order : 0xffffffffu);
  uint tile = simd_min(best_gain == maximum && best_order == order ? best_tile : 0xffffffffu);
  if (!lane) output[index] = tiles[index * p.bin_tiles + tile].split;
}

#include "scalar_root.metal.h"

kernel void select_splits(device const Split* candidates [[buffer(0)]],
                          device Split* best [[buffer(1)]],
                          device const float2* scales [[buffer(2)]],
                          device const float2* bounds [[buffer(3)]],
                          constant SplitParams& p [[buffer(4)]],
                          uint node [[thread_position_in_grid]]) {
  if (node >= p.nodes) return;
  Split result = candidates[ulong(node) * p.cols];
  for (uint f = 1; f < p.cols; ++f) {
    Split other = candidates[ulong(node) * p.cols + f];
    if (other.gain > result.gain) result = other;
  }
  if (result.feature >= 0) {
    auto sums = split_parent_sums(long2(result.left_sum.g, result.left_sum.h),
                                  long2(result.right_sum.g, result.right_sum.h), scales[p.group]);
    result.applied_parent_weight = split_parent_weight(sums, p.reg_alpha, p.reg_lambda,
                                                       p.max_delta_step, bounds[node]);
    result.applied_parent_hessian = cox_float_result(sums.y);
  }
  best[node] = result;
}

inline bool category_member(device const uint* categories, uint begin, uint count, float value) {
  if (fp32_less(value, 0) || value >= 16777216.0f) return false;
  uint category = uint(value), lo = 0, hi = count;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    if (categories[begin + mid] < category) lo = mid + 1; else hi = mid;
  }
  return lo < count && categories[begin + lo] == category;
}
inline bool split_left(float value, float threshold, uint default_left, uint split_type,
                       uint begin, uint count, device const uint* categories) {
  if (isnan(value)) return bool(default_left);
  return split_type ? !category_member(categories, begin, count, value) : fp32_less(value, threshold);
}

#include "vector_tree.metal.h"
#include "refresh.metal.h"

// Segmented scans follow the GPU row-partitioner design in CUDA XGBoost. Each
// threadgroup scans one tile; a second kernel scans its node's tile counts.
kernel void plan_partitions(device const uint* counts [[buffer(0)]],
                            device const uint* offsets [[buffer(1)]],
                            device const int2* children [[buffer(2)]],
                            device PartitionTask* tasks [[buffer(3)]],
                            device PartitionSegment* segments [[buffer(4)]],
                            device atomic_uint* arguments [[buffer(5)]],
                            constant uint& parents [[buffer(6)]],
                            uint node [[thread_position_in_grid]]) {
  if (node >= parents || children[node].x < 0) return;
  uint count = counts[node], tiles = count / 256 + uint(count % 256 != 0);
  uint first = atomic_fetch_add_explicit(arguments, tiles, memory_order_relaxed);
  uint segment = uint(children[node].x) / 2;
  segments[segment] = PartitionSegment{offsets[node], offsets[node] + count, first, first + tiles,
                                      uint(children[node].x), uint(children[node].y)};
  for (uint tile = 0; tile < tiles; ++tile) {
    uint row = tile * 256;
    uint begin = offsets[node] + row, end = begin + min(256u, count - row);
    tasks[first + tile] = PartitionTask{begin, end, node, segment};
  }
}

kernel void partition_segment_flags(device const float* data [[buffer(0)]],
                                    device const uint* rows [[buffer(1)]],
                                    device const Split* splits [[buffer(2)]],
                                    device const PartitionTask* tasks [[buffer(3)]],
                                    device uint* ranks [[buffer(4)]],
                                    device uint* counts [[buffer(5)]],
                                    device const uint* categories [[buffer(6)]],
                                    constant PartitionParams& p [[buffer(7)]],
                                    uint block [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[8];
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  bool left = false;
  if (index < task.end) {
    Split split = splits[task.node];
    float value = data[ulong(rows[index]) * p.cols + uint(split.feature)];
    left = split_left(value, split.threshold, split.default_left, split.split_type,
                      split.category_begin, split.category_count, categories);
  }
  uint prefix = simd_prefix_exclusive_sum(uint(left));
  uint sum = simd_sum(uint(left));
  if ((tid & 31) == 0) sums[tid / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = 0; i < tid / 32; ++i) prefix += sums[i];
  if (index < task.end) ranks[index] = prefix | (left ? 0x80000000u : 0u);
  if (tid == 255) counts[block] = prefix + uint(left);
}
kernel void partition_segment_scan(device const PartitionSegment* segments [[buffer(0)]],
                                   device uint* block_counts [[buffer(1)]],
                                   device uint* counts [[buffer(2)]],
                                   device uint* offsets [[buffer(3)]],
                                   constant PartitionParams& p [[buffer(4)]],
                                   uint id [[thread_position_in_grid]]) {
  if (id >= p.nodes / 2) return;
  PartitionSegment segment = segments[id];
  uint sum = 0;
  for (uint block = segment.task_begin; block < segment.task_end; ++block) {
    uint count = block_counts[block];
    block_counts[block] = sum;
    sum += count;
  }
  counts[segment.left] = sum; counts[segment.right] = segment.end - segment.begin - sum;
  offsets[segment.left] = segment.begin; offsets[segment.right] = segment.begin + sum;
}
kernel void partition_segment_scatter(device const uint* rows [[buffer(0)]],
                                      device const PartitionTask* tasks [[buffer(1)]],
                                      device const PartitionSegment* segments [[buffer(2)]],
                                      device const uint* ranks [[buffer(3)]],
                                      device const uint* block_counts [[buffer(4)]],
                                      device const uint* offsets [[buffer(5)]],
                                      device uint* output [[buffer(6)]],
                                      constant PartitionParams& p [[buffer(7)]],
                                      uint block [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]]) {
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  if (index >= task.end) return;
  PartitionSegment segment = segments[task.segment];
  uint packed = ranks[index];
  uint rank = (packed & 0x7fffffffu) + block_counts[block];
  uint address = (packed >> 31) ? offsets[segment.left] + rank
                               : offsets[segment.right] + index - segment.begin - rank;
  output[address] = rows[index];
}
// Atomics use the same persistent node segments as the stable scan path.
// Each task belongs to one parent, so a SIMD group has at most two child keys.
kernel void partition_atomic_count(device const float* data [[buffer(0)]],
                                   device const uint* rows [[buffer(1)]],
                                   device const Split* splits [[buffer(2)]],
                                   device const PartitionTask* tasks [[buffer(3)]],
                                   device uint* ranks [[buffer(4)]],
                                   device atomic_uint* counts [[buffer(5)]],
                                   device const uint* categories [[buffer(6)]],
                                   constant PartitionParams& p [[buffer(7)]],
                                   uint block [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  bool valid = index < task.end, left = false;
  if (valid) {
    Split split = splits[task.node];
    float value = data[ulong(rows[index]) * p.cols + uint(split.feature)];
    left = split_left(value, split.threshold, split.default_left, split.split_type,
                      split.category_begin, split.category_count, categories);
    ranks[index] = uint(left);
  }
  if (p.padding) {
    uint sum = simd_sum(uint(valid && left));
    if (simd_is_first() && sum) atomic_fetch_add_explicit(counts + 2 * task.segment, sum, memory_order_relaxed);
  } else if (valid && left) atomic_fetch_add_explicit(counts + 2 * task.segment, 1, memory_order_relaxed);
}
kernel void partition_atomic_offsets(device const PartitionSegment* segments [[buffer(0)]],
                                     device uint* counts [[buffer(1)]],
                                     device uint* offsets [[buffer(2)]],
                                     constant PartitionParams& p [[buffer(3)]],
                                     uint id [[thread_position_in_grid]]) {
  if (id >= p.nodes / 2) return;
  PartitionSegment segment = segments[id];
  uint left = counts[segment.left];
  counts[segment.right] = segment.end - segment.begin - left;
  offsets[segment.left] = segment.begin;
  offsets[segment.right] = segment.begin + left;
}
kernel void partition_atomic_scatter(device const uint* rows [[buffer(0)]],
                                     device const PartitionTask* tasks [[buffer(1)]],
                                     device const uint* ranks [[buffer(2)]],
                                     device const uint* offsets [[buffer(3)]],
                                     device atomic_uint* cursors [[buffer(4)]],
                                     device uint* output [[buffer(5)]],
                                     constant PartitionParams& p [[buffer(6)]],
                                     uint block [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  bool valid = index < task.end;
  bool left = valid && ranks[index] != 0;
  if (p.padding) {
    for (uint side = 0; side < 2; ++side) {
      bool match = valid && uint(!left) == side;
      uint count = simd_sum(uint(match)), rank = simd_prefix_exclusive_sum(uint(match));
      uint child = 2 * task.segment + side, begin = 0;
      if (simd_is_first() && count) begin = atomic_fetch_add_explicit(cursors + child, count, memory_order_relaxed);
      begin = simd_broadcast_first(begin);
      if (match) output[offsets[child] + begin + rank] = rows[index];
    }
  } else if (valid) {
    uint child = 2 * task.segment + uint(!left);
    uint begin = atomic_fetch_add_explicit(cursors + child, 1, memory_order_relaxed);
    output[offsets[child] + begin] = rows[index];
  }
}
kernel void partition_segment_copy(device const uint* source [[buffer(0)]],
                                   device const PartitionTask* tasks [[buffer(1)]],
                                   device uint* destination [[buffer(2)]],
                                   constant PartitionParams& p [[buffer(3)]],
                                   uint block [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  if (index < task.end) destination[index] = source[index];
}

#include "tree_driver.metal.h"
#include "tree_driver_fused.metal.h"

struct DenseRow {
  device const float* values;
  float get(uint feature) const { return values[feature]; }
};
struct CachedRow {
  threadgroup const float* values;
  uint stride;
  float get(uint feature) const { return values[feature * stride]; }
};
template<class Row>
inline uint leaf_index(Row row, device const Node* nodes, uint begin,
                       device const uint* categories, device const float* category_bounds = nullptr) {
  uint node = 0;
  while (nodes[begin + node].left >= 0) {
    Node n = nodes[begin + node];
    float value = row.get(uint(n.feature));
    bool numeric_category = n.split_type && category_bounds != nullptr;
    bool left = split_left(value, numeric_category ? category_bounds[n.feature] : n.threshold,
                           n.default_left, numeric_category ? 0 : n.split_type,
                           n.category_begin, n.category_count, categories);
    node = uint(left ? n.left : n.right);
  }
  return node;
}
kernel void predict(device const float* data [[buffer(0)]],
                    device const Node* nodes [[buffer(1)]],
                    device const TreeMeta* trees [[buffer(2)]],
                    device const float* base_score [[buffer(3)]],
                    device const float* base_margin [[buffer(4)]],
                    device float* output [[buffer(5)]],
                    device const uint* categories [[buffer(6)]],
                    device const float* weights [[buffer(7)]],
                    device const float* leaf_weights [[buffer(8)]],
                    device const float* category_bounds [[buffer(9)]],
                    constant PredictParams& p [[buffer(10)]],
                    uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  ulong idx = ulong(id.x) * p.groups + id.y;
  float sum = p.has_margin ? base_margin[ulong(id.x) * p.margin_groups + id.y] : base_score[id.y];
  for (uint t = p.tree_begin; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if ((tree.outputs == 1 && tree.group != id.y) || tree.round < p.begin || tree.round >= p.end) continue;
    uint node = leaf_index(DenseRow{data + ulong(id.x) * p.cols}, nodes, tree.begin, categories,
                           p.bin_partition ? category_bounds : nullptr);
    float leaf = tree.outputs > 1 ? leaf_weights[tree.leaf_begin + ulong(nodes[tree.begin + node].right) * tree.outputs + id.y]
                                 : nodes[tree.begin + node].leaf;
    sum += leaf * (p.has_tree_weights ? weights[t] : 1.0f);
  }
  output[idx] = sum;
}
// Cooperatively load contiguous rows. Use an odd row stride to avoid the
// worst bank conflicts at power-of-two feature counts. All lanes reach the
// barrier, including tail rows.
inline void cache_rows(device const float* data, threadgroup float* cache,
                       uint block, uint lane, uint threads, constant PredictParams& p) {
  ulong start = ulong(block) * threads;
  for (uint i = lane; i < threads * p.cols; i += threads) {
    uint row = i / p.cols, feature = i % p.cols;
    cache[row * (p.cols | 1u) + feature] = start + row < p.rows ? data[(start + row) * p.cols + feature] : NAN;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
kernel void predict_cached(device const float* data [[buffer(0)]],
                           device const Node* nodes [[buffer(1)]],
                           device const TreeMeta* trees [[buffer(2)]],
                           device const float* base_score [[buffer(3)]],
                           device const float* base_margin [[buffer(4)]],
                           device float* output [[buffer(5)]],
                           device const uint* categories [[buffer(6)]],
                           device const float* weights [[buffer(7)]],
                           device const float* leaf_weights [[buffer(8)]],
                           device const float* category_bounds [[buffer(9)]],
                           constant PredictParams& p [[buffer(10)]],
                           threadgroup float* cache [[threadgroup(0)]],
                           uint2 block [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],
                           uint2 shape [[threads_per_threadgroup]]) {
  uint threads = shape.x;
  cache_rows(data, cache, block.x, lane, threads, p);
  uint row = block.x * threads + lane, group = block.y;
  if (row >= p.rows || group >= p.groups) return;
  float sum = p.has_margin ? base_margin[ulong(row) * p.margin_groups + group] : base_score[group];
  CachedRow values{cache + lane * (p.cols | 1u), 1};
  for (uint t = p.tree_begin; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if ((tree.outputs == 1 && tree.group != group) || tree.round < p.begin || tree.round >= p.end) continue;
    uint node = leaf_index(values, nodes, tree.begin, categories, p.bin_partition ? category_bounds : nullptr);
    float leaf = tree.outputs > 1 ? leaf_weights[tree.leaf_begin + ulong(nodes[tree.begin + node].right) * tree.outputs + group]
                                 : nodes[tree.begin + node].leaf;
    sum += leaf * (p.has_tree_weights ? weights[t] : 1.0f);
  }
  output[ulong(row) * p.groups + group] = sum;
}
inline void transform_values(device float* output, constant PredictParams& p, uint row) {
  if (row >= p.rows) return;
  if (p.objective == 3 || p.objective == 4) {
    float max_value = -INFINITY, denom = 0;
    for (uint k = 0; k < p.groups; ++k) max_value = max(max_value, output[ulong(row) * p.groups + k]);
    for (uint k = 0; k < p.groups; ++k) denom += exp(output[ulong(row) * p.groups + k] - max_value);
    for (uint k = 0; k < p.groups; ++k) output[ulong(row) * p.groups + k] = exp(output[ulong(row) * p.groups + k] - max_value) / denom;
  } else if (p.objective == 12) {
    // The reference returns ordered quantiles while preserving raw margin order.
    for (uint i = 1; i < p.groups; ++i) {
      float value = output[ulong(row) * p.groups + i];
      uint position = i;
      while (position && output[ulong(row) * p.groups + position - 1] > value) {
        output[ulong(row) * p.groups + position] = output[ulong(row) * p.groups + position - 1];
        --position;
      }
      output[ulong(row) * p.groups + position] = value;
    }
  } else if (p.objective == 13) {
    float prediction = output[ulong(row) * p.groups];
    for (uint k = 1; k < p.groups; ++k) {
      prediction += 1.0e-6f + softplus(output[ulong(row) * p.groups + k]);
      output[ulong(row) * p.groups + k] = prediction;
    }
  } else for (uint k = 0; k < p.groups; ++k) {
    ulong idx = ulong(row) * p.groups + k;
    float x = output[idx];
    if (p.objective == 1) x = sigmoid(x);
    if (p.objective == 5 || p.objective == 7 || p.objective == 8 || p.objective == 15) x = exp(x);
    if (p.objective == 16) x = cox_platform_expf(x);
    if (p.objective == 10) x = float(x > 0);
    output[idx] = x;
  }
}
kernel void transform(device float* output [[buffer(0)]],
                      constant PredictParams& p [[buffer(1)]],
                      uint row [[thread_position_in_grid]]) {
  transform_values(output, p, row);
}
kernel void transform_copy(device const float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           constant PredictParams& p [[buffer(2)]],
                           uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  for (uint k = 0; k < p.groups; ++k) output[ulong(row) * p.groups + k] = input[ulong(row) * p.groups + k];
  transform_values(output, p, row);
}
kernel void argmax(device const float* input [[buffer(0)]],
                   device float* output [[buffer(1)]],
                   constant PredictParams& p [[buffer(2)]],
                   uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint best = 0;
  for (uint k = 1; k < p.groups; ++k) if (input[ulong(row) * p.groups + k] > input[ulong(row) * p.groups + best]) best = k;
  output[row] = float(best);
}
kernel void predict_leaf(device const float* data [[buffer(0)]],
                         device const Node* nodes [[buffer(1)]],
                         device const TreeMeta* trees [[buffer(2)]],
                         device float* output [[buffer(3)]],
                         device const uint* categories [[buffer(4)]],
                         constant PredictParams& p [[buffer(5)]],
                         uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.trees) return;
  TreeMeta tree = trees[id.y];
  output[ulong(id.x) * p.trees + id.y] = float(leaf_index(DenseRow{data + ulong(id.x) * p.cols}, nodes, tree.begin, categories));
}
kernel void predict_leaf_cached(device const float* data [[buffer(0)]],
                                device const Node* nodes [[buffer(1)]],
                                device const TreeMeta* trees [[buffer(2)]],
                                device float* output [[buffer(3)]],
                                device const uint* categories [[buffer(4)]],
                                constant PredictParams& p [[buffer(5)]],
                                threadgroup float* cache [[threadgroup(0)]],
                                uint block [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint threads [[threads_per_threadgroup]]) {
  cache_rows(data, cache, block, lane, threads, p);
  uint row = block * threads + lane;
  if (row >= p.rows) return;
  CachedRow values{cache + lane * (p.cols | 1u), 1};
  for (uint t = 0; t < p.trees; ++t) {
    output[ulong(row) * p.trees + t] = float(leaf_index(values, nodes, trees[t].begin, categories));
  }
}

// The pinned reference uses the same eight-point endpoint quadrature rule.
// Express each leaf as a product of conditional branch probabilities, merging
// repeated splits into a single interval for each feature.
constant float4 kQuadLow = {0.00039422386907972395f, 0.010336129926145077f,
                            0.056279875338077545f, 0.1666947454214096f};
constant float4 kQuadHigh = {0.3501293957233429f, 0.5818122625350952f,
                             0.807002604007721f, 0.960684061050415f};
constant float4 kWeightLow = {0.0020098998211324215f, 0.022608758881688118f,
                              0.07442181557416916f, 0.14807750284671783f};
constant float4 kWeightHigh = {0.21460627019405365f, 0.23928482830524445f,
                               0.19977226853370667f, 0.09921863675117493f};
inline float path_one(ShapPath path, float value, device const uint* categories) {
  if (isnan(value)) return float(path.missing);
  if (path.category_mode) {
    bool member = category_member(categories, path.category_begin, path.category_count, value);
    return float(path.category_mode == 1 ? member : !member);
  }
  return float(fp32_greater_equal(value, path.lower) && fp32_less(value, path.upper));
}
kernel void shap_contributions(device const float* data [[buffer(0)]],
                                device const ShapPath* paths [[buffer(1)]],
                                device const ShapLeaf* leaves [[buffer(2)]],
                                device const float* base [[buffer(3)]],
                                device const float* margin [[buffer(4)]],
                                device float* output [[buffer(5)]],
                                device const uint* categories [[buffer(6)]],
                                device const float* weights [[buffer(7)]],
                                constant PredictParams& p [[buffer(8)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  bool interactions = p.type == 4;
  ulong columns = ulong(p.cols) + 1;
  ulong elements = interactions ? columns * columns : columns;
  ulong offset = (ulong(id.x) * p.groups + id.y) * elements;
  for (ulong i = 0; i < elements; ++i) output[offset + i] = 0;
  float bias = p.has_margin ? margin[ulong(id.x) * p.groups + id.y] : base[id.y];
  ulong row = ulong(id.x) * p.cols;
  for (uint leaf_id = 0; leaf_id < p.trees; ++leaf_id) {
    ShapLeaf leaf = leaves[leaf_id];
    if (leaf.group != id.y || leaf.round < p.begin || leaf.round >= p.end) continue;
    float weight = p.has_tree_weights ? weights[leaf.tree] : 1.0f;
    bias += leaf.expectation * weight;
    float4 product_lo = leaf.value * kWeightLow, product_hi = leaf.value * kWeightHigh;
    for (uint i = 0; i < leaf.count; ++i) {
      ShapPath term = paths[leaf.begin + i];
      float one = path_one(term, data[row + term.feature], categories);
      product_lo *= term.zero + (one - term.zero) * kQuadLow;
      product_hi *= term.zero + (one - term.zero) * kQuadHigh;
    }
    for (uint i = 0; i < leaf.count; ++i) {
      ShapPath term = paths[leaf.begin + i];
      float difference = path_one(term, data[row + term.feature], categories) - term.zero;
      float4 di_lo = difference / (term.zero + difference * kQuadLow);
      float4 di_hi = difference / (term.zero + difference * kQuadHigh);
      float contribution = dot(product_lo, di_lo) + dot(product_hi, di_hi);
      ulong index = interactions ? ulong(term.feature) * columns + term.feature : term.feature;
      output[offset + index] += contribution * weight;
      if (interactions) for (uint j = 0; j < i; ++j) {
        ShapPath partner = paths[leaf.begin + j];
        float delta = path_one(partner, data[row + partner.feature], categories) - partner.zero;
        float4 dj_lo = delta / (partner.zero + delta * kQuadLow);
        float4 dj_hi = delta / (partner.zero + delta * kQuadHigh);
        float interaction = (0.5f * (dot(product_lo * di_lo, dj_lo) + dot(product_hi * di_hi, dj_hi))) * weight;
        output[offset + ulong(term.feature) * columns + partner.feature] += interaction;
        output[offset + ulong(partner.feature) * columns + term.feature] += interaction;
        output[offset + ulong(term.feature) * columns + term.feature] -= interaction;
        output[offset + ulong(partner.feature) * columns + partner.feature] -= interaction;
      }
    }
  }
  output[offset + elements - 1] = bias;
}
kernel void approximate_contributions(device const float* data [[buffer(0)]],
                                      device const Node* nodes [[buffer(1)]],
                                      device const TreeMeta* trees [[buffer(2)]],
                                      device const float* means [[buffer(3)]],
                                      device const float* base [[buffer(4)]],
                                      device const float* margin [[buffer(5)]],
                                      device float* output [[buffer(6)]],
                                      device const uint* categories [[buffer(7)]],
                                      device const float* weights [[buffer(8)]],
                                      constant PredictParams& p [[buffer(9)]],
                                      uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  // The reference's approximate interaction mode puts Saabas values on the diagonal.
  bool interactions = p.type == 5;
  ulong columns = ulong(p.cols) + 1;
  ulong elements = interactions ? columns * columns : columns;
  ulong offset = (ulong(id.x) * p.groups + id.y) * elements;
  for (ulong i = 0; i < elements; ++i) output[offset + i] = 0;
  float bias = p.has_margin ? margin[ulong(id.x) * p.groups + id.y] : base[id.y];
  for (uint t = 0; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if (tree.group != id.y || tree.round < p.begin || tree.round >= p.end) continue;
    uint index = 0;
    float weight = p.has_tree_weights ? weights[t] : 1.0f;
    bias += means[tree.begin] * weight;
    while (nodes[tree.begin + index].left >= 0) {
      Node node = nodes[tree.begin + index];
      float value = data[ulong(id.x) * p.cols + uint(node.feature)];
      bool left = split_left(value, node.threshold, node.default_left, node.split_type,
                             node.category_begin, node.category_count, categories);
      uint next = uint(left ? node.left : node.right);
      ulong dst = interactions ? ulong(node.feature) * columns + uint(node.feature) : uint(node.feature);
      output[offset + dst] += (means[tree.begin + next] - means[tree.begin + index]) * weight;
      index = next;
    }
  }
  output[offset + elements - 1] = bias;
}

inline float log_gamma_positive(float x) {
  // Lanczos, g=7. The Poisson metric calls this only for x >= 1.
  float z = x - 1;
  float a = 0.99999999999980993f;
  a += 676.5203681218851f / (z + 1);
  a -= 1259.1392167224028f / (z + 2);
  a += 771.32342877765313f / (z + 3);
  a -= 176.61502916214059f / (z + 4);
  a += 12.507343278686905f / (z + 5);
  a -= 0.13857109526572012f / (z + 6);
  a += 9.9843695780195716e-6f / (z + 7);
  a += 1.5056327351493116e-7f / (z + 8);
  float t = z + 7.5f;
  return 0.91893853320467274f + (z + 0.5f) * log(t) - t + log(a);
}
kernel void metric_partial(device const float* predictions [[buffer(0)]],
                           device const float* labels [[buffer(1)]],
                           device const float* weights [[buffer(2)]],
                           device float2* partial [[buffer(3)]],
                           constant MetricParams& p [[buffer(4)]],
                           uint block [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float2 values[256];
  bool multiclass = p.type == 4 || p.type == 5;
  ulong count = ulong(p.rows) * (multiclass ? 1 : p.groups);
  ulong idx = ulong(block) * 256 + tid;
  float2 value = 0;
  if (idx < count) {
    uint row = uint(multiclass ? idx : idx / p.groups);
    uint group = multiclass ? 0 : uint(idx % p.groups);
    float weight = weights[row], loss = 0;
    float label = labels[ulong(row) * p.label_cols + min(group, p.label_cols - 1)];
    float pred = predictions[ulong(row) * p.groups + group];
    switch (p.type) {
      case 0: loss = (pred - label) * (pred - label); break;
      case 1: loss = fabs(pred - label); break;
      case 2: { float delta = log1p_stable(pred) - log1p_stable(label); loss = delta * delta; break; }
      case 3: loss = -label * log(max(pred, 1.0e-16f)) - (1 - label) * log(max(1 - pred, 1.0e-16f)); break;
      case 4: {
        uint best = 0;
        for (uint k = 1; k < p.groups; ++k) if (predictions[ulong(row) * p.groups + k] > predictions[ulong(row) * p.groups + best]) best = k;
        loss = float(best != uint(label)); break;
      }
      case 5: loss = -log(max(predictions[ulong(row) * p.groups + uint(label)], 1.0e-16f)); break;
      // As in the reference EvalError, a fractional label counts partially.
      case 6: loss = pred > p.parameter ? 1 - label : label; break;
      case 7: loss = fabs((pred - label) / label); break;
      case 8: {
        float z = (pred - label) / p.parameter;
        loss = p.parameter * p.parameter * (sqrt(1 + z * z) - 1); break;
      }
      case 9: loss = log_gamma_positive(label + 1) + pred - label * log(max(pred, 1.0e-16f)); break;
      case 10: loss = 2 * (-log(label / pred) + label / pred - 1); break;
      case 11: loss = -label * pow(pred, 1 - p.parameter) / (1 - p.parameter) + pow(pred, 2 - p.parameter) / (2 - p.parameter); break;
      case 12: loss = label / pred + log(pred); break;
    }
    value = float2(weight * loss, weight);
  }
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (tid < delta) values[tid] += values[tid + delta];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) partial[block] = values[0];
}
kernel void metric_reduce(device const float2* partial [[buffer(0)]],
                          device float2* output [[buffer(1)]],
                          constant MetricParams& p [[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float2 values[256];
  ulong count = ulong(p.rows) * ((p.type == 4 || p.type == 5) ? 1 : p.groups);
  uint blocks = uint((count + 255) / 256);
  float2 sum = 0, correction = 0;
  for (uint i = tid; i < blocks; i += 256) {
    float2 y = partial[i] - correction, next = sum + y;
    correction = (next - sum) - y; sum = next;
  }
  values[tid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (tid < delta) values[tid] += values[tid + delta];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) output[0] = values[0];
}
kernel void metric_sort(device const float* predictions [[buffer(0)]],
                        device const float* labels [[buffer(1)]],
                        device const float* weights [[buffer(2)]],
                        device float4* output [[buffer(3)]],
                        constant MetricParams& p [[buffer(4)]],
                        uint2 block [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float4 values[256];
  uint row = block.x * 256 + tid;
  float4 value = float4(-INFINITY, 0, 0, 0);
  if (row < p.rows) {
    float y = labels[ulong(row) * p.label_cols + min(block.y, p.label_cols - 1)];
    if (p.multiclass) y = float(uint(y) == block.y);
    value = float4(predictions[ulong(row) * p.groups + block.y], y * weights[row], (1 - y) * weights[row], 0);
  }
  values[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) for (uint j = k >> 1; j; j >>= 1) {
    float4 a = values[tid], b = values[tid ^ j];
    bool take_max = ((tid & k) == 0) == ((tid & j) == 0);
    float4 next = take_max ? (a.x >= b.x ? a : b) : (a.x <= b.x ? a : b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  output[ulong(block.y) * p.stride + row] = values[tid];
}
kernel void metric_merge(device const float4* source [[buffer(0)]],
                         device float4* dest [[buffer(1)]],
                         constant SortParams& p [[buffer(2)]],
                         uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.stride || id.y >= p.cols) return;
  uint begin = (id.x / (2 * p.width)) * (2 * p.width);
  uint middle = min(begin + p.width, p.stride), end = min(begin + 2 * p.width, p.stride);
  bool left = id.x < middle;
  uint own_begin = left ? begin : middle, other_begin = left ? middle : begin, other_end = left ? end : middle;
  ulong column = ulong(id.y) * p.stride;
  float4 value = source[column + id.x];
  uint lo = other_begin, hi = other_end;
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    float other = source[column + mid].x;
    if (other > value.x || (!left && other == value.x)) lo = mid + 1; else hi = mid;
  }
  dest[column + begin + id.x - own_begin + lo - other_begin] = value;
}
// The reference accumulates (tp, fp) in double. A float sum stops growing once
// a class weight passes 2^24, so carry each sum as an unevaluated float pair
// (high + low), which keeps about 48 significand bits.
inline void auc_accumulate(thread float2& high, thread float2& low, float2 value) {
  float2 sum = high + value, value_part = sum - high, high_part = sum - value_part;
  float2 error = (high - high_part) + (value - value_part) + low;
  high = sum + error;
  low = error - (high - sum);
}
inline float auc_pr_segment(float2 previous, float2 step, float positives) {
  // The mean precision over this segment is r*L + q*(1-L), where r is
  // previous precision, q is this tie group's precision, and
  // L = log1p(u)/u for u = step mass / previous mass. All weights are
  // nonnegative, avoiding cancellation between the reference formula's terms.
  float step_scale = max(step.x, step.y);
  float2 normalized_step = step / step_scale;
  float q = normalized_step.x / (normalized_step.x + normalized_step.y);
  float rise = step.x / positives;
  float previous_scale = max(previous.x, previous.y);
  if (previous_scale == 0) return rise * q;
  float2 normalized_previous = previous / previous_scale;
  float r = normalized_previous.x / (normalized_previous.x + normalized_previous.y);

  // Scale before summing: each class total may be finite even when their
  // sum overflows. Separate scales above preserve each precision ratio.
  float scale = max(previous_scale, step_scale);
  float2 before = previous / scale, increment = step / scale;
  float cumulative = before.x + before.y, mass = increment.x + increment.y;
  float retained, added;
  if (mass <= 0.25f * cumulative) {
    float u = mass / cumulative, z = u / (2 + u);
    // (u - log1p(u))/u, using log1p(u) = 2*atanh(z). Computing this
    // normalized series directly avoids cancellation and squaring tiny u.
    added = z * (1 - z) * (1 + z * (2.0f / 3 + z * (1 + z * (0.8f + z * (1 + z * (6.0f / 7 + z * (1 + z * (8.0f / 9))))))));
    retained = 1 - added;
  } else if (mass <= cumulative) {
    // Here u is bounded by one; evaluate its logarithm directly to avoid
    // subtracting nearly equal logarithms in the reciprocal expression.
    float u = mass / cumulative;
    retained = log1p_stable(u) / u;
    added = 1 - retained;
  } else {
    // Use the reciprocal ratio, bounded by one, so neither u nor the
    // logarithm's intermediates can overflow. v==0 is the limiting value.
    float v = cumulative / mass;
    retained = v == 0 ? 0 : v * (log1p_stable(v) - log(v));
    added = 1 - retained;
  }
  return rise * (r * retained + q * added);
}
kernel void auc_reduce(device const float4* sorted [[buffer(0)]],
                       device float4* output [[buffer(1)]],
                       constant MetricParams& p [[buffer(2)]],
                       uint group [[thread_position_in_grid]]) {
  if (group >= p.groups) return;
  ulong column = ulong(group) * p.stride;
  float2 total_high = 0, total_low = 0;
  for (uint r = 0; r < p.rows; ++r) auc_accumulate(total_high, total_low, sorted[column + r].yz);
  float2 total = total_high + total_low;
  float2 high = 0, low = 0;
  float area = 0, correction = 0;
  for (uint row = 0; row < p.rows;) {
    // Each tie group's increment is summed on its own, so no step subtracts
    // two nearly equal cumulative sums.
    float2 previous = high + low, step_high = 0, step_low = 0;
    uint end = row;
    while (end < p.rows && sorted[column + end].x == sorted[column + row].x) {
      float2 value = sorted[column + end].yz;
      auc_accumulate(high, low, value); auc_accumulate(step_high, step_low, value); ++end;
    }
    float2 step = step_high + step_low;
    float delta = 0;
    if (p.type == 13) delta = step.y * (2 * previous.x + step.x) * 0.5f;
    else if (step.x != 0 && total.x > 0) delta = auc_pr_segment(previous, step, total.x);
    float y = delta - correction, next = area + y;
    correction = (next - area) - y; area = next;
    row = end;
  }
  if (p.type == 13) area /= total.x * total.y;
  output[group] = float4(area, total.x, total.y, 0);
}

kernel void stream_copy(device const float4* input [[buffer(0)]],
                        device float4* output [[buffer(1)]],
                        constant ulong& count [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (ulong(index) < count) output[index] = input[index];
}
