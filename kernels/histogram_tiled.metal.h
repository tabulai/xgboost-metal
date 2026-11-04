// Histogram dataflow variants for scalar GPU-driven trees (at most 256 bins).
// They change only where bin codes are loaded from, how threadgroups find their
// task, how a single owner publishes its slice, and how a verified constant
// Hessian is summed. Every (row, feature) pair still contributes the same exact
// modulo-2^64 integer gradient and Hessian, so the resulting histograms are
// bit-identical to the original kernels.

// Derived uint8 bin copy. Word group g holds features 4g..4g+3, one byte each,
// and `chunk` consecutive word groups are interleaved per row:
// index = (g / chunk * rows + row) * chunk + g % chunk. A grouped task then
// replaces up to four scattered 2-byte column gathers with one 4-byte load.
// Code 255 marks a missing value only when histogram bins <= 255; with 256
// bins it is an ordinary bin. Any code that cannot be represented sets
// `invalid`, and the host keeps using the canonical uint16 storage instead.
kernel void pack_histogram_bins(device const ushort* bins [[buffer(0)]],
                                device uint* packed [[buffer(1)]],
                                device atomic_uint* invalid [[buffer(2)]],
                                constant PackBinsParams& p [[buffer(3)]],
                                uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows) return;
  uint word = 0;
  bool unrepresentable = false;
  for (uint k = 0; k < 4; ++k) {
    uint feature = id.y * 4 + k;
    if (feature >= p.cols) break;
    uint code = load_bin(bins, ulong(feature) * p.rows + id.x, p.bins);
    if (code == 0xffffffffu) {
      unrepresentable = unrepresentable || p.bins > 255;
      code = 255;
    } else {
      unrepresentable = unrepresentable || code > 255 || (code == 255 && p.bins <= 255);
    }
    word |= (code & 255u) << (8 * k);
  }
  if (unrepresentable) atomic_store_explicit(invalid, 1, memory_order_relaxed);
  packed[(ulong(id.y / p.chunk) * p.rows + id.x) * p.chunk + id.y % p.chunk] = word;
}

struct HistogramWork { uint begin, end, node, feature; bool single; };

// Original records name their feature group. Row-tile records name only rows:
// the threadgroup index supplies the group, either group-major (all tiles of a
// group adjacent) or tile-major (all groups of a tile adjacent). The grid then
// holds tiles x groups threadgroups, so the tile count is grid / groups.
inline HistogramWork histogram_work(device const HistTask* tasks, constant HistParams& p,
                                    uint block, uint grid) {
  HistTask task;
  uint feature;
  if (p.tasks & kHistogramTileRecords) {
    uint groups = (p.cols - 1) / p.padding0 + 1;
    uint group, tile;
    if (p.tasks & kHistogramTileMajor) {
      tile = block / groups;
      group = block - tile * groups;
    } else {
      uint tiles = grid / groups;
      group = block / tiles;
      tile = block - group * tiles;
    }
    task = tasks[tile];
    feature = group * p.padding0;
  } else {
    task = tasks[block];
    feature = task.feature;
  }
  return HistogramWork{task.begin, task.end, task.node & ~kHistogramSingleOwnerTag, feature,
                       (task.node & kHistogramSingleOwnerTag) != 0};
}

// Constant quantized Hessians. When every row's quantized Hessian equals the
// first row's (unweighted squared error: rint(1.0f * scale) for each row), a
// bin's Hessian sum is its row count times that constant, modulo 2^64. The
// histogram kernels then add only gradients plus a 32-bit count and multiply
// at flush. Any other value sets `mismatch`, and every task of the tree then
// uses the original 16-byte sums.
kernel void histogram_unit_hessian_check(device const long2* gradients [[buffer(0)]],
                                         device atomic_uint* mismatch [[buffer(1)]],
                                         constant HistParams& p [[buffer(2)]],
                                         uint id [[thread_position_in_grid]]) {
  long unit = gradients[p.group].y;
  bool differs = id < p.rows && gradients[ulong(id) * p.groups + p.group].y != unit;
  if (simd_any(differs) && simd_is_first()) atomic_store_explicit(mismatch, 1, memory_order_relaxed);
}

struct HistogramSlice {
  uint begin, end, feature, features, length, bins_per_feature;
  ulong global_begin;
  bool single;
};

// Accumulate and publish one task, after its threadgroup scratch is cleared.
// - Missing: missing rows are summed privately and reduced per SIMD group.
//   Otherwise (fully dense input) a missing code cannot occur; it would be
//   added directly to the device missing bin, as the original dense kernels do.
// - Present: nonroot missing rows are skipped before their gradient load and
//   the built missing bin stays zero (see histogram_present.metal.h).
// - Unit: scratch holds [g_lo, g_hi] per bin followed by one count per bin;
//   Hessians are materialized as count * unit at flush.
// - A single owner stores every bin, zeros included, instead of adding into a
//   cleared slice: 0 + total is the same modulo-2^64 value. Shared slices skip
//   adding zero pairs, which leaves their sums unchanged.
template <bool Packed, bool Missing, bool Present, bool Unit, bool CertI32, uint MaxFeatures>
inline void histogram_tiled_body(device const ushort* bins, device const uint* row_ids,
                                 device const long2* gradients, device atomic_uint* hist,
                                 constant HistParams& p, HistogramSlice slice, long unit,
                                 uint shift_g, uint shift_h,
                                 uint2 block, uint tid, threadgroup atomic_uint* local) {
  uint features = slice.features, length = slice.length, bins_per_feature = slice.bins_per_feature;
  ulong global_begin = slice.global_begin;
  auto counts = local + (CertI32 ? length : 2 * length);
  // Packed words: hoist the chunk arithmetic, which is uniform per task.
  ulong packed_base = 0, packed_base_hi = 0;
  uint packed_stride = 1, packed_shift = 0;
  uint missing_code = p.bins <= 255 ? 255u : 256u;
  if (Packed) {
    uint chunk = p.tasks >> kHistogramPackedShift;
    uint word_group = slice.feature >> 2;
    packed_base = ulong(word_group / chunk) * p.rows * chunk + word_group % chunk;
    if (MaxFeatures == 8) {
      uint hi_group = word_group + 1;
      packed_base_hi = ulong(hi_group / chunk) * p.rows * chunk + hi_group % chunk;
    }
    packed_stride = chunk;
    packed_shift = 8 * (slice.feature & 3);
  }
  auto words = reinterpret_cast<device const uint*>(bins);
  ulong2 missing[MaxFeatures];
  for (uint f = 0; f < MaxFeatures; ++f) missing[f] = ulong2(0);
  for (uint i = slice.begin + tid; i < slice.end; i += 256) {
    uint row = row_ids[i];
    uint codes[MaxFeatures];
    if (Packed) {
      if (MaxFeatures == 8) {
        uint2 pair;
        if ((packed_stride & 1u) == 0u) {
          pair = *reinterpret_cast<device const uint2*>(words + packed_base + ulong(row) * packed_stride);
        } else {
          pair.x = words[packed_base + ulong(row) * packed_stride];
          pair.y = features > 4 ? words[packed_base_hi + ulong(row) * packed_stride] : 0u;
        }
        for (uint f = 0; f < 4; ++f) {
          uint code = (pair.x >> (8 * f)) & 255u;
          codes[f] = code == missing_code ? 0xffffffffu : code;
        }
        for (uint f = 4; f < MaxFeatures; ++f) {
          uint code = (pair.y >> (8 * (f - 4))) & 255u;
          codes[f] = code == missing_code ? 0xffffffffu : code;
        }
      } else {
        uint word = words[packed_base + ulong(row) * packed_stride] >> packed_shift;
        for (uint f = 0; f < MaxFeatures; ++f) {
          uint code = (word >> (8 * f)) & 255u;
          codes[f] = code == missing_code ? 0xffffffffu : code;
        }
      }
    } else {
      for (uint f = 0; f < MaxFeatures; ++f)
        codes[f] = f < features ? load_bin(bins, ulong(slice.feature + f) * p.rows + row, p.bins) : 0u;
    }
    if (Present) {
      bool any_present = false;
      for (uint f = 0; f < MaxFeatures; ++f)
        any_present = any_present || (f < features && codes[f] != 0xffffffffu);
      if (!any_present) continue;
    }
    auto source = gradients + ulong(row) * p.groups + p.group + block.y;
    long2 value = Unit ? long2(reinterpret_cast<device const long*>(source)[0], unit) : source[0];
    uint sg_bits = CertI32 ? as_type<uint>(int(value.x >> shift_g)) : 0u;
    uint sh_bits = (CertI32 && !Unit) ? as_type<uint>(int(value.y >> shift_h)) : 0u;
    for (uint f = 0; f < MaxFeatures; ++f) {
      if (f >= features) break;
      uint bin = codes[f];
      if (bin == 0xffffffffu) {
        if (Present) continue;
        if (Missing) {
          missing[f] += Unit ? ulong2(as_type<ulong>(value.x), 1) : as_type<ulong2>(value);
        } else {
          ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
          add64(hist + offset, value.x); add64(hist + offset + 2, value.y);
        }
      } else if (CertI32) {
        if (Unit) {
          uint idx = f * p.bins + bin;
          atomic_fetch_add_explicit(local + idx, sg_bits, memory_order_relaxed);
          atomic_fetch_add_explicit(counts + idx, 1u, memory_order_relaxed);
        } else {
          uint offset = (f * p.bins + bin) * 2;
          atomic_fetch_add_explicit(local + offset, sg_bits, memory_order_relaxed);
          atomic_fetch_add_explicit(local + offset + 1, sh_bits, memory_order_relaxed);
        }
      } else if (Unit) {
        add64(local + (f * p.bins + bin) * 2, value.x);
        atomic_fetch_add_explicit(counts + f * p.bins + bin, 1u, memory_order_relaxed);
      } else {
        uint offset = (f * p.bins + bin) * 4;
        add64(local + offset, value.x); add64(local + offset + 2, value.y);
      }
    }
  }
  if (Missing && !Present) {
    uint lane = tid & 31;
    for (uint f = 0; f < MaxFeatures; ++f) {
      if (f >= features) break;
      ulong2 value = histogram_missing_reduce(missing[f], lane);
      if (Unit) value.y *= as_type<ulong>(unit);
      if (lane == 0 && (value.x != 0 || value.y != 0)) {
        ulong offset = global_begin + (f * bins_per_feature + p.bins) * 4;
        add64(hist + offset, as_type<long>(value.x));
        add64(hist + offset + 2, as_type<long>(value.y));
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto slots = reinterpret_cast<device uint4*>(hist);
  for (uint bin = tid; bin < length; bin += 256) {
    uint f = MaxFeatures == 1 ? 0u : bin / p.bins;
    ulong offset = global_begin + (f * bins_per_feature + bin - f * p.bins) * 4;
    long2 value;
    if (CertI32) {
      if (Unit) {
        long sg = long(as_type<int>(atomic_load_explicit(local + bin, memory_order_relaxed)));
        ulong cnt = ulong(atomic_load_explicit(counts + bin, memory_order_relaxed));
        value = long2(as_type<long>(as_type<ulong>(sg) << shift_g),
                      as_type<long>(cnt * as_type<ulong>(unit)));
      } else {
        long sg = long(as_type<int>(atomic_load_explicit(local + bin * 2, memory_order_relaxed)));
        long sh = long(as_type<int>(atomic_load_explicit(local + bin * 2 + 1, memory_order_relaxed)));
        value = long2(as_type<long>(as_type<ulong>(sg) << shift_g),
                      as_type<long>(as_type<ulong>(sh) << shift_h));
      }
    } else if (Unit) {
      value = long2(load64(local + bin * 2),
                    as_type<long>(ulong(atomic_load_explicit(counts + bin, memory_order_relaxed)) * as_type<ulong>(unit)));
    } else {
      value = long2(load64(local + bin * 4), load64(local + bin * 4 + 2));
    }
    if (slice.single) {
      slots[offset / 4] = as_type<uint4>(value);
    } else if (value.x != 0 || value.y != 0) {
      add64(hist + offset, value.x);
      add64(hist + offset + 2, value.y);
    }
  }
}

// One threadgroup builds up to MaxFeatures features of one row range, like
// histogram_grouped(_missing/_present). A single-owner task is the only task
// of its (node, feature group) slice, whose fill was skipped. Its missing bin
// is cleared before the device barrier that precedes any missing-bin atomic,
// so missing sums still add into zero. Unit kernels select count-based
// Hessians only when the tree's check found no mismatch.
template <bool Packed, bool Missing, bool Present, bool Unit, uint LocalWords, uint MaxFeatures>
inline void histogram_tiled(device const ushort* bins, device const uint* row_ids,
                            device const long2* gradients, device const float2* scales,
                            device const HistTask* tasks,
                            device atomic_uint* hist, device const uint* unit_mismatch,
                            constant HistParams& p, uint2 block, uint2 grid, uint tid,
                            threadgroup atomic_uint* local) {
  HistogramWork work = histogram_work(tasks, p, block.x, grid.x);
  HistogramSlice slice;
  slice.begin = work.begin;
  slice.end = work.end;
  slice.feature = work.feature;
  slice.features = min(min(p.padding0, MaxFeatures), p.cols - work.feature);
  slice.bins_per_feature = p.bins + 1;
  slice.length = slice.features * p.bins;
  slice.global_begin = ((ulong(work.node) * max(p.padding1, 1u) + block.y) * p.cols + work.feature) *
                       slice.bins_per_feature * 4;
  slice.single = work.single;
  bool unit_active = Unit && !unit_mismatch[0];
  long unit = unit_active ? gradients[p.group + block.y].y : 0;
  uint shift_g = 0, shift_h = 0;
  bool cert_i32 = false;
  if (p.tasks & kHistogramCertI32) {
    GradCert c = reinterpret_cast<device const GradCert*>(scales + p.groups)[p.group + block.y];
    ulong or_g = (ulong(c.or_g_hi) << 32) | ulong(c.or_g_lo);
    ulong max_g = (ulong(c.max_g_hi) << 32) | ulong(c.max_g_lo);
    shift_g = or_g ? uint(63 - clz(or_g & (0ul - or_g))) : 0u;
    ulong smax_g = max_g >> shift_g;
    ulong rows_in_task = ulong(slice.end - slice.begin);
    bool fit_g = smax_g <= 0x7ffffffful && smax_g * rows_in_task <= 0x7ffffffful;
    if (unit_active) {
      cert_i32 = fit_g;
    } else {
      ulong or_h = (ulong(c.or_h_hi) << 32) | ulong(c.or_h_lo);
      ulong max_h = (ulong(c.max_h_hi) << 32) | ulong(c.max_h_lo);
      shift_h = or_h ? uint(63 - clz(or_h & (0ul - or_h))) : 0u;
      ulong smax_h = max_h >> shift_h;
      bool fit_h = smax_h <= 0x7ffffffful && smax_h * rows_in_task <= 0x7ffffffful;
      cert_i32 = fit_g && fit_h;
    }
  }
  uint words_per_bin = cert_i32 ? 2u : (unit_active ? 3u : 4u);
  if (MaxFeatures == 8 && slice.features > 4 && slice.length * words_per_bin > LocalWords) {
    constexpr uint SubMax = MaxFeatures == 8 ? 4u : MaxFeatures;
    uint total_features = slice.features;
    for (uint sub = 0; sub < 2; ++sub) {
      uint sub_offset = sub * 4;
      if (sub_offset >= total_features) break;
      HistogramSlice sub_slice = slice;
      sub_slice.feature = slice.feature + sub_offset;
      sub_slice.features = min(4u, total_features - sub_offset);
      sub_slice.length = sub_slice.features * p.bins;
      sub_slice.global_begin = slice.global_begin + ulong(sub_offset) * slice.bins_per_feature * 4;
      if (sub == 0 && work.single && tid < total_features)
        reinterpret_cast<device uint4*>(hist)[slice.global_begin / 4 + tid * slice.bins_per_feature + p.bins] = uint4(0);
      if (sub > 0) threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < sub_slice.length * words_per_bin && i < LocalWords; i += 256)
        atomic_store_explicit(local + i, 0, memory_order_relaxed);
      if (sub == 0 && work.single) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
      else threadgroup_barrier(mem_flags::mem_threadgroup);
      if (unit_active) {
        histogram_tiled_body<Packed, Missing, Present, true, false, SubMax>(
            bins, row_ids, gradients, hist, p, sub_slice, unit, 0, 0, block, tid, local);
      } else {
        histogram_tiled_body<Packed, Missing, Present, false, false, SubMax>(
            bins, row_ids, gradients, hist, p, sub_slice, 0, 0, 0, block, tid, local);
      }
    }
    return;
  }
  if (work.single && tid < slice.features)
    reinterpret_cast<device uint4*>(hist)[slice.global_begin / 4 + tid * slice.bins_per_feature + p.bins] = uint4(0);
  for (uint i = tid; i < slice.length * words_per_bin && i < LocalWords; i += 256)
    atomic_store_explicit(local + i, 0, memory_order_relaxed);
  if (work.single) threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  else threadgroup_barrier(mem_flags::mem_threadgroup);
  if (unit_active) {
    if (cert_i32)
      histogram_tiled_body<Packed, Missing, Present, true, true, MaxFeatures>(
          bins, row_ids, gradients, hist, p, slice, unit, shift_g, 0, block, tid, local);
    else
      histogram_tiled_body<Packed, Missing, Present, true, false, MaxFeatures>(
          bins, row_ids, gradients, hist, p, slice, unit, 0, 0, block, tid, local);
  } else {
    if (cert_i32)
      histogram_tiled_body<Packed, Missing, Present, false, true, MaxFeatures>(
          bins, row_ids, gradients, hist, p, slice, 0, shift_g, shift_h, block, tid, local);
    else
      histogram_tiled_body<Packed, Missing, Present, false, false, MaxFeatures>(
          bins, row_ids, gradients, hist, p, slice, 0, 0, 0, block, tid, local);
  }
}

#define HISTOGRAM_TILED_ARGUMENTS \
    device const ushort* bins [[buffer(0)]], device const uint* row_ids [[buffer(1)]], \
    device const long2* gradients [[buffer(2)]], device const float2* scales [[buffer(3)]], \
    device const HistTask* tasks [[buffer(4)]], device atomic_uint* hist [[buffer(5)]], \
    device const uint* unit_mismatch [[buffer(6)]], constant HistParams& p [[buffer(7)]], \
    uint2 block [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]], \
    uint tid [[thread_index_in_threadgroup]]

// Scratch suffixes follow metal_histogram_compact_scratch: c1/c2 are the
// compact one/two-feature sizes, w1/w4/w8 the single and grouped sizes.
#define HISTOGRAM_TILED_KERNEL(name, packed, missing, present, unit, words, features) \
  kernel void name(HISTOGRAM_TILED_ARGUMENTS) { \
    threadgroup atomic_uint local[words]; \
    histogram_tiled<packed, missing, present, unit, words, features>( \
        bins, row_ids, gradients, scales, tasks, hist, unit_mismatch, p, block, grid, tid, local); \
  }
#define HISTOGRAM_TILED_SOURCES(name, missing, present, words, features) \
  HISTOGRAM_TILED_KERNEL(name, false, missing, present, false, words, features) \
  HISTOGRAM_TILED_KERNEL(name##_packed, true, missing, present, false, words, features) \
  HISTOGRAM_TILED_KERNEL(name##_unit, false, missing, present, true, words, features) \
  HISTOGRAM_TILED_KERNEL(name##_packed_unit, true, missing, present, true, words, features)

HISTOGRAM_TILED_SOURCES(histogram_tiled_dense_c1, false, false, 4 * 256, 1)
HISTOGRAM_TILED_SOURCES(histogram_tiled_dense_c2, false, false, 4 * 512, 2)
HISTOGRAM_TILED_SOURCES(histogram_tiled_dense_w1, false, false, 4 * 513, 1)
HISTOGRAM_TILED_SOURCES(histogram_tiled_dense_w4, false, false, 4 * 1024, 4)
HISTOGRAM_TILED_SOURCES(histogram_tiled_dense_w8, false, false, 4 * 1024, 8)
HISTOGRAM_TILED_SOURCES(histogram_tiled_missing_c1, true, false, 4 * 256, 1)
HISTOGRAM_TILED_SOURCES(histogram_tiled_missing_c2, true, false, 4 * 512, 2)
HISTOGRAM_TILED_SOURCES(histogram_tiled_missing_w1, true, false, 4 * 513, 1)
HISTOGRAM_TILED_SOURCES(histogram_tiled_missing_w4, true, false, 4 * 1024, 4)
HISTOGRAM_TILED_SOURCES(histogram_tiled_missing_w8, true, false, 4 * 1024, 8)
HISTOGRAM_TILED_SOURCES(histogram_tiled_present_w1, true, true, 4 * 513, 1)
HISTOGRAM_TILED_SOURCES(histogram_tiled_present_w4, true, true, 4 * 1024, 4)
HISTOGRAM_TILED_SOURCES(histogram_tiled_present_w8, true, true, 4 * 1024, 8)
#undef HISTOGRAM_TILED_SOURCES
#undef HISTOGRAM_TILED_KERNEL
#undef HISTOGRAM_TILED_ARGUMENTS
