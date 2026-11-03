// Certified float32 screening of scalar numeric split candidates. Each lane
// encloses the exact float32 gain that the unchanged software scorer returns
// for its forward/backward candidates. Only candidates whose upper bound
// reaches the best certified lower bound of the same node, or that cannot be
// classified, run that exact scorer. A skipped candidate's exact gain is
// strictly below another candidate's, so it can neither win nor tie. With a
// per-feature cutoff the Split written here is byte-identical to the
// unscreened kernel; the node cutoff below relaxes this only for features
// that cannot reach the node maximum.
//
// Native float32 arithmetic below only forms bounds. Every result passes
// through an outward step before further use, which keeps each bound valid for
// correctly rounded RTE or RTZ arithmetic and prevents contraction. Magnitude
// guards keep operands finite and normal; values near zero widen to a normal
// absolute floor, so flushed subnormal results cannot shrink an interval.
constant float kScreenFloor = 0x1p-100f;
enum : uint { kScreenInvalid = 0, kScreenValid = 1, kScreenUncertain = 2 };
enum : uint { kScreenSubtracted = 0, kScreenTerminal = 1, kScreenRoot = 2 };

inline float screen_down(float value, uint steps = 1) {
  uint bits = as_type<uint>(value), magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x0d800000u) return -kScreenFloor;  // |value| < 2^-100
  if (magnitude > 0x7f800000u || bits == 0xff800000u) return value;  // NaN, -inf
  return as_type<float>((bits >> 31) ? bits + steps : bits - steps);
}
inline float screen_up(float value, uint steps = 1) {
  uint bits = as_type<uint>(value), magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x0d800000u) return kScreenFloor;
  if (magnitude > 0x7f800000u || bits == 0x7f800000u) return value;  // NaN, +inf
  return as_type<float>((bits >> 31) ? bits - steps : bits + steps);
}

// One-sided steps for values known to be positive: zero becomes NaN going
// down and +inf becomes NaN going up, so range checks reject them.
inline float screen_down_positive(float value, uint steps) { return as_type<float>(as_type<uint>(value) - steps); }
inline float screen_up_positive(float value, uint steps) { return as_type<float>(as_type<uint>(value) + steps); }

// Metal comparisons may flush subnormal gains. A skipped candidate must still
// compare strictly below the certified one, so a cutoff never lies inside the
// open subnormal/zero band; values there drop to -2^-126 (still a lower bound).
inline float screen_safe_cutoff(float value) {
  uint magnitude = as_type<uint>(value) & 0x7fffffffu;
  return magnitude < 0x00800000u ? -0x1p-126f : value;
}
inline float screen_decode_key(uint key) {
  // Inverse of fp32_order_key for finite values; zero is the cleared state.
  if (!key) return -INFINITY;
  return as_type<float>((key & 0x80000000u) ? key ^ 0x80000000u : ~key);
}

// Enclose the scorer's binary64 statistic RN64(value) * 2^-E between floats.
// The leading 24 bits of |value| truncate to an exact float32 significand;
// RN64 cannot leave that 24-bit cell, whose ends are representable. E is the
// scorer's own scale decoding. Magnitudes outside [2^-60, 2^61) are refused.
inline bool screen_statistic(long value, float scale, thread float& lo, thread float& hi) {
  if (!value) { lo = hi = 0; return true; }
  ulong bits = as_type<ulong>(value);
  ulong magnitude = value < 0 ? 0ul - bits : bits;
  uint top = 63 - uint(clz(magnitude));
  uint significand = top > 23 ? uint(magnitude >> (top - 23)) : uint(magnitude << (23 - top));
  bool inexact = top > 23 && (magnitude & ((1ul << (top - 23)) - 1));
  uint scale_bits = as_type<uint>(scale), power = (scale_bits >> 23) & 255;
  int exponent = power ? int(power) - 127 : 31 - int(clz(scale_bits & 0x7fffffu)) - 149;
  int biased = int(top) - exponent + 127;
  if (biased < 67 || biased > 187) return false;
  uint low = (uint(biased) << 23) | (significand & 0x7fffffu);
  float a = as_type<float>(low), b = as_type<float>(low + uint(inexact));
  lo = value < 0 ? -b : a;
  hi = value < 0 ? -a : b;
  return true;
}

// Enclose the float32 gain RN32(-RN64(2Gw + RN64(D * RN32(w^2)))) of
// statistics with positive H in [h_lo, h_hi], where w = RN32(RN64(-G / D)),
// D = RN64(H + lambda), for zero L1 and no clamps. With X = G^2 / D and the
// relative rounding errors d1 (w, <= 2^-24 + 2^-53), d2 (square, 2^-24) and
// d3, d4 (binary64), the bracket equals X(1 - d1^2 - e(1 + d1)^2), where
// (1 + e) = (1 + d2)(1 + d3): the first-order error of w cancels at the
// stationary point. With the final rounding, the gain lies in
// X(1 +- 2^-22), twice the 2^-23 worst case. Guards keep w, its square and
// the gain normal, which these relative bounds require. A zero G gives +-0.
inline bool screen_gain(long gradient, float scale, float h_lo, float h_hi,
                        constant SplitParams& p, thread float& lo, thread float& hi) {
  if (!gradient) { lo = hi = 0; return true; }
  float g_lo, g_hi;
  if (!screen_statistic(gradient, scale, g_lo, g_hi)) return false;
  float small = gradient < 0 ? -g_hi : g_lo, large = gradient < 0 ? -g_lo : g_hi;
  float d_lo = screen_down(h_lo + p.reg_lambda), d_hi = screen_up(h_hi + p.reg_lambda);
  if (!(d_lo >= 0x1p-62f && d_hi <= 0x1p62f)) return false;
  // Positive normal operands: small/d_hi > 2^(e(small) - e(d_hi) - 1) and
  // large/d_lo < 2^(e(large) + 1 - e(d_lo)) keep |w| inside (2^-62, 2^62).
  int w_min = int(as_type<uint>(small) >> 23) - int(as_type<uint>(d_hi) >> 23) - 1;
  int w_max = int(as_type<uint>(large) >> 23) - int(as_type<uint>(d_lo) >> 23) + 1;
  if (w_min < -62 || w_max > 62) return false;
  // The remaining operands and results are positive. A one-sided bit step is
  // then outward for finite results; an underflowed or overflowed quotient
  // becomes NaN or out of range and fails the checks below. Wider steps on
  // quotients tolerate a non-correctly rounded native division.
  float x_lo = screen_down_positive(screen_down_positive(small * small, 1) / d_hi, 4);
  float x_hi = screen_up_positive(screen_up_positive(large * large, 1) / d_lo, 4);
  if (!(x_lo >= 0x1p-100f && x_hi <= 0x1p100f)) return false;
  lo = screen_down_positive(x_lo * (1 - 0x1p-22f), 1);
  hi = screen_up_positive(x_hi * (1 + 0x1p-22f), 1);
  return true;
}

// Classify one child: kScreenInvalid when the exact scorer certainly rejects
// its Hessian, kScreenUncertain when that or its gain bound is undecided.
inline uint screen_child(long2 sums, float2 scale, constant SplitParams& p,
                         thread float& lo, thread float& hi) {
  if (sums.y <= 0) return kScreenInvalid;
  float h_lo, h_hi;
  if (!screen_statistic(sums.y, scale.y, h_lo, h_hi)) return kScreenUncertain;
  // The scorer compares binary64 H against abs(min_child_weight).
  float minimum = fabs(p.min_child_weight);
  if (h_hi < minimum) return kScreenInvalid;
  if (!(h_lo >= minimum)) return kScreenUncertain;
  return screen_gain(sums.x, scale.x, h_lo, h_hi, p, lo, hi) ? kScreenValid : kScreenUncertain;
}

// The parent gain is the same expression for the node total, without the
// min_child_weight test. A nonpositive Hessian gives exactly zero.
inline bool screen_parent_gain(long2 sums, float2 scale, constant SplitParams& p,
                               thread float& lo, thread float& hi) {
  if (sums.y <= 0) { lo = hi = 0; return true; }
  float h_lo, h_hi;
  return screen_statistic(sums.y, scale.y, h_lo, h_hi) &&
         screen_gain(sums.x, scale.x, h_lo, h_hi, p, lo, hi);
}

// A certainly invalid child makes the exact scorer return before any write,
// whatever the other child is. Otherwise enclose RN32(RN64(RN32(RN64(lg + rg))
// - parent_gain)), using monotonicity of each rounding.
inline uint screen_candidate(long2 left, long2 right, float2 scale, float parent_lo,
                             float parent_hi, constant SplitParams& p,
                             thread float& lo, thread float& hi) {
  float a_lo, a_hi, b_lo, b_hi;
  uint a = screen_child(left, scale, p, a_lo, a_hi);
  if (a == kScreenInvalid) return kScreenInvalid;
  uint b = screen_child(right, scale, p, b_lo, b_hi);
  if (b == kScreenInvalid) return kScreenInvalid;
  if (a != kScreenValid || b != kScreenValid) return kScreenUncertain;
  float sum_lo = screen_down(a_lo + b_lo), sum_hi = screen_up(a_hi + b_hi);
  lo = screen_down(sum_lo - parent_hi);
  hi = screen_up(sum_hi - parent_lo);
  return kScreenValid;
}

// Scorer 0 uses generic statistics, 1 fast statistics, 2 the fused score.
// Each matches the parent block and consider_split_* of its unscreened twin.
template <uint Scorer>
inline void screen_parent(long2 total_int, float2 scale, constant SplitParams& p,
                          float2 bounds, threadgroup uint* parent) {
  auto total = Scorer ? scalar_histogram_sum_fast(total_int, scale) : scalar_histogram_sum(total_int, scale);
  float2 score;
  if (Scorer == 2) score = scalar_weight_gain_fused(total, p, bounds);
  else {
    score.x = scalar_weight(total, p, bounds);
    score.y = scalar_gain_fast(total, score.x, p);
  }
  parent[0] = as_type<uint>(score.x);
  parent[1] = as_type<uint>(cox_float_result(total.x));
  parent[2] = as_type<uint>(cox_float_result(total.y));
  parent[3] = as_type<uint>(score.y);
}

// The gain half of scalar_weight_gain_fused for a given (unclamped) weight:
// the same curvature, square, products, signed-zero rule and rounding.
inline float screen_fused_gain(ulong2 sums, float weight, constant SplitParams& p) {
  bool positive_hessian = !(sums.y & kCoxSign) && (sums.y & ~kCoxSign);
  if (!positive_hessian) return 0;
  ulong curvature = cox_add_bits(sums.y, cox_float_bits(p.reg_lambda));
  auto w = cox_float_bits(weight);
  auto square = scalar_square_fast_bits(weight, w);
  auto quadratic = cox_multiply_bits(curvature, square);
  auto sum = cox_fma_bits(cox_add_bits(sums.x, sums.x), w, quadratic);
  if (as_type<uint>(p.reg_alpha) == 0 &&
      (as_type<uint>(weight) & 0x7f800000u) != 0x7f800000u &&
      ((sum >> 52) & 2047) != 2047) {
    if (!(sum & ~kCoxSign)) sum = 0;
  } else {
    auto alpha = cox_float_bits(p.reg_alpha);
    sum = cox_fma_bits(cox_add_bits(alpha, alpha), w & ~kCoxSign, sum);
  }
  return cox_float_result(sum ^ kCoxSign);
}

// A GPU-driver child's node sums are its parent's selected left/right sums,
// and its unclamped weight and cover are that split's left/right weight and
// Hessian, computed by the same scorer family. Only the gain (no division)
// and the gradient total are recomputed: {weight, total_g, total_h, gain}.
template <uint Scorer>
inline float4 screen_inherited_parent(long2 total_int, float2 scale, constant SplitParams& p,
                                      device const Split* source, bool right) {
  auto total = Scorer ? scalar_histogram_sum_fast(total_int, scale) : scalar_histogram_sum(total_int, scale);
  float w = right ? source->right_weight : source->left_weight;
  float h = right ? source->right_h : source->left_h;
  float gain = Scorer == 2 ? screen_fused_gain(total, w, p) : scalar_gain_fast(total, w, p);
  return float4(w, cox_float_result(total.x), h, gain);
}

template <uint Scorer>
inline void screen_exact(thread Split& best, long2 left, long2 right, float2 scale, float threshold,
                         uint default_left, uint feature, int monotone, float2 bounds,
                         float parent_gain, constant SplitParams& p) {
  if (Scorer == 2)
    consider_split_score_fused(best, left, right, scale, threshold, default_left, feature,
                               monotone, bounds, parent_gain, p);
  else if (Scorer == 1)
    consider_split_stats_fast<false, true>(best, left, right, scale, threshold, default_left,
                                           feature, monotone, bounds, parent_gain, p);
  else
    consider_split_gain<false>(best, left, right, scale, threshold, default_left, feature,
                               monotone, bounds, parent_gain, p);
}

// One 256-thread group per (node, feature); thread tid owns bin tid. The
// candidates are the skip-empty set of the sparse/compact kernels: forward at
// the first or a nonzero bin, backward (with missing statistics) at the last
// or a nonzero bin. Omitted zero bins duplicate an adjacent candidate's
// integer sums, with a later order, so they never win.
//
// Screening uses a float enclosure of the parent gain, so the exact parent
// statistics are computed only when exact work or the default record needs
// them. With the node cutoff, a group without survivors that has seen a
// certified bound writes a losing record instead: some candidate of this
// node then has a finite gain above every candidate of this feature, and
// split selection (select_splits or select_splits_simd) keeps only the first
// strictly greater feature gain.
template <uint Scorer, uint Mode>
inline void evaluate_screened_feature(
    device uint* histogram, device const float2* scales,
    device const float* cuts, device const uint* counts, device const float* minima,
    device const int* monotone, device const float2* bounds, device const uchar* allowed,
    device Split* output, device const long2* node_sums, constant SplitParams& p,
    device const uint* parent_histogram, device const HistSubtract* subtraction,
    device atomic_uint* node_cut, device atomic_uint* statistics, device const Split* inherited,
    constant ScreenParams& sp,
    uint2 id, uint tid, threadgroup long2* group_sums, threadgroup long2* prefixes,
    threadgroup float* group_lower, threadgroup uint* bounded, threadgroup ushort* survivors,
    threadgroup atomic_uint* survivor_count, threadgroup uint* choices, threadgroup uint* parent) {
  if (id.x >= p.cols || id.y >= p.nodes) return;
  uint lane = tid & 31, group = tid >> 5;
  uint feature = id.x, node = id.y, count = counts[feature];
  long2 derived_value(0);
  bool derived = false;
  if (Mode != kScreenRoot) {
    // Identical fused sibling subtraction to the unscreened kernels.
    HistSubtract task = subtraction[id.y / 2];
    derived = id.y == task.other;
    if (derived) {
      ulong entries = ulong(p.cols) * (p.bins + 1);
      ulong feature_offset = ulong(id.x) * (p.bins + 1);
      ulong parent_offset = (ulong(task.parent) * entries + feature_offset) * 4;
      ulong built_offset = (ulong(task.built) * entries + feature_offset) * 4;
      ulong output_offset = (ulong(task.other) * entries + feature_offset) * 4;
      if (Mode == kScreenSubtracted) {
        for (uint bin = tid; bin <= p.bins; bin += 256) {
          long2 value = load_hist_int(parent_histogram, parent_offset + ulong(bin) * 4) -
                        load_hist_int(histogram, built_offset + ulong(bin) * 4);
          ulong g = as_type<ulong>(value.x), h = as_type<ulong>(value.y);
          ulong offset = output_offset + ulong(bin) * 4;
          histogram[offset] = uint(g); histogram[offset + 1] = uint(g >> 32);
          histogram[offset + 2] = uint(h); histogram[offset + 3] = uint(h >> 32);
        }
        threadgroup_barrier(mem_flags::mem_device);
      } else if (tid < count) {
        derived_value = load_hist_int(parent_histogram, parent_offset + ulong(tid) * 4) -
                        load_hist_int(histogram, built_offset + ulong(tid) * 4);
      }
    }
  }
  ulong index = ulong(node) * p.cols + feature;
  ulong offset = index * (p.bins + 1) * 4;
  long2 value = tid < count ?
      (Mode == kScreenTerminal && derived ? derived_value :
       load_hist_int(histogram, offset + ulong(tid) * 4)) : long2(0);
  long2 prefix = histogram_prefix(value, lane);
  if (lane == 31) group_sums[group] = prefix;
  if (tid == 0) atomic_store_explicit(survivor_count, 0u, memory_order_relaxed);
  float2 scale = scales[p.group];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  long2 preceding = 0, present = 0;
  for (uint g = 0; g < 8; ++g) {
    long2 sum = group_sums[g];
    present += sum;
    if (g < group) preceding += sum;
  }
  long2 total_int = Mode == kScreenRoot ? present + load_hist_int(histogram, offset + ulong(p.bins) * 4) :
                                          node_sums[node];
  long2 left = preceding + prefix;
  long2 missing = total_int - present;
  if (tid < count) prefixes[tid] = left;
  bool open = tid < count && allowed[index];
  bool nonzero = value.x != 0 || value.y != 0;
  bool forward = open && (nonzero || tid == 0);
  bool backward = open && (missing.x != 0 || missing.y != 0) && (nonzero || tid + 1 == count);
  // Uniform eligibility; anything else treats every candidate as uncertain.
  // Bit tests keep subnormal L1/limits (which comparisons may flush) out.
  float2 node_bounds = bounds[node];
  float parent_lo = 0, parent_hi = 0;
  bool screen = !(as_type<uint>(p.reg_alpha) & 0x7fffffffu) &&
                !(as_type<uint>(p.max_delta_step) & 0x7fffffffu) && isfinite(p.reg_lambda) &&
                monotone[feature] == 0 && as_type<uint>(node_bounds.x) == 0xff800000u &&
                as_type<uint>(node_bounds.y) == 0x7f800000u &&
                screen_parent_gain(total_int, scale, p, parent_lo, parent_hi);
  uint forward_class = kScreenInvalid, backward_class = kScreenInvalid;
  float forward_hi = 0, backward_hi = 0, lower = -INFINITY;
  if (forward) {
    float lo = 0, hi = 0;
    forward_class = screen ? screen_candidate(left, total_int - left, scale, parent_lo, parent_hi, p, lo, hi) :
                             kScreenUncertain;
    if (forward_class == kScreenValid) { forward_hi = hi; lower = lo; }
  }
  if (backward) {
    float lo = 0, hi = 0;
    long2 right = present - (left - value);
    backward_class = screen ? screen_candidate(total_int - right, right, scale, parent_lo, parent_hi, p, lo, hi) :
                              kScreenUncertain;
    if (backward_class == kScreenValid) { backward_hi = hi; lower = max(lower, lo); }
  }
  float simd_lower = simd_max(lower);
  bool node_cut_enabled = Mode != kScreenRoot && sp.node_cut;
  device atomic_uint* cell = node_cut_enabled ? node_cut + sp.cut_offset + node : node_cut;
  if (node_cut_enabled && lane == 0 && simd_lower > -INFINITY)
    atomic_fetch_max_explicit(cell, fp32_order_key(screen_safe_cutoff(simd_lower)), memory_order_relaxed);
  if (lane == 0) group_lower[group] = simd_lower;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float local_cutoff = group_lower[0];
  for (uint g = 1; g < 8; ++g) local_cutoff = max(local_cutoff, group_lower[g]);
  // Every stored key is a safe lower bound certified by some candidate of
  // this node; any value read, stale or not, remains valid. Reading it after
  // this group's own bound gives the node's other features time to publish.
  // SIMD groups may read different values, so the cutoff is not uniform.
  float cutoff = local_cutoff;
  if (node_cut_enabled) {
    uint key = 0;
    if (lane == 0) key = atomic_load_explicit(cell, memory_order_relaxed);
    cutoff = max(cutoff, screen_decode_key(simd_broadcast(key, 0)));
  }
  cutoff = cutoff > -INFINITY ? screen_safe_cutoff(cutoff) : cutoff;
  if (lane == 0) bounded[group] = uint(cutoff > -INFINITY);
  bool forward_survives = forward_class == kScreenUncertain ||
                          (forward_class == kScreenValid && forward_hi >= cutoff);
  bool backward_survives = backward_class == kScreenUncertain ||
                           (backward_class == kScreenValid && backward_hi >= cutoff);
  uint mine = uint(forward_survives) + uint(backward_survives);
  uint rank = simd_prefix_exclusive_sum(mine);
  uint simd_total = simd_sum(mine);
  uint base = 0;
  if (lane == 0 && simd_total)
    base = atomic_fetch_add_explicit(survivor_count, simd_total, memory_order_relaxed);
  base = simd_broadcast(base, 0) + rank;
  if (forward_survives && base < 256) survivors[base] = ushort(tid);
  if (backward_survives && base + uint(forward_survives) < 256)
    survivors[base + uint(forward_survives)] = ushort(tid | 0x100u);
  if (sp.statistics) {
    uint4 local(uint(forward) + uint(backward),
                uint(forward_class == kScreenValid) + uint(backward_class == kScreenValid),
                uint(forward_class == kScreenUncertain) + uint(backward_class == kScreenUncertain), mine);
    uint4 totals = simd_sum(local);
    if (lane == 0)
      for (uint k = 0; k < 4; ++k) atomic_fetch_add_explicit(statistics + k, totals[k], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint survivor_total = atomic_load_explicit(survivor_count, memory_order_relaxed);
  if (sp.statistics && tid == 0) {
    atomic_fetch_add_explicit(statistics + 4, 1u, memory_order_relaxed);
    if (survivor_total) atomic_fetch_add_explicit(statistics + 5, 1u, memory_order_relaxed);
  }
  // Uniform. Without the node cutoff a certified candidate always survives
  // its own feature's bound, so only node-cut features take this branch.
  bool certified = false;
  for (uint g = 0; g < 8; ++g) certified = certified || bounded[g];
  if (!survivor_total && certified) {
    if (tid == 0) {
      Split losing = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0};
      output[index] = losing;
    }
    return;
  }
  float4 node_stats;
  if (Mode != kScreenRoot && sp.inherit) {
    // Uniform per node, so every SIMD group derives it without a barrier.
    node_stats = screen_inherited_parent<Scorer>(total_int, scale, p, inherited + node / 2, node & 1);
  } else {
    if (tid == 0) screen_parent<Scorer>(total_int, scale, p, node_bounds, parent);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    node_stats = float4(as_type<float>(parent[0]), as_type<float>(parent[1]),
                        as_type<float>(parent[2]), as_type<float>(parent[3]));
  }
  float w = node_stats.x;
  float parent_gain = node_stats.w;
  Split best = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, w, w, node_stats.y, node_stats.z};
  best.parent_weight = w;
  uint order = 0xffffffffu;
  // Survivors occupy the first lanes, so typically one SIMD group runs exact
  // code. Each lane scores at most one listed candidate; the reduction below
  // orders ties explicitly. An overflowing list instead scores each lane's
  // own survivors, forward first, exactly as the unscreened lane does.
  uint pending = 0, first = 0, second = 0;
  if (survivor_total <= 256) {
    if (tid < survivor_total) { first = survivors[tid]; pending = 1; }
  } else {
    if (forward_survives) first = tid;
    if (backward_survives) { if (forward_survives) second = tid | 0x100u; else first = tid | 0x100u; }
    pending = mine;
  }
  for (uint k = 0; k < pending; ++k) {
    uint entry = k ? second : first;
    uint bin = entry & 0xffu;
    bool reverse = entry >> 8;
    long2 inclusive = prefixes[bin];
    long2 exclusive = bin ? prefixes[bin - 1] : long2(0);
    long2 right = reverse ? present - exclusive : total_int - inclusive;
    long2 candidate_left = reverse ? total_int - right : inclusive;
    float threshold = reverse ? (bin == 0 ? minima[feature] : cuts[ulong(feature) * p.bins + bin - 1]) :
                                cuts[ulong(feature) * p.bins + bin];
    float previous = best.gain;
    screen_exact<Scorer>(best, candidate_left, right, scale, threshold, uint(reverse), feature,
                         monotone[feature], node_bounds, parent_gain, p);
    if (best.gain > previous) order = reverse ? count + (count - 1 - bin) : bin;
  }
  float maximum = simd_max(best.gain);
  uint first_order = simd_min(best.gain == maximum ? order : 0xffffffffu);
  uint winner = simd_min(best.gain == maximum && order == first_order ? lane : 0xffffffffu);
  if (lane == winner) {
    choices[group * 3] = as_type<uint>(best.gain);
    choices[group * 3 + 1] = order;
    choices[group * 3 + 2] = tid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float gain = lane < 8 ? as_type<float>(choices[lane * 3]) : -INFINITY;
    uint position = lane < 8 ? choices[lane * 3 + 1] : 0xffffffffu;
    float group_maximum = simd_max(gain);
    uint group_first = simd_min(gain == group_maximum ? position : 0xffffffffu);
    uint selected = simd_min(lane < 8 && gain == group_maximum && position == group_first ? lane : 0xffffffffu);
    if (lane == 0) choices[24] = choices[selected * 3 + 2];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == choices[24]) output[index] = best;
}

#define SCREEN_SPLIT_STORAGE \
  threadgroup long2 sums[8], prefixes[256]; \
  threadgroup float lower[8]; \
  threadgroup uint bounded[8]; \
  threadgroup ushort survivors[256]; \
  threadgroup atomic_uint survivor_count; \
  threadgroup uint choices[25], parent[4]

#define SCREEN_SPLIT_ARGUMENTS \
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]], \
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]], \
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]], \
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]], \
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]], \
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]], \
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]], \
    constant SplitParams& p [[buffer(14)]], device const uint* parent_histogram [[buffer(15)]], \
    device const HistSubtract* subtraction [[buffer(16)]], device atomic_uint* node_cut [[buffer(17)]], \
    device atomic_uint* statistics [[buffer(18)]], device const Split* inherited [[buffer(19)]], \
    constant ScreenParams& sp [[buffer(20)]], \
    uint2 id [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]

#define SCREEN_SPLIT_KERNEL(name, scorer, mode) \
  kernel void name(SCREEN_SPLIT_ARGUMENTS) { \
    SCREEN_SPLIT_STORAGE; \
    evaluate_screened_feature<scorer, mode>(histogram, scales, cuts, counts, minima, monotone, \
        bounds, allowed, output, node_sums, p, parent_histogram, subtraction, node_cut, statistics, \
        inherited, sp, id, tid, sums, prefixes, lower, bounded, survivors, &survivor_count, choices, parent); \
  }

SCREEN_SPLIT_KERNEL(evaluate_splits_screened_subtracted_gain, 0, kScreenSubtracted)
SCREEN_SPLIT_KERNEL(evaluate_splits_screened_subtracted_gain_stats, 1, kScreenSubtracted)
SCREEN_SPLIT_KERNEL(evaluate_splits_screened_subtracted_gain_stats_score, 2, kScreenSubtracted)
SCREEN_SPLIT_KERNEL(evaluate_splits_screened_terminal_gain, 0, kScreenTerminal)
SCREEN_SPLIT_KERNEL(evaluate_splits_screened_terminal_gain_stats, 1, kScreenTerminal)
SCREEN_SPLIT_KERNEL(evaluate_splits_screened_terminal_gain_stats_score, 2, kScreenTerminal)

// The root twin has the root kernels' first fifteen bindings.
#define SCREEN_ROOT_ARGUMENTS \
    device uint* histogram [[buffer(0)]], device const float2* scales [[buffer(1)]], \
    device const float* cuts [[buffer(2)]], device const uint* counts [[buffer(3)]], \
    device const float* minima [[buffer(4)]], device const int* monotone [[buffer(5)]], \
    device const float2* bounds [[buffer(6)]], device const uchar* allowed [[buffer(7)]], \
    device Split* output [[buffer(8)]], device const uchar* categorical [[buffer(9)]], \
    device const float2* sorted [[buffer(10)]], device const uint* ranks [[buffer(11)]], \
    device uint* categories [[buffer(12)]], device const long2* node_sums [[buffer(13)]], \
    constant SplitParams& p [[buffer(14)]], device atomic_uint* statistics [[buffer(15)]], \
    constant ScreenParams& sp [[buffer(16)]], \
    uint2 id [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]

#define SCREEN_ROOT_KERNEL(name, scorer) \
  kernel void name(SCREEN_ROOT_ARGUMENTS) { \
    SCREEN_SPLIT_STORAGE; \
    evaluate_screened_feature<scorer, kScreenRoot>(histogram, scales, cuts, counts, minima, monotone, \
        bounds, allowed, output, node_sums, p, histogram, nullptr, nullptr, statistics, nullptr, \
        sp, id, tid, sums, prefixes, lower, bounded, survivors, &survivor_count, choices, parent); \
  }

SCREEN_ROOT_KERNEL(evaluate_splits_screened_root_gain, 0)
SCREEN_ROOT_KERNEL(evaluate_splits_screened_root_gain_stats, 1)
SCREEN_ROOT_KERNEL(evaluate_splits_screened_root_gain_stats_score, 2)
#undef SCREEN_ROOT_KERNEL
#undef SCREEN_ROOT_ARGUMENTS
#undef SCREEN_SPLIT_KERNEL
#undef SCREEN_SPLIT_ARGUMENTS
#undef SCREEN_SPLIT_STORAGE
