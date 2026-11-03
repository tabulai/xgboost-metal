// Test-only library: the production metallib does not contain this entry point.
// For each candidate, run every exact scalar scorer family and the screening
// enclosure used by scalar_screen.metal.h on the same integer statistics.
#include "boost.metal"

kernel void screen_probe_candidates(device const long4* sums [[buffer(0)]],
                                    device const float2* scales [[buffer(1)]],
                                    device uint4* exact [[buffer(2)]],
                                    device float4* enclosure [[buffer(3)]],
                                    device uint4* parents [[buffer(4)]],
                                    device uint2* inherited [[buffer(5)]],
                                    constant SplitParams& p [[buffer(6)]],
                                    uint i [[thread_position_in_grid]]) {
  if (i >= p.nodes) return;
  long4 value = sums[i];
  long2 left(value.x, value.y), right(value.z, value.w), total = left + right;
  float2 scale = scales[i], bounds(-INFINITY, INFINITY);
  auto fast = scalar_histogram_sum_fast(total, scale);
  auto generic = scalar_histogram_sum(total, scale);
  float2 fused = scalar_weight_gain_fused(fast, p, bounds);
  float stats_gain = scalar_gain_fast(fast, scalar_weight(fast, p, bounds), p);
  float generic_gain = scalar_gain_fast(generic, scalar_weight(generic, p, bounds), p);
  Split a = {-INFINITY, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0}, b = a, c = a;
  consider_split_score_fused(a, left, right, scale, 0, 0, 0, 0, bounds, fused.y, p);
  consider_split_stats_fast<false, true>(b, left, right, scale, 0, 0, 0, 0, bounds, stats_gain, p);
  consider_split_gain<false>(c, left, right, scale, 0, 0, 0, 0, bounds, generic_gain, p);
  float parent_lo = 0, parent_hi = 0, lo = 0, hi = 0;
  bool parent = screen_parent_gain(total, scale, p, parent_lo, parent_hi);
  uint result = parent ? screen_candidate(left, right, scale, parent_lo, parent_hi, p, lo, hi) : 3u;
  exact[i] = uint4(as_type<uint>(a.gain), as_type<uint>(b.gain), as_type<uint>(c.gain), result);
  enclosure[i] = float4(lo, hi, parent_lo, parent_hi);
  parents[i] = uint4(as_type<uint>(fused.y), as_type<uint>(stats_gain), as_type<uint>(generic_gain), uint(parent));
  // Parent gains recomputed from a selected split's (unclamped) weight.
  inherited[i] = uint2(as_type<uint>(screen_fused_gain(fast, fused.x, p)),
                       as_type<uint>(scalar_gain_fast(fast, fused.x, p)));
}
