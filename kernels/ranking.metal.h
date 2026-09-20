// LambdaRank uses the CUDA implementation's pair-parallel, two-pass layout.
// Metal accumulates quantized 64-bit values through paired 32-bit atomics;
// scheduling therefore cannot change the result, even within a large query.
inline SurvivalReal ranking_integer(long value) {
  float high = float(value);
  return sr(float2(high, float(value - long(high))));
}
inline SurvivalReal ranking_gain(float value, uint exponential) {
  return exponential ? ranking_integer(long((1u << uint(value)) - 1)) : sr(value);
}
inline SurvivalReal ranking_max(SurvivalReal a, SurvivalReal b) { return survival_less(a, b) ? b : a; }
inline SurvivalReal ranking_epsilon() { return sr(float2(1.0e-16f, -1.68623835e-24f)); }
inline bool ranking_before(uint a, uint b, device const float* primary, device const float* secondary, uint mode) {
  if (a == 0xffffffffu || b == 0xffffffffu) return a < b;
  if (primary[a] != primary[b]) return primary[a] > primary[b];
  if (mode && secondary[a] != secondary[b]) return secondary[a] > secondary[b];
  return a < b;
}
kernel void ranking_order_tiles(device const float* primary [[buffer(0)]],
                                 device const float* secondary [[buffer(1)]],
                                 device const RankingTile* tiles [[buffer(2)]],
                                 device uint* output [[buffer(3)]],
                                 constant RankingParams& p [[buffer(4)]],
                                 uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint values[256];
  auto tile = tiles[block];
  values[tid] = tid < tile.size ? uint(tile.begin + tid) : 0xffffffffu;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 2; k <= 256; k <<= 1) for (uint j = k >> 1; j; j >>= 1) {
    uint a = values[tid], b = values[tid ^ j];
    bool take_min = ((tid & k) == 0) == ((tid & j) == 0);
    uint next = ranking_before(a, b, primary, secondary, p.mode) == take_min ? a : b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    values[tid] = next;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid < tile.size) output[tile.begin + tid] = values[tid];
}
kernel void ranking_order_merge(device const float* primary [[buffer(0)]],
                                 device const float* secondary [[buffer(1)]],
                                 device const uint* query_ptr [[buffer(2)]],
                                 device const uint* row_query [[buffer(3)]],
                                 device const uint* source [[buffer(4)]], device uint* output [[buffer(5)]],
                                 constant RankingParams& p [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint query = row_query[row];
  ulong start = query_ptr[query], stop = query_ptr[query + 1];
  ulong begin = start + ((ulong(row) - start) / (2ul * p.width)) * (2ul * p.width);
  ulong middle = min(begin + p.width, stop), end = min(begin + 2ul * p.width, stop);
  bool left = row < middle;
  ulong own = left ? begin : middle, other = left ? middle : begin;
  ulong lo = other, hi = left ? end : middle;
  uint index = source[row];
  while (lo < hi) {
    ulong mid = lo + (hi - lo) / 2;
    if (ranking_before(source[mid], index, primary, secondary, p.mode)) lo = mid + 1;
    else hi = mid;
  }
  output[begin + row - own + lo - other] = index;
}
kernel void ranking_inverse(device const uint* order [[buffer(0)]], device const uint* query_ptr [[buffer(1)]],
                             device const uint* row_query [[buffer(2)]], device uint* inverse [[buffer(3)]],
                             constant RankingParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row < p.rows) inverse[order[row]] = row - query_ptr[row_query[row]];
}
kernel void ranking_discounts(device SurvivalReal* discount [[buffer(0)]],
                               constant RankingParams& p [[buffer(1)]], uint row [[thread_position_in_grid]]) {
  if (row < p.max_size) discount[row] = sr(kExtendedLn2) / survival_log(ranking_integer(long(row) + 2));
}
kernel void ranking_weight_inputs(device const float* weights [[buffer(0)]], device SurvivalSum* output [[buffer(1)]],
                                   constant RankingParams& p [[buffer(2)]], uint query [[thread_position_in_grid]]) {
  if (query < p.queries) output[query] = SurvivalSum{sr(0), sr(weights[query])};
}
kernel void ranking_weight_norm(device const SurvivalSum* total [[buffer(0)]], device float* norm [[buffer(1)]],
                                constant RankingParams& p [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  if (!tid) *norm = extended_result(ranking_integer(p.queries) / total[0].weight);
}
kernel void ranking_idcg_partial(device const float* labels [[buffer(0)]], device const uint* order [[buffer(1)]],
                                  device const uint* query_ptr [[buffer(2)]], device const RankingTile* tiles [[buffer(3)]],
                                  device const SurvivalReal* discount [[buffer(4)]], device SurvivalSum* partial [[buffer(5)]],
                                  constant RankingParams& p [[buffer(6)]],
                                  uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum values[256];
  auto tile = tiles[block];
  uint rank = uint(tile.begin) - query_ptr[tile.query] + tid;
  auto value = sr(0);
  if (tid < tile.size && rank < p.num_pairs) value = ranking_gain(labels[order[tile.begin + tid]], p.exp_gain) * discount[rank];
  auto total = survival_block_reduce(SurvivalSum{value, sr(0)}, values, tid);
  if (!tid) partial[block] = total;
}
kernel void ranking_idcg_finish(device const SurvivalSum* partial [[buffer(0)]], device const uint* tile_ptr [[buffer(1)]],
                                 device SurvivalReal* inverse [[buffer(2)]], constant RankingParams& p [[buffer(3)]],
                                 uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  auto total = sr(0);
  for (uint i = tile_ptr[query]; i < tile_ptr[query + 1]; ++i) total = total + partial[i].value;
  inverse[query] = total.significand.x == 0 ? sr(0) : sr(1) / total;
}

// Segmented scans preserve query boundaries at both SIMD and threadgroup edges.
inline SurvivalSum ranking_warp_scan(SurvivalSum value, uint query, uint lane) {
  for (ushort delta = 1; delta < 32; delta *= 2) {
    SurvivalSum previous{survival_shuffle_up(value.value, delta), survival_shuffle_up(value.weight, delta)};
    auto previous_query = simd_shuffle_up(query, delta);
    if (lane >= delta && query == previous_query) value = previous + value;
  }
  return value;
}
kernel void ranking_scan_tiles(device const SurvivalSum* input [[buffer(0)]], device const uint* queries [[buffer(1)]],
                                device SurvivalSum* output [[buffer(2)]], device SurvivalSum* totals [[buffer(3)]],
                                device uint* total_queries [[buffer(4)]], constant RankingParams& p [[buffer(5)]],
                                uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum warps[8]; threadgroup uint warp_queries[8];
  ulong row = ulong(block) * 256 + tid;
  uint query = row < p.rows ? queries[row] : 0xffffffffu, lane = tid & 31, warp = tid / 32;
  auto value = ranking_warp_scan(row < p.rows ? input[row] : survival_zero(), query, lane);
  if (lane == 31) { warps[warp] = value; warp_queries[warp] = query; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp == 0) {
    auto total = ranking_warp_scan(lane < 8 ? warps[lane] : survival_zero(), lane < 8 ? warp_queries[lane] : 0xffffffffu, lane);
    if (lane < 8) warps[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp && query == warp_queries[warp - 1]) value = warps[warp - 1] + value;
  if (row < p.rows) output[row] = value;
  if (row == min(ulong(p.rows), (ulong(block) + 1) * 256) - 1) {
    totals[block] = value; total_queries[block] = query;
  }
}
kernel void ranking_scan_offsets(device SurvivalSum* output [[buffer(0)]], device const uint* queries [[buffer(1)]],
                                  device const SurvivalSum* totals [[buffer(2)]], device const uint* total_queries [[buffer(3)]],
                                  constant RankingParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
  if (row < p.rows && row >= 256 && queries[row] == total_queries[row / 256 - 1])
    output[row] = totals[row / 256 - 1] + output[row];
}
kernel void ranking_map_inputs(device const float* labels [[buffer(0)]], device const uint* order [[buffer(1)]],
                                device const uint* query_ptr [[buffer(2)]], device const uint* row_query [[buffer(3)]],
                                device SurvivalSum* output [[buffer(4)]], constant RankingParams& p [[buffer(5)]],
                                uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  auto y = sr(labels[order[row]]);
  output[row] = SurvivalSum{y, y / ranking_integer(long(row - query_ptr[row_query[row]]) + 1)};
}

// minstd_rand and libc++'s independent-bits/rejection distribution. In mean
// sampling the draw order is bucket, sample, then document, as in the CPU source.
inline uint ranking_random(thread uint& state) {
  ulong product = ulong(state) * 48271;
  ulong value = (product & 0x7ffffffful) + (product >> 31);
  if (value >= 2147483647ul) value -= 2147483647ul;
  state = uint(value);
  return state;
}
inline uint ranking_uniform(thread uint& state, uint range) {
  if (range == 1) return 0;
  uint width = 32 - clz(range - 1), n = (width + 29) / 30, w0 = width / n;
  uint y0 = (2147483646u >> w0) << w0;
  if (2147483646u - y0 > y0 / n) { ++n; w0 = width / n; y0 = (2147483646u >> w0) << w0; }
  uint n0 = n - width % n, y1 = (2147483646u >> (w0 + 1)) << (w0 + 1);
  uint result;
  do {
    result = 0;
    for (uint i = 0; i < n; ++i) {
      uint bits = w0 + uint(i >= n0), limit = i < n0 ? y0 : y1, value;
      do { value = ranking_random(state) - 1; } while (value >= limit);
      result = (result << bits) + (value & ((1u << bits) - 1));
    }
  } while (result >= range);
  return result;
}
kernel void ranking_mean_pairs(device const float* labels [[buffer(0)]], device const uint* order [[buffer(1)]],
                                device const uint* query_ptr [[buffer(2)]], device const ulong* pair_ptr [[buffer(3)]],
                                device uint2* pairs [[buffer(4)]], constant RankingParams& p [[buffer(5)]],
                                uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  uint begin = query_ptr[query], size = query_ptr[query + 1] - begin;
  uint state = (p.seed + query) % 2147483647u;
  if (!state) state = 1;
  ulong output = pair_ptr[query];
  for (uint i = 0; i < size;) {
    uint j = i + 1;
    while (j < size && labels[order[begin + i]] == labels[order[begin + j]]) ++j;
    uint outside = i + size - j;
    if (!outside) {
      for (; output < pair_ptr[query + 1]; ++output) pairs[output] = uint2(begin, begin);
      break;
    }
    for (uint sample = 0; sample < p.num_pairs; ++sample) for (uint position = i; position < j; ++position) {
      uint chosen = ranking_uniform(state, outside);
      if (chosen >= i) chosen = chosen - i + j;
      pairs[output++] = uint2(order[begin + position], order[begin + chosen]);
    }
    i = j;
  }
}
kernel void ranking_bias_init(device SurvivalReal* plus [[buffer(0)]], device SurvivalReal* minus [[buffer(1)]],
                               constant RankingParams& p [[buffer(2)]], uint position [[thread_position_in_grid]]) {
  if (position < p.positions) { plus[position] = sr(1); minus[position] = sr(1); }
}

inline uint2 ranking_topk_pair(ulong pair, uint size, uint k) {
  uint lo = 0, hi = min(size, k);
  while (lo < hi) {
    uint mid = lo + (hi - lo) / 2;
    ulong offset = ulong(mid) * (2ul * size - mid - 1) / 2;
    if (offset <= pair) lo = mid + 1;
    else hi = mid;
  }
  uint i = lo - 1;
  ulong offset = ulong(i) * (2ul * size - i - 1) / 2;
  return uint2(i, i + 1 + uint(pair - offset));
}
inline SurvivalReal ranking_delta_map(float yh, float yl, uint rh, uint rl, uint begin, uint size,
                                       device const SurvivalSum* statistics) {
  if (rh > rl) { uint rank = rh; rh = rl; rl = rank; float label = yh; yh = yl; yl = label; }
  auto n = statistics[begin + rh].value, m = statistics[begin + rl].value;
  auto b = statistics[begin + rl - 1].weight - statistics[begin + rh].weight;
  auto hr = ranking_integer(long(rh) + 1), lr = ranking_integer(long(rl) + 1);
  auto a = yh < yl ? m / lr - (n + sr(1)) / hr : n / hr - m / lr;
  return (yh < yl ? a - b : a + b) / statistics[begin + size - 1].value;
}
struct RankingPartial { SurvivalReal sum_lambda, max_cost; float max_grad, max_hess; uint pad0, pad1; };
inline RankingPartial ranking_partial_add(RankingPartial a, RankingPartial b) {
  return RankingPartial{a.sum_lambda + b.sum_lambda, ranking_max(a.max_cost, b.max_cost),
                        max(a.max_grad, b.max_grad), max(a.max_hess, b.max_hess), 0, 0};
}
kernel void ranking_pair_gradients(device const float* prediction [[buffer(0)]], device const float* labels [[buffer(1)]],
                                    device const uint* order [[buffer(2)]], device const uint* inverse [[buffer(3)]],
                                    device const uint* query_ptr [[buffer(4)]], device const ulong* pair_ptr [[buffer(5)]],
                                    device const RankingTile* tiles [[buffer(6)]], device uint2* pairs [[buffer(7)]],
                                    device const SurvivalReal* discounts [[buffer(8)]], device const SurvivalReal* idcg [[buffer(9)]],
                                    device const SurvivalSum* map_statistics [[buffer(10)]],
                                    device const SurvivalReal* bias_plus [[buffer(11)]], device const SurvivalReal* bias_minus [[buffer(12)]],
                                    device Pair* gradients [[buffer(13)]], device SurvivalReal* costs [[buffer(14)]],
                                    device RankingPartial* partial [[buffer(15)]], constant RankingParams& p [[buffer(16)]],
                                    uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup RankingPartial shared[256];
  auto tile = tiles[block];
  RankingPartial result{sr(0), sr(0), 0, 0, 0, 0};
  if (tid < tile.size) {
    ulong index = tile.begin + tid;
    uint begin = query_ptr[tile.query], size = query_ptr[tile.query + 1] - begin;
    uint2 pair;
    if (p.mean) pair = pairs[index];
    else {
      auto ranks = ranking_topk_pair(index - pair_ptr[tile.query], size, p.num_pairs);
      pair = uint2(order[begin + ranks.x], order[begin + ranks.y]);
    }
    if (labels[pair.x] < labels[pair.y]) pair = pair.yx;
    pairs[index] = pair;
    Pair pg{0, 0}; auto cost = sr(0);
    if (labels[pair.x] != labels[pair.y]) {
      uint rh = inverse[pair.x], rl = inverse[pair.y];
      auto delta = sr(1);
      if (p.objective == 17) {
        auto high = ranking_gain(labels[pair.x], p.exp_gain), low = ranking_gain(labels[pair.y], p.exp_gain);
        auto original = high * discounts[rh] + low * discounts[rl];
        auto changed = low * discounts[rh] + high * discounts[rl];
        delta = (original - changed) * idcg[tile.query];
      } else if (p.objective == 18) {
        delta = ranking_delta_map(labels[pair.x], labels[pair.y], rh, rl, begin, size, map_statistics);
      }
      delta = survival_abs(delta);
      float difference = prediction[pair.x] - prediction[pair.y];
      float exponential = extended_result(survival_exp(sr(min(-difference, 88.7f))));
      float sigmoid_value = 1.0f / ((exponential + 1.0f) + 1.0e-16f);
      auto sigmoid = sr(sigmoid_value), complement = sr(1) - sigmoid;
      if (p.score_normalization && prediction[order[begin]] != prediction[order[begin + size - 1]])
        delta = delta / (sr(fabs(difference)) + sr(float2(0.01f, 2.23517418e-10f)));
      if (p.unbiased) cost = survival_log(sr(1) / complement) * delta;
      auto lambda = -complement * delta;
      auto hessian = ranking_max(sigmoid * complement, ranking_epsilon()) * delta * sr(2);
      uint high_position = pair.x - begin, low_position = pair.y - begin;
      if (p.unbiased && high_position < p.positions && low_position < p.positions) {
        auto plus = bias_plus[high_position], minus = bias_minus[low_position];
        bool use_plus = !survival_less(plus, ranking_epsilon()), use_minus = !survival_less(minus, ranking_epsilon());
        if (use_plus && use_minus) { auto bias = plus * minus; lambda = lambda / bias; hessian = hessian / bias; }
        if (use_minus) result.max_cost = ranking_max(result.max_cost, cost / minus);
        if (use_plus) result.max_cost = ranking_max(result.max_cost, cost / plus);
      }
      pg = Pair{extended_result(lambda), extended_result(hessian)};
    }
    gradients[index] = pg;
    if (p.unbiased) costs[index] = cost;
    result.sum_lambda = sr(pg.g) * sr(-2);
    result.max_grad = fabs(pg.g); result.max_hess = fabs(pg.h);
  }
  shared[tid] = result;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint delta = 128; delta; delta >>= 1) {
    if (tid < delta) shared[tid] = ranking_partial_add(shared[tid], shared[tid + delta]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!tid) partial[block] = shared[0];
}
kernel void ranking_group_statistics(device const RankingPartial* partial [[buffer(0)]],
                                      device const uint* tile_ptr [[buffer(1)]], device const ulong* pair_ptr [[buffer(2)]],
                                      device RankingStats* statistics [[buffer(3)]], constant RankingParams& p [[buffer(4)]],
                                      uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  RankingPartial total{sr(0), sr(0), 0, 0, 0, 0};
  for (uint tile = tile_ptr[query]; tile < tile_ptr[query + 1]; ++tile) total = ranking_partial_add(total, partial[tile]);
  auto norm = sr(1);
  if (p.normalization) {
    if (p.mean) norm = sr(1) / ranking_integer(p.num_pairs);
    else if (total.sum_lambda.significand.x > 0)
      norm = (survival_log(sr(1) + total.sum_lambda) / sr(kExtendedLn2)) / total.sum_lambda;
  }
  ulong count = pair_ptr[query + 1] - pair_ptr[query];
  int count_bits = count > 1 ? 64 - clz(count - 1) : 0;
  int ge = 0, he = 0;
  frexp(total.max_grad, ge); frexp(total.max_hess, he);
  statistics[query] = RankingStats{extended_result(norm), 60 - count_bits - ge, 60 - count_bits - he,
                                   60 - count_bits - total.max_cost.exponent};
}
inline long ranking_quantize(SurvivalReal value, int shift) {
  // Convert both significand words before rounding; double-range bias costs
  // need not fit in an intermediate float32 value.
  float2 scaled = ldexp(value.significand, int2(value.exponent + shift));
  long high = long(rint(scaled.x));
  return high + long(rint((scaled.x - float(high)) + scaled.y));
}
kernel void ranking_pair_accumulate(device const Pair* gradients [[buffer(0)]], device const SurvivalReal* costs [[buffer(1)]],
                                     device const uint2* pairs [[buffer(2)]], device const RankingTile* tiles [[buffer(3)]],
                                     device const uint* query_ptr [[buffer(4)]], device const RankingStats* statistics [[buffer(5)]],
                                     device const SurvivalReal* bias_plus [[buffer(6)]], device const SurvivalReal* bias_minus [[buffer(7)]],
                                     device atomic_uint* accumulated [[buffer(8)]], device atomic_uint* position_costs [[buffer(9)]],
                                     device const uint* inverse [[buffer(10)]], constant RankingParams& p [[buffer(11)]], uint block [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  auto tile = tiles[block];
  bool valid = tid < tile.size;
  ulong index = tile.begin + tid;
  auto pair = valid ? pairs[index] : uint2(0); auto gradient = valid ? gradients[index] : Pair{0, 0}; auto stats = statistics[tile.query];
  long g = long(rint(ldexp(gradient.g, stats.grad_shift))), h = long(rint(ldexp(gradient.h, stats.hess_shift)));
  if (p.mode && !p.mean) {
    // Top-k's adjacent pairs share their leading score rank. Reduce that
    // endpoint within each SIMD segment before issuing contended atomics.
    bool first = valid && inverse[pair.x] < inverse[pair.y];
    uint leading = valid ? (first ? pair.x : pair.y) : 0xffffffffu;
    uint tail = first ? pair.y : pair.x, lane = tid & 31;
    long lg = first ? g : -g, lh = h;
    for (ushort delta = 1; delta < 32; delta <<= 1) {
      uint next = simd_shuffle_down(leading, delta);
      long ng = as_type<long>(simd_shuffle_down(as_type<uint2>(lg), delta));
      long nh = as_type<long>(simd_shuffle_down(as_type<uint2>(lh), delta));
      if (lane + delta < 32 && leading == next) { lg += ng; lh += nh; }
    }
    uint previous = simd_shuffle_up(leading, 1);
    if (valid && (!lane || leading != previous)) {
      add64(accumulated + ulong(leading) * 4, lg); add64(accumulated + ulong(leading) * 4 + 2, lh);
    }
    if (valid) {
      add64(accumulated + ulong(tail) * 4, first ? -g : g); add64(accumulated + ulong(tail) * 4 + 2, h);
    }
  } else if (valid) {
    add64(accumulated + ulong(pair.x) * 4, g); add64(accumulated + ulong(pair.x) * 4 + 2, h);
    add64(accumulated + ulong(pair.y) * 4, -g); add64(accumulated + ulong(pair.y) * 4 + 2, h);
  }
  if (p.unbiased && valid) {
    uint begin = query_ptr[tile.query], high = pair.x - begin, low = pair.y - begin;
    if (high < p.positions && low < p.positions) {
      auto cost = costs[index];
      if (!survival_less(bias_minus[low], ranking_epsilon()))
        add64(position_costs + ulong(pair.x) * 4, ranking_quantize(cost / bias_minus[low], stats.cost_shift));
      if (!survival_less(bias_plus[high], ranking_epsilon()))
        add64(position_costs + ulong(pair.y) * 4 + 2, ranking_quantize(cost / bias_plus[high], stats.cost_shift));
    }
  }
}
kernel void ranking_gradient_finish(device const uint* accumulated [[buffer(0)]], device const uint* row_query [[buffer(1)]],
                                      device const RankingStats* statistics [[buffer(2)]], device const float* weights [[buffer(3)]],
                                      device const float* weight_norm [[buffer(4)]], device Pair* output [[buffer(5)]],
                                      constant RankingParams& p [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint query = row_query[row]; auto stats = statistics[query];
  auto integers = load_hist_int(accumulated, ulong(row) * 4);
  float g = ldexp(float(integers.x), -stats.grad_shift), h = ldexp(float(integers.y), -stats.hess_shift);
  g = ((g * stats.norm) * weights[query]) * weight_norm[0];
  h = ((h * stats.norm) * weights[query]) * weight_norm[0];
  output[row] = Pair{g, h};
}

kernel void ranking_bias_partial(device const uint* costs [[buffer(0)]], device const uint* query_ptr [[buffer(1)]],
                                   device const RankingStats* statistics [[buffer(2)]], device SurvivalSum* partial [[buffer(3)]],
                                   constant RankingParams& p [[buffer(4)]], uint2 block [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum shared[256];
  uint query = block.x * 256 + tid;
  auto value = survival_zero();
  if (query < p.queries && block.y < query_ptr[query + 1] - query_ptr[query]) {
    uint row = query_ptr[query] + block.y;
    auto integers = load_hist_int(costs, ulong(row) * 4);
    value = SurvivalSum{ranking_integer(integers.x), ranking_integer(integers.y)};
    value.value.exponent -= statistics[query].cost_shift;
    value.weight.exponent -= statistics[query].cost_shift;
  }
  auto total = survival_block_reduce(value, shared, tid);
  if (!tid) partial[ulong(block.y) * ((ulong(p.queries) + 255) / 256) + block.x] = total;
}
kernel void ranking_bias_reduce(device const SurvivalSum* partial [[buffer(0)]], device SurvivalSum* total [[buffer(1)]],
                                 constant RankingParams& p [[buffer(2)]], uint position [[thread_position_in_grid]]) {
  if (position >= p.positions) return;
  uint blocks = uint((ulong(p.queries) + 255) / 256);
  auto value = survival_zero();
  for (uint i = 0; i < blocks; ++i) value = value + partial[ulong(position) * blocks + i];
  total[position] = value;
}
inline SurvivalReal ranking_bias_ratio(SurvivalReal numerator, SurvivalReal denominator, SurvivalReal regularizer) {
  if (numerator.significand.x == 0) return regularizer.significand.x == 0 ? sr(1) : sr(0);
  return survival_exp(survival_log(numerator / denominator) * regularizer);
}
kernel void ranking_bias_update(device const SurvivalSum* costs [[buffer(0)]], device SurvivalReal* plus [[buffer(1)]],
                                 device SurvivalReal* minus [[buffer(2)]], constant RankingParams& p [[buffer(3)]],
                                 uint position [[thread_position_in_grid]]) {
  if (position >= p.positions) return;
  auto regularizer = SurvivalReal{float2(p.regularizer_high, p.regularizer_low), p.regularizer_exponent};
  if (!survival_less(costs[0].value, ranking_epsilon()))
    plus[position] = ranking_bias_ratio(costs[position].value, costs[0].value, regularizer);
  if (!survival_less(costs[0].weight, ranking_epsilon()))
    minus[position] = ranking_bias_ratio(costs[position].weight, costs[0].weight, regularizer);
}

kernel void ranking_metric_queries(device const float* labels [[buffer(0)]], device const uint* order [[buffer(1)]],
                                    device const uint* query_ptr [[buffer(2)]], device const float* weights [[buffer(3)]],
                                    device const SurvivalReal* discount [[buffer(4)]], device const SurvivalReal* idcg [[buffer(5)]],
                                    device SurvivalSum* output [[buffer(6)]], constant RankingParams& p [[buffer(7)]],
                                    uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  uint begin = query_ptr[query], size = query_ptr[query + 1] - begin, n = min(size, p.num_pairs);
  auto value = sr(0), hits = sr(0), weight = sr(weights[query]);
  if (p.mode == 0) {
    if (idcg[query].significand.x <= 0) {
      // The pinned CPU and CUDA NDCG paths both leave this fallback unweighted.
      output[query] = SurvivalSum{sr(p.minus ? 0 : 1), weight};
      return;
    }
    for (uint i = 0; i < n; ++i)
      value = value + (discount[i] * ranking_gain(labels[order[begin + i]], p.exp_gain)) * idcg[query];
    value = value * weight;
  } else if (p.mode == 1) {
    for (uint i = 0; i < size; ++i) {
      auto y = sr(labels[order[begin + i]]);
      hits = hits + y;
      if (i < n) value = value + (hits / ranking_integer(long(i) + 1)) * y;
    }
    if (hits.significand.x > 0) {
      auto k = ranking_integer(p.num_pairs);
      value = value / (survival_less(hits, k) ? hits : k);
    } else value = sr(p.minus ? 0 : 1);
    value = value * weight;
  } else {
    for (uint i = 0; i < n; ++i) value = value + sr(labels[order[begin + i]] * weights[query]);
    value = value / ranking_integer(n);
  }
  output[query] = SurvivalSum{value, weight};
}
kernel void ranking_metric_finish(device const SurvivalSum* input [[buffer(0)]], device const SurvivalSum* weights [[buffer(1)]],
                                   device MetricRatio* output [[buffer(2)]], constant RankingParams& p [[buffer(3)]],
                                   uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  auto value = weights[0].weight.significand.x > 0 ? input[0].value / weights[0].weight : input[0].value;
  if (p.mode >= 3 && weights[0].weight.significand.x == 0) value = sr(NAN);
  *output = MetricRatio{value.significand.x, value.significand.y, value.exponent, 1};
}

kernel void ranking_auc_merge(device const float* prediction [[buffer(0)]], device const uint* query_ptr [[buffer(1)]],
                               device const uint* row_query [[buffer(2)]], device const uint* source [[buffer(3)]],
                               device uint* output [[buffer(4)]], device ulong2* counts [[buffer(5)]],
                               constant RankingParams& p [[buffer(6)]], uint row [[thread_position_in_grid]]) {
  if (row >= p.rows) return;
  uint query = row_query[row], index = source[row];
  ulong start = query_ptr[query], stop = query_ptr[query + 1];
  ulong begin = start + ((ulong(row) - start) / (2ul * p.width)) * (2ul * p.width);
  ulong middle = min(begin + p.width, stop), end = min(begin + 2ul * p.width, stop);
  bool left = row < middle;
  ulong own = left ? begin : middle, other = left ? middle : begin, other_end = left ? end : middle;
  float value = prediction[index];
  ulong lo = other, hi = other_end;
  while (lo < hi) {
    ulong mid = lo + (hi - lo) / 2; uint candidate = source[mid];
    if (prediction[candidate] < value || (prediction[candidate] == value && candidate < index)) lo = mid + 1;
    else hi = mid;
  }
  output[begin + row - own + lo - other] = index;
  if (left) {
    lo = other; hi = other_end;
    while (lo < hi) {
      ulong mid = lo + (hi - lo) / 2;
      if (prediction[source[mid]] < value) lo = mid + 1;
      else hi = mid;
    }
    ulong less = lo - other;
    lo = other; hi = other_end;
    while (lo < hi) {
      ulong mid = lo + (hi - lo) / 2;
      if (prediction[source[mid]] <= value) lo = mid + 1;
      else hi = mid;
    }
    counts[index] += ulong2(less, lo - other - less);
  }
}
kernel void ranking_auc_queries(device const ulong2* counts [[buffer(0)]], device const uint* query_ptr [[buffer(1)]],
                                 device const float* weights [[buffer(2)]], device SurvivalSum* output [[buffer(3)]],
                                 constant RankingParams& p [[buffer(4)]], uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  uint begin = query_ptr[query], end = query_ptr[query + 1];
  if (end - begin < 3) { output[query] = survival_zero(); return; }
  ulong2 sum(0);
  for (uint i = begin; i < end; ++i) sum += counts[i];
  float squared = weights[query] * weights[query];
  auto numerator = ranking_integer(long(sum.x)) * sr(squared) + ranking_integer(long(sum.y)) * sr(0.5f * squared);
  auto n = ranking_integer(long(end - begin));
  auto denominator = (n * (n - sr(1)) / sr(2)) * sr(squared);
  auto value = denominator.significand.x == 0 ? sr(0) : numerator / denominator;
  output[query] = SurvivalSum{value, sr(1)};
}
inline SurvivalReal ranking_pr_area(SurvivalReal fp0, SurvivalReal fp, SurvivalReal tp0, SurvivalReal tp,
                                      SurvivalReal total_positive) {
  auto previous = tp0 / total_positive, current = tp / total_positive;
  if (tp.exponent == tp0.exponent && all(tp.significand == tp0.significand)) return sr(0);
  auto h = (fp - fp0) / (tp - tp0), a = h + sr(1), b = (fp0 - h * tp0) / total_positive;
  if (b.significand.x == 0) return (current - previous) / a;
  return (current - previous - (b / a) * (survival_log(a * current + b) - survival_log(a * previous + b))) / a;
}
kernel void ranking_aucpr_queries(device const float* prediction [[buffer(0)]], device const float* labels [[buffer(1)]],
                                   device const uint* order [[buffer(2)]], device const uint* query_ptr [[buffer(3)]],
                                   device const float* weights [[buffer(4)]], device SurvivalSum* output [[buffer(5)]],
                                   constant RankingParams& p [[buffer(6)]], uint query [[thread_position_in_grid]]) {
  if (query >= p.queries) return;
  uint begin = query_ptr[query], end = query_ptr[query + 1];
  float weight = weights[query]; auto positive = sr(0), negative = sr(0);
  for (uint i = begin; i < end; ++i) {
    positive = positive + sr(labels[i] * weight);
    negative = negative + sr((1.0f - labels[i]) * weight);
  }
  if (positive.significand.x <= 0 || negative.significand.x <= 0) { output[query] = survival_zero(); return; }
  auto tp0 = sr(0), fp0 = sr(0), tp = sr(labels[order[begin]] * weight);
  auto fp = (sr(1) - sr(labels[order[begin]])) * sr(weight), value = sr(0);
  for (uint i = begin + 1; i < end; ++i) {
    if (prediction[order[i]] != prediction[order[i - 1]]) {
      value = value + ranking_pr_area(fp0, fp, tp0, tp, positive);
      fp0 = fp; tp0 = tp;
    }
    tp = tp + sr(labels[order[i]] * weight);
    fp = fp + sr((1.0f - labels[order[i]]) * weight);
  }
  value = value + ranking_pr_area(fp0, fp, tp0, tp, positive);
  output[query] = isnan(value.significand.x) ? survival_zero() : SurvivalSum{value, sr(1)};
}
