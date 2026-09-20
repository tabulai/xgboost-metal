// Refresh reuses the CUDA-inspired stable row partitioner. Leaves retain row
// order; wide reductions avoid the grower's fixed-point gradient quantization.
kernel void refresh_rows(device uint* rows [[buffer(0)]], device uint* counts [[buffer(1)]],
                          device uint* offsets [[buffer(2)]], constant RefreshParams& p [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
  if (id < p.rows) rows[id] = id;
  if (id < p.nodes) { counts[id] = id == 0 ? p.rows : 0; offsets[id] = 0; }
}

kernel void refresh_partition_plan(device const uint* ids [[buffer(0)]],
                                    device const Node* nodes [[buffer(1)]],
                                    device const uint* counts [[buffer(2)]],
                                    device const uint* offsets [[buffer(3)]],
                                    device PartitionTask* tasks [[buffer(4)]],
                                    device PartitionSegment* segments [[buffer(5)]],
                                    device atomic_uint* arguments [[buffer(6)]],
                                    constant RefreshParams& p [[buffer(7)]],
                                    uint id [[thread_position_in_grid]]) {
  if (id >= p.count) return;
  uint node = ids[p.begin + id], count = counts[node];
  uint tiles = count / 256 + uint(count % 256 != 0);
  uint first = atomic_fetch_add_explicit(arguments, tiles, memory_order_relaxed);
  segments[id] = PartitionSegment{offsets[node], offsets[node] + count, first, first + tiles,
                                  uint(nodes[node].left), uint(nodes[node].right)};
  for (uint tile = 0; tile < tiles; ++tile) {
    uint begin = offsets[node] + tile * 256;
    tasks[first + tile] = PartitionTask{begin, begin + min(256u, count - tile * 256), node, id};
  }
}

kernel void refresh_partition_flags(device const float* data [[buffer(0)]], device const uint* rows [[buffer(1)]],
                                      device const Node* nodes [[buffer(2)]], device const PartitionTask* tasks [[buffer(3)]],
                                      device uint* ranks [[buffer(4)]], device uint* counts [[buffer(5)]],
                                      device const uint* categories [[buffer(6)]], constant RefreshParams& p [[buffer(7)]],
                                      uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[8];
  auto task = tasks[block];
  uint index = task.begin + tid;
  bool left = false;
  if (index < task.end) {
    auto split = nodes[task.node];
    ulong address = p.column_major ? ulong(split.feature) * p.rows + rows[index]
                             : ulong(rows[index]) * p.cols + uint(split.feature);
    left = split_left(data[address], split.threshold, split.default_left, split.split_type,
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

kernel void refresh_sum_plan(device const uint* leaves [[buffer(0)]],
                              device const uint* counts [[buffer(1)]], device const uint* offsets [[buffer(2)]],
                              device PartitionTask* tasks [[buffer(3)]], device uint2* ranges [[buffer(4)]],
                              device atomic_uint* arguments [[buffer(5)]], constant RefreshParams& p [[buffer(6)]],
                              uint id [[thread_position_in_grid]]) {
  if (id >= p.count) return;
  uint node = leaves[id], count = counts[node];
  uint tiles = count / kRefreshRowsPerTile + uint(count % kRefreshRowsPerTile != 0);
  uint first = atomic_fetch_add_explicit(arguments, tiles, memory_order_relaxed);
  ranges[id] = uint2(first, first + tiles);
  for (uint tile = 0; tile < tiles; ++tile) {
    uint begin = offsets[node] + tile * kRefreshRowsPerTile;
    tasks[first + tile] = PartitionTask{begin, begin + min(kRefreshRowsPerTile, count - tile * kRefreshRowsPerTile), node, id};
  }
}

kernel void refresh_partial(device const Pair* gradient [[buffer(0)]], device const uint* rows [[buffer(1)]],
                             device const PartitionTask* tasks [[buffer(2)]], device SurvivalSum* partial [[buffer(3)]],
                             constant RefreshParams& p [[buffer(4)]], uint block [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum scratch[256];
  auto task = tasks[block];
  auto sum = survival_zero();
  for (ulong i = ulong(task.begin) + tid; i < task.end; i += 256) {
    auto pair = gradient[ulong(rows[i]) * p.groups + p.group];
    sum = sum + SurvivalSum{sr(pair.g), sr(pair.h)};
  }
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) partial[block] = total;
}

kernel void refresh_leaf_sums(device const SurvivalSum* partial [[buffer(0)]],
                               device const uint2* ranges [[buffer(1)]], device const uint* leaves [[buffer(2)]],
                               device SurvivalSum* sums [[buffer(3)]], constant RefreshParams& p [[buffer(4)]],
                               uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum scratch[256];
  auto range = ranges[block];
  auto sum = survival_zero();
  for (ulong i = ulong(range.x) + tid; i < range.y; i += 256) sum = sum + partial[i];
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) sums[leaves[block]] = total;
}

kernel void refresh_parent_sums(device const uint* ids [[buffer(0)]], device const Node* nodes [[buffer(1)]],
                                 device SurvivalSum* sums [[buffer(2)]], constant RefreshParams& p [[buffer(3)]],
                                 uint id [[thread_position_in_grid]]) {
  if (id >= p.count) return;
  uint node = ids[p.begin + id];
  sums[node] = sums[nodes[node].left] + sums[nodes[node].right];
}

inline SurvivalReal refresh_l1(SurvivalReal gradient, float alpha) {
  if (survival_less(sr(alpha), gradient)) return gradient - sr(alpha);
  if (survival_less(gradient, sr(-alpha))) return gradient + sr(alpha);
  return sr(0);
}
inline SurvivalReal refresh_weight(SurvivalSum sum, constant RefreshParams& p) {
  if (!survival_less(sr(0), sum.weight)) return sr(0);
  auto weight = -refresh_l1(sum.value, p.reg_alpha) / (sum.weight + sr(p.reg_lambda));
  if (p.max_delta_step != 0 && survival_less(sr(p.max_delta_step), survival_abs(weight)))
    weight = sr(copysign(p.max_delta_step, weight.significand.x));
  return weight;
}
inline SurvivalReal refresh_gain(SurvivalSum sum, constant RefreshParams& p) {
  if (!survival_less(sr(0), sum.weight)) return sr(0);
  if (p.max_delta_step == 0) {
    auto gradient = refresh_l1(sum.value, p.reg_alpha);
    return (gradient * gradient) / (sum.weight + sr(p.reg_lambda));
  }
  auto weight = refresh_weight(sum, p);
  return -(sr(2) * sum.value * weight + (sum.weight + sr(p.reg_lambda)) * (weight * weight) +
           sr(2) * sr(p.reg_alpha) * survival_abs(weight));
}

kernel void refresh_apply(device const uint* ids [[buffer(0)]], device Node* nodes [[buffer(1)]],
                           device const SurvivalSum* sums [[buffer(2)]], device float* weights [[buffer(3)]],
                           constant RefreshParams& p [[buffer(4)]], uint id [[thread_position_in_grid]]) {
  if (id >= p.count) return;
  uint index = ids[id];
  auto node = nodes[index];
  float weight = extended_result(refresh_weight(sums[index], p));
  weights[index] = weight;
  node.cover = extended_result(sums[index].weight);
  if (node.left < 0) {
    if (p.refresh_leaf) node.leaf = node.threshold = weight * p.learning_rate;
  } else {
    node.gain = extended_result(refresh_gain(sums[node.left], p) + refresh_gain(sums[node.right], p) -
                                refresh_gain(sums[index], p));
  }
  nodes[index] = node;
}

kernel void prune_level(device const uint* ids [[buffer(0)]], device Node* nodes [[buffer(1)]],
                         device const float* weights [[buffer(2)]], constant RefreshParams& p [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
  if (id >= p.count) return;
  uint index = ids[p.begin + id];
  auto node = nodes[index];
  if (node.left < 0 || nodes[node.left].left >= 0 || nodes[node.right].left >= 0) return;
  if (!(node.gain < p.min_split_loss || (p.max_depth != 0 && p.depth + 1 > p.max_depth))) return;
  // Match RegTree::MarkDelete's masked split index and default direction. The
  // deleted leaves, their statistics and their categorical sets remain stored.
  nodes[node.left].feature = nodes[node.right].feature = 0x7fffffff;
  nodes[node.left].default_left = nodes[node.right].default_left = 1;
  node.left = node.right = -1;
  node.leaf = node.threshold = p.learning_rate * weights[index];
  nodes[index] = node;
}
