// Control for the conservative scalar, numeric, unconstrained depthwise path.
// Each kernel uses one thread. Existing histogram, split and partition kernels
// retain all training arithmetic; this driver only copies their results and
// generates dense work descriptions. Model leaves and eta scaling remain in
// the host's original apply path after the one tree completion wait.
// The host limits this path to depth ten. A 512-node final frontier can span
// two CPU pop batches, but terminal expansion only appends children in node-ID
// order. Every frontier that requires another partition has at most 256 nodes.

kernel void tree_driver_init(device TreeDriverState* state [[buffer(0)]],
                              device uint* ids [[buffer(1)]],
                              device IntPair* sums [[buffer(2)]],
                              device uint* counts [[buffer(3)]],
                              device uint* offsets [[buffer(4)]],
                              device uint* flags [[buffer(5)]],
                              device const uint* sampled_count [[buffer(6)]],
                              constant TreeDriverParams& p [[buffer(7)]],
                              uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  state[0] = TreeDriverState{1, 1, 0, 0};
  ids[0] = 0;
  sums[0] = IntPair{0, 0};
  counts[0] = (p.control_flags & kTreeDriverSampledRows) ? sampled_count[0] : p.partition.rows;
  offsets[0] = 0;
  for (uint node = 0; node < p.node_capacity; ++node) flags[node] = 0;
}

kernel void tree_driver_apply(device TreeDriverState* state [[buffer(0)]],
                               device const uint* ids [[buffer(1)]],
                               device const Split* best [[buffer(2)]],
                               device const uint* counts [[buffer(3)]],
                               device const uint* offsets [[buffer(4)]],
                               device Split* history [[buffer(5)]],
                               device uint* flags [[buffer(6)]],
                               device uint* next_ids [[buffer(7)]],
                               device IntPair* next_sums [[buffer(8)]],
                               device Split* selected_splits [[buffer(9)]],
                               device uint* selected_counts [[buffer(10)]],
                               device uint* selected_offsets [[buffer(11)]],
                               device int2* selected_children [[buffer(12)]],
                               device uint* parent_histograms [[buffer(13)]],
                               constant TreeDriverParams& p [[buffer(14)]],
                               uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  uint frontier = state[0].frontier_count;
  uint depth = state[0].depth;
  uint created = state[0].node_count;
  // Always retire the current frontier, even if the tree has already stopped.
  // Subsequent pre-encoded levels then receive explicit zero dispatch counts.
  state[0].frontier_count = 0;
  state[0].depth = depth + 1;
  if (state[0].error) return;
  uint selected = 0;
  for (uint slot = 0; slot < frontier; ++slot) {
    uint id = ids[slot];
    if (id >= created || id >= p.node_capacity) {
      state[0].error = kTreeDriverNodeError;
      return;
    }
    Split split = best[slot];
    history[id] = split;
    flags[id] = kTreeDriverEvaluated;
    // Match Driver::Pop, including gain equality at min_split_loss. Dense
    // frontiers retain increasing tree IDs, so compaction preserves CPU order.
    bool expand = split.feature >= 0 && split.gain > 1.0e-6f &&
                  fp32_greater_equal(split.gain, p.min_gain) && depth < p.max_depth;
    if (!expand) continue;
    if (created > p.node_capacity || p.node_capacity - created < 2) {
      state[0].error = kTreeDriverCapacityError;
      return;
    }
    uint left = created;
    created += 2;
    flags[id] |= kTreeDriverExpanded;
    // Terminal children are created but never evaluated. Their inherited
    // covers, leaves and base weights must survive the host replay untouched.
    if (depth + 1 >= p.max_depth) continue;
    uint child = 2 * selected;
    next_ids[child] = left;
    next_ids[child + 1] = left + 1;
    next_sums[child] = split.left_sum;
    next_sums[child + 1] = split.right_sum;
    selected_splits[selected] = split;
    selected_counts[selected] = counts[slot];
    selected_offsets[selected] = offsets[slot];
    selected_children[selected] = int2(int(child), int(child + 1));
    // Retain the source slot before compacting selected parents. Subtraction
    // reads the previous level's original histogram layout.
    parent_histograms[selected] = slot;
    ++selected;
  }
  state[0].node_count = created;
  state[0].frontier_count = 2 * selected;
}

kernel void tree_driver_prepare(device const TreeDriverState* state [[buffer(0)]],
                                 device SplitParams* split [[buffer(1)]],
                                 device HistogramPlanParams* histogram [[buffer(2)]],
                                 device PartitionParams* partition [[buffer(3)]],
                                 device uint* parent_count [[buffer(4)]],
                                 device TreeDriverDispatch* plan_histogram_args [[buffer(5)]],
                                 device TreeDriverDispatch* histogram_args [[buffer(6)]],
                                 device TreeDriverDispatch* plan_partition_args [[buffer(7)]],
                                 device TreeDriverDispatch* partition_args [[buffer(8)]],
                                 device TreeDriverDispatch* partition_scan_args [[buffer(9)]],
                                 device TreeDriverDispatch* evaluate_args [[buffer(10)]],
                                 device TreeDriverDispatch* select_args [[buffer(11)]],
                                 device TreeDriverDispatch* subtract_args [[buffer(12)]],
                                 device ulong* zero_bytes [[buffer(13)]],
                                 device TreeDriverDispatch* zero_args [[buffer(14)]],
                                 constant TreeDriverParams& p [[buffer(15)]],
                                 uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  uint nodes = state[0].error ? 0 : state[0].frontier_count;
  uint parents = state[0].depth ? nodes / 2 : 0;
  split[0] = p.split;
  split[0].nodes = nodes;
  split[0].use_parent_sums = uint(state[0].depth != 0);
  histogram[0] = p.histogram;
  histogram[0].nodes = nodes;
  histogram[0].subtract = uint(p.histogram.subtract && parents);
  partition[0] = p.partition;
  partition[0].nodes = 2 * parents;
  parent_count[0] = parents;

  // These are tightly packed 12-byte argument records, not Metal uint3
  // vectors (which have 16-byte alignment/storage). Reset both atomic task
  // counters every level, including an empty/error frontier in a reused bank.
  plan_histogram_args[0] = TreeDriverDispatch{(nodes + 63) / 64, 1, 1};
  histogram_args[0] = TreeDriverDispatch{0, 1, 1};
  // Parallel planning uses one threadgroup per parent; parallel selection uses
  // one SIMD-group threadgroup per node. Zero flags keep the original grids.
  plan_partition_args[0] = TreeDriverDispatch{
      p.control_flags & kTreeDriverParallelPlan ? parents : (parents + 63) / 64, 1, 1};
  partition_args[0] = TreeDriverDispatch{0, 1, 1};
  partition_scan_args[0] = TreeDriverDispatch{(parents + 63) / 64, 1, 1};
  uint features = p.parallel_splits ? p.split.cols : (p.split.cols + 63) / 64;
  evaluate_args[0] = TreeDriverDispatch{nodes ? features : 0, nodes, 1};
  select_args[0] = TreeDriverDispatch{
      p.control_flags & kTreeDriverParallelSelect ? nodes : (nodes + 63) / 64, 1, 1};
  subtract_args[0] = TreeDriverDispatch{
      nodes ? (p.histogram_entries + 255) / 256 : 0,
      histogram[0].subtract ? parents : 0, 1};
  ulong bytes = ulong(nodes) * p.histogram_entries * 16;
  zero_bytes[0] = bytes;
  zero_args[0] = TreeDriverDispatch{uint(min((bytes + 4095) / 4096, 65535ul)), 1, 1};
}
