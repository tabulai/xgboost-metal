// Fuse only the existing single-thread control operations. Their copies below
// intentionally preserve the original kernels as a same-binary control.
// Apply may return early on an error, but its wrapper always prepares the next
// level so all indirect arguments are zeroed for empty/error frontiers. No
// model floating-point arithmetic, threadgroup scratch, or barriers are added.
// Keep the fused controller state local through apply and preparation;
// each wrapper publishes the resulting state once at the end.

inline void tree_driver_fused_apply(thread TreeDriverState& state,
                               device const uint* ids,
                               device const Split* best,
                               device const uint* counts,
                               device const uint* offsets,
                               device Split* history,
                               device uint* flags,
                               device uint* next_ids,
                               device IntPair* next_sums,
                               device Split* selected_splits,
                               device uint* selected_counts,
                               device uint* selected_offsets,
                               device int2* selected_children,
                               device uint* parent_histograms,
                               constant TreeDriverParams& p) {
  uint frontier = state.frontier_count;
  uint depth = state.depth;
  uint created = state.node_count;
  // Always retire the current frontier, even if the tree has already stopped.
  // Subsequent pre-encoded levels then receive explicit zero dispatch counts.
  state.frontier_count = 0;
  state.depth = depth + 1;
  if (state.error) return;
  uint selected = 0;
  for (uint slot = 0; slot < frontier; ++slot) {
    uint id = ids[slot];
    if (id >= created || id >= p.node_capacity) {
      state.error = kTreeDriverNodeError;
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
      state.error = kTreeDriverCapacityError;
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
  state.node_count = created;
  state.frontier_count = 2 * selected;
}

inline void tree_driver_fused_prepare(TreeDriverState state,
                                 device SplitParams* split,
                                 device HistogramPlanParams* histogram,
                                 device PartitionParams* partition,
                                 device uint* parent_count,
                                 device TreeDriverDispatch* plan_histogram_args,
                                 device TreeDriverDispatch* histogram_args,
                                 device TreeDriverDispatch* plan_partition_args,
                                 device TreeDriverDispatch* partition_args,
                                 device TreeDriverDispatch* partition_scan_args,
                                 device TreeDriverDispatch* evaluate_args,
                                 device TreeDriverDispatch* select_args,
                                 device TreeDriverDispatch* subtract_args,
                                 device ulong* zero_bytes,
                                 device TreeDriverDispatch* zero_args,
                                 constant TreeDriverParams& p) {
  uint nodes = state.error ? 0 : state.frontier_count;
  uint parents = state.depth ? nodes / 2 : 0;
  split[0] = p.split;
  split[0].nodes = nodes;
  split[0].use_parent_sums = uint(state.depth != 0);
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
  plan_partition_args[0] = TreeDriverDispatch{(parents + 63) / 64, 1, 1};
  partition_args[0] = TreeDriverDispatch{0, 1, 1};
  partition_scan_args[0] = TreeDriverDispatch{(parents + 63) / 64, 1, 1};
  uint features = p.parallel_splits ? p.split.cols : (p.split.cols + 63) / 64;
  evaluate_args[0] = TreeDriverDispatch{nodes ? features : 0, nodes, 1};
  select_args[0] = TreeDriverDispatch{(nodes + 63) / 64, 1, 1};
  subtract_args[0] = TreeDriverDispatch{
      nodes ? (p.histogram_entries + 255) / 256 : 0,
      histogram[0].subtract ? parents : 0, 1};
  ulong bytes = ulong(nodes) * p.histogram_entries * 16;
  zero_bytes[0] = bytes;
  zero_args[0] = TreeDriverDispatch{uint(min((bytes + 4095) / 4096, 65535ul)), 1, 1};
}

kernel void tree_driver_init_prepare(device TreeDriverState* state [[buffer(0)]],
    device uint* ids [[buffer(1)]],
    device IntPair* sums [[buffer(2)]],
    device uint* counts [[buffer(3)]],
    device uint* offsets [[buffer(4)]],
    device uint* flags [[buffer(5)]],
    device SplitParams* split [[buffer(6)]],
    device HistogramPlanParams* histogram [[buffer(7)]],
    device PartitionParams* partition [[buffer(8)]],
    device uint* parent_count [[buffer(9)]],
    device TreeDriverDispatch* plan_histogram_args [[buffer(10)]],
    device TreeDriverDispatch* histogram_args [[buffer(11)]],
    device TreeDriverDispatch* plan_partition_args [[buffer(12)]],
    device TreeDriverDispatch* partition_args [[buffer(13)]],
    device TreeDriverDispatch* partition_scan_args [[buffer(14)]],
    device TreeDriverDispatch* evaluate_args [[buffer(15)]],
    device TreeDriverDispatch* select_args [[buffer(16)]],
    device TreeDriverDispatch* subtract_args [[buffer(17)]],
    device ulong* zero_bytes [[buffer(18)]],
    device TreeDriverDispatch* zero_args [[buffer(19)]],
    constant TreeDriverParams& p [[buffer(20)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  TreeDriverState local_state{1, 1, 0, 0};
  ids[0] = 0;
  sums[0] = IntPair{0, 0};
  counts[0] = p.partition.rows;
  offsets[0] = 0;
  for (uint node = 0; node < p.node_capacity; ++node) flags[node] = 0;
  tree_driver_fused_prepare(local_state, split, histogram, partition, parent_count, plan_histogram_args, histogram_args, plan_partition_args, partition_args, partition_scan_args, evaluate_args, select_args, subtract_args, zero_bytes, zero_args, p);
  state[0] = local_state;
}

kernel void tree_driver_apply_prepare(device TreeDriverState* state [[buffer(0)]],
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
    device SplitParams* split [[buffer(14)]],
    device HistogramPlanParams* histogram [[buffer(15)]],
    device PartitionParams* partition [[buffer(16)]],
    device uint* parent_count [[buffer(17)]],
    device TreeDriverDispatch* plan_histogram_args [[buffer(18)]],
    device TreeDriverDispatch* histogram_args [[buffer(19)]],
    device TreeDriverDispatch* plan_partition_args [[buffer(20)]],
    device TreeDriverDispatch* partition_args [[buffer(21)]],
    device TreeDriverDispatch* partition_scan_args [[buffer(22)]],
    device TreeDriverDispatch* evaluate_args [[buffer(23)]],
    device TreeDriverDispatch* select_args [[buffer(24)]],
    device TreeDriverDispatch* subtract_args [[buffer(25)]],
    device ulong* zero_bytes [[buffer(26)]],
    device TreeDriverDispatch* zero_args [[buffer(27)]],
    constant TreeDriverParams& p [[buffer(28)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid) return;
  TreeDriverState local_state = state[0];
  tree_driver_fused_apply(local_state, ids, best, counts, offsets, history, flags, next_ids, next_sums, selected_splits, selected_counts, selected_offsets, selected_children, parent_histograms, p);
  tree_driver_fused_prepare(local_state, split, histogram, partition, parent_count, plan_histogram_args, histogram_args, plan_partition_args, partition_args, partition_scan_args, evaluate_args, select_args, subtract_args, zero_bytes, zero_args, p);
  state[0] = local_state;
}
