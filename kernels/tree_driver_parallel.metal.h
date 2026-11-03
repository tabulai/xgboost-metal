// Parallel form of the fused controller. Expansion runs in one threadgroup
// over a frontier of at most kTreeDriverSlots slots; the next level's
// parameters and indirect arguments are still prepared by thread zero.
//
// Node IDs: an exclusive prefix over accepted slots, in slot order, equals
// the serial `left = created; created += 2`. Terminal-ness depends only on
// depth, so the same rank is the compacted selected-parent index on every
// nonterminal level.
//
// Errors: the first node or capacity error in slot order stops exactly the
// writes that the serial loop stops (a slot failing the capacity check still
// records its evaluated history first). The state, including the unchanged
// node count after an error, equals the serial state.
//
// Flags: initialization no longer clears every node. Apply zeroes both
// children's flags when it creates them, and every evaluated node overwrites
// its own flag. The host reads flags only for created nodes, so it observes
// the same values without a serial clear of up to 2047 entries.

kernel void tree_driver_init_prepare_parallel(device TreeDriverState* state [[buffer(0)]],
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
  flags[0] = 0;
  tree_driver_fused_prepare(local_state, split, histogram, partition, parent_count, plan_histogram_args, histogram_args, plan_partition_args, partition_args, partition_scan_args, evaluate_args, select_args, subtract_args, zero_bytes, zero_args, p);
  state[0] = local_state;
}

kernel void tree_driver_apply_prepare_parallel(device TreeDriverState* state [[buffer(0)]],
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
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]],
    uint groups [[simdgroups_per_threadgroup]]) {
  // Bit 31 marks an accepted slot; the low bits hold its exclusive rank.
  threadgroup uint ranks[kTreeDriverSlots];
  threadgroup uint totals[32];
  // Smallest 2 * slot + (capacity error), or all ones without an error.
  threadgroup atomic_uint first_error;
  TreeDriverState local_state = state[0];
  uint frontier = local_state.frontier_count, depth = local_state.depth, created = local_state.node_count;
  // The host bounds every frontier by the slot capacity; never index past it.
  bool overflow = !local_state.error && frontier > kTreeDriverSlots;
  bool active = !local_state.error && !overflow;
  if (!tid) atomic_store_explicit(&first_error, 0xffffffffu, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint accepted = 0;
  for (uint base = 0; active && base < frontier; base += threads) {
    uint slot = base + tid;
    bool expand = false;
    if (slot < frontier) {
      // Match Driver::Pop, including gain equality at min_split_loss.
      float gain = best[slot].gain;
      expand = best[slot].feature >= 0 && gain > 1.0e-6f &&
               fp32_greater_equal(gain, p.min_gain) && depth < p.max_depth;
    }
    uint rank = simd_prefix_exclusive_sum(uint(expand));
    uint sum = simd_sum(uint(expand));
    if (!lane) totals[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint chunk = 0;
    rank += accepted;
    for (uint i = 0; i < groups; ++i) {
      uint total = totals[i];
      if (i < group) rank += total;
      chunk += total;
    }
    if (slot < frontier) {
      ranks[slot] = rank | (expand ? 0x80000000u : 0u);
      // The serial loop's `created` value when it reaches this slot.
      uint id = ids[slot], position = created + 2 * rank, error = 0xffffffffu;
      if (id >= position || id >= p.node_capacity) error = 2 * slot;
      else if (expand && (position > p.node_capacity || p.node_capacity - position < 2)) error = 2 * slot + 1;
      if (error != 0xffffffffu) atomic_fetch_min_explicit(&first_error, error, memory_order_relaxed);
    }
    accepted += chunk;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint error = atomic_load_explicit(&first_error, memory_order_relaxed);
  bool terminal = depth + 1 >= p.max_depth;
  for (uint slot = tid; active && slot < frontier && slot <= error / 2; slot += threads) {
    uint id = ids[slot];
    if (slot == error / 2) {
      if (error & 1) {
        history[id] = best[slot];
        flags[id] = kTreeDriverEvaluated;
      }
      break;
    }
    uint packed = ranks[slot], rank = packed & 0x7fffffffu;
    bool expand = packed >> 31;
    Split split = best[slot];
    history[id] = split;
    flags[id] = expand ? kTreeDriverEvaluated | kTreeDriverExpanded : kTreeDriverEvaluated;
    if (!expand) continue;
    uint left = created + 2 * rank;
    flags[left] = 0;
    flags[left + 1] = 0;
    // Terminal children are created but never evaluated. Their inherited
    // covers, leaves and base weights must survive the host replay untouched.
    if (terminal) continue;
    uint child = 2 * rank;
    next_ids[child] = left;
    next_ids[child + 1] = left + 1;
    next_sums[child] = split.left_sum;
    next_sums[child + 1] = split.right_sum;
    selected_splits[rank] = split;
    selected_counts[rank] = counts[slot];
    selected_offsets[rank] = offsets[slot];
    selected_children[rank] = int2(int(child), int(child + 1));
    // Retain the source slot before compacting selected parents. Subtraction
    // reads the previous level's original histogram layout.
    parent_histograms[rank] = slot;
  }
  // Every thread has read the incoming state before thread zero replaces it.
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (tid) return;
  // Always retire the current frontier, even if the tree has already stopped.
  local_state.frontier_count = 0;
  local_state.depth = depth + 1;
  if (overflow) local_state.error = kTreeDriverCapacityError;
  else if (active && error != 0xffffffffu)
    local_state.error = error & 1 ? kTreeDriverCapacityError : kTreeDriverNodeError;
  else if (active) {
    local_state.node_count = created + 2 * accepted;
    local_state.frontier_count = terminal ? 0 : 2 * accepted;
  }
  tree_driver_fused_prepare(local_state, split, histogram, partition, parent_count, plan_histogram_args, histogram_args, plan_partition_args, partition_args, partition_scan_args, evaluate_args, select_args, subtract_args, zero_bytes, zero_args, p);
  state[0] = local_state;
}

// Applied parent statistics are read only by the host replay and the
// asynchronous node emission of expanded nodes, which runs after this pass. Deferred selection leaves them unset; this pass then applies the
// same helpers to each expanded node's recorded split (whose integer child
// sums select copied unchanged), once per tree and in parallel, instead of
// one dependent binary64 division chain per level. GPU-driver bounds are
// uniformly unbounded (the host fills every slot with -inf/+inf), so slot
// zero's bounds equal the ones each selection read.
kernel void tree_driver_applied_statistics(device const TreeDriverState* state [[buffer(0)]],
                                           device Split* history [[buffer(1)]],
                                           device const uint* flags [[buffer(2)]],
                                           device const float2* scales [[buffer(3)]],
                                           device const float2* bounds [[buffer(4)]],
                                           constant TreeDriverParams& p [[buffer(5)]],
                                           uint id [[thread_position_in_grid]]) {
  TreeDriverState result = state[0];
  if (result.error || id >= result.node_count || id >= p.node_capacity) return;
  if (!id && (p.control_flags & kTreeDriverDenseRoot)) return;
  if (!(flags[id] & kTreeDriverExpanded)) return;
  Split split = history[id];
  auto sums = split_parent_sums(long2(split.left_sum.g, split.left_sum.h),
                                long2(split.right_sum.g, split.right_sum.h), scales[p.split.group]);
  history[id].applied_parent_weight = split_parent_weight(sums, p.split.reg_alpha, p.split.reg_lambda,
                                                          p.split.max_delta_step, bounds[0]);
  history[id].applied_parent_hessian = cox_float_result(sums.y);
}
