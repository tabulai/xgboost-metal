// CUDA-inspired batch scans, restricted to exact integer ranks. Implicit unit
// weights and at most 2^24 retained elements make every group sum and prefix
// exactly representable in float32. Other ranks and mixed zero signs retain
// ordered handling. All 256 threads participate in each prefix and barrier.
//
// Group heads/tails write disjoint scalar rank fields before weight formation.
// Pruning reuses both float2 sort buffers, keeping selected ranks and values
// separate until the final gather. A single float2 buffer would be too small
// when the batch is only slightly larger than its float4 summary capacity.
// The default (p.reserved) marks ordered work with UINT_MAX in the existing
// size array. A following 64-thread dispatch consumes those marks before the
// feature reducer reads counts. Diagnostic forcing retains the one-pass path.
kernel void sketch_summarize_parallel(device float2* input [[buffer(0)]],
                                    device const SketchBatch* batches [[buffer(1)]],
                                    device float4* summaries [[buffer(2)]],
                                    device uint* sizes [[buffer(3)]],
                                    device const float2* queue [[buffer(4)]],
                                    device float2* scratch [[buffer(5)]],
                                    constant SketchParams& p [[buffer(6)]],
                                    uint batch [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  if (batch >= p.tasks) return;
  auto task = batches[batch];
  auto source = input + task.offset;
  auto output = summaries + task.offset;
  if (!task.reserved) {
    if (!tid) sizes[batch] = p.reserved ? 0xffffffffu : sketch_summarize_ordered(source, task, output, queue);
    return;
  }
  uint chunk = (task.size + 255) / 256;
  uint begin = min(tid * chunk, task.size), end = min(begin + chunk, task.size);
  uint repair = 0, weight = 0, heads = 0;
  for (uint i = begin; i < end; ++i) {
    float2 value = source[i];
    bool head = !i || !fp32_equal(value.x, source[i - 1].x);
    if (!head) repair |= as_type<uint>(value.x) != as_type<uint>(source[i - 1].x);
    heads += head;
    weight += uint(value.y);
  }
  threadgroup uint scan[9];
  sketch_prefix_count(repair, tid, scan);
  if (scan[8]) {
    if (!tid) sizes[batch] = p.reserved ? 0xffffffffu : sketch_summarize_ordered(source, task, output, queue);
    return;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint rank = sketch_prefix_count(weight, tid, scan);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint group = sketch_prefix_count(heads, tid, scan), size = scan[8];
  // Run count is only an upper bound on distinct values. Keep actual
  // full-retention batches on the old mapping, even inside a large plan.
  if (p.reserved && size <= task.limit) {
    if (!tid) sizes[batch] = 0xffffffffu;
    return;
  }
  auto fields = reinterpret_cast<device float*>(output);
  for (uint i = begin; i < end; ++i) {
    float2 value = source[i];
    if (!i || !fp32_equal(value.x, source[i - 1].x)) {
      fields[ulong(group) * 4] = float(rank);
      fields[ulong(group) * 4 + 3] = value.x;
      ++group;
    }
    rank += uint(value.y);
    if (i + 1 == task.size || !fp32_equal(value.x, source[i + 1].x))
      fields[ulong(group - 1) * 4 + 1] = float(rank);
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  for (uint i = tid; i < size; i += 256)
    fields[ulong(i) * 4 + 2] = float(uint(fields[ulong(i) * 4 + 1]) - uint(fields[ulong(i) * 4]));
  threadgroup_barrier(mem_flags::mem_device);
  if (size <= task.limit) {
    if (!tid) sizes[batch] = size;
    return;
  }
  if (task.limit < 2) {
    if (!tid) sizes[batch] = task.limit;
    return;
  }
  auto ranks = source;
  auto values = scratch + task.offset;
  for (uint i = tid; i < task.limit; i += 256) {
    float4 entry = output[sketch_prune_index(output, size, task.limit, i)];
    ranks[i] = entry.xy;
    values[i] = entry.zw;
  }
  threadgroup_barrier(mem_flags::mem_device);
  uint prune_chunk = (task.limit + 255) / 256;
  uint prune_begin = min(tid * prune_chunk, task.limit), prune_end = min(prune_begin + prune_chunk, task.limit);
  uint count = 0;
  for (uint i = prune_begin; i < prune_end; ++i)
    count += !i || !fp32_equal(values[i].y, values[i - 1].y);
  uint position = sketch_prefix_count(count, tid, scan);
  for (uint i = prune_begin; i < prune_end; ++i)
    if (!i || !fp32_equal(values[i].y, values[i - 1].y)) output[position++] = float4(ranks[i], values[i]);
  if (!tid) sizes[batch] = scan[8];
}
