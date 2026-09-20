// CUDA's separate sort/scan stages motivate separate scheduling here. Arbitrary
// weights still require the CPU's tie permutation and ordered float32 sums.
// Detection is parallel; repair has a uniform batch owner; entry formation and
// pruning keep the original per-thread mapping. Counts hold repair flags until
// the final dispatch. All three passes share the existing command and buffers.
kernel void sketch_weighted_detect(device const float2* input [[buffer(0)]],
                                   device const SketchBatch* batches [[buffer(1)]],
                                   device uint* flags [[buffer(2)]],
                                   constant SketchParams& p [[buffer(3)]],
                                   uint batch [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  if (batch >= p.tasks) return;
  auto task = batches[batch];
  auto source = input + task.offset;
  uint repair = 0;
  for (uint i = tid + 1; i < task.size; i += 256) {
    float2 left = source[i - 1], right = source[i];
    repair |= fp32_equal(left.x, right.x) &&
              (as_type<uint>(left.x) != as_type<uint>(right.x) ||
               (!task.reserved && !fp32_equal(left.y, right.y)));
  }
  threadgroup uint scan[9];
  sketch_prefix_count(repair, tid, scan);
  if (!tid) flags[batch] = scan[8] != 0;
}

// One thread per group makes the batch index uniform to the compiler while
// retaining the complete original queue and unchanged libc++ tie sort.
kernel void sketch_weighted_repair(device float2* input [[buffer(0)]],
                                   device const SketchBatch* batches [[buffer(1)]],
                                   device const uint* flags [[buffer(2)]],
                                   device const float2* queue [[buffer(3)]],
                                   constant SketchParams& p [[buffer(4)]],
                                   uint batch [[threadgroup_position_in_grid]]) {
  if (batch >= p.tasks || !flags[batch]) return;
  auto task = batches[batch];
  auto source = input + task.offset;
  for (uint i = 0; i < task.size; ++i) source[i] = queue[task.input + i];
  sketch_tie_sort(source, task.size);
}

// The 64-thread mapping is deliberate: giving all formation and pruning work
// to uniform owners regressed the earlier million-row weighted control.
kernel void sketch_weighted_form(device const float2* input [[buffer(0)]],
                                 device const SketchBatch* batches [[buffer(1)]],
                                 device float4* summaries [[buffer(2)]],
                                 device uint* sizes [[buffer(3)]],
                                 constant SketchParams& p [[buffer(4)]],
                                 uint batch [[thread_position_in_grid]]) {
  if (batch >= p.tasks) return;
  auto task = batches[batch];
  auto source = input + task.offset;
  auto output = summaries + task.offset;
  uint size = 0;
  float total = 0;
  for (uint i = 0; i < task.size;) {
    float2 value = source[i];
    uint j = i + 1;
    while (j < task.size && fp32_equal(source[j].x, value.x))
      value.y = sketch_add(value.y, source[j++].y);
    output[size++] = float4(total, sketch_add(total, value.y), value.y, value.x);
    total = sketch_add(total, value.y);
    i = j;
  }
  sizes[batch] = sketch_prune_serial(output, size, task.limit);
}
