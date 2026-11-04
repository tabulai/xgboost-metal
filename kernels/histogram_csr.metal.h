// Native Metal 3.1 Sparse-Bin CSR Histograms, CSR Partitioning, CSR Prediction,
// CSR TreeSHAP / Saabas, and CSC/CSR Linear Coordinate Descent.

struct CsrRow {
  device const uint* col_idx;
  device const float* val;
  uint begin, end;
  float get(uint feature) const {
    uint lo = begin, hi = end;
    while (lo < hi) {
      uint mid = lo + ((hi - lo) >> 1);
      uint col = col_idx[mid];
      if (col < feature) lo = mid + 1;
      else if (col > feature) hi = mid;
      else return val[mid];
    }
    return NAN;
  }
};

// Threadgroup-cooperative flattened CSR histogram accumulation + exact node-total reduction.
kernel void histogram_csr_accumulate(device const uint* csr_row_ptr [[buffer(0)]],
                                     device const uint* csr_col_idx [[buffer(1)]],
                                     device const ushort* csr_bins [[buffer(2)]],
                                     device const uint* row_ids [[buffer(3)]],
                                     device const long2* gradients [[buffer(4)]],
                                     device const HistTask* tasks [[buffer(5)]],
                                     device atomic_uint* hist [[buffer(6)]],
                                     device atomic_uint* node_totals [[buffer(7)]],
                                     constant HistParams& p [[buffer(8)]],
                                     uint2 block [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint s_row_off[257];
  threadgroup uint s_row_begin[256];
  threadgroup uint4 s_grads[256];
  threadgroup uint s_warp_nnz[8];

  HistTask task = tasks[block.x];
  uint node = task.node & ~kHistogramSingleOwnerTag;
  uint target = block.y;
  uint grad_group = p.padding1 > 1 ? target : p.group;
  ulong stride = ulong(p.bins) + 1;
  ulong hist_base = ((ulong(node) * p.padding1 + target) * p.cols) * stride * 4;

  ulong2 thread_node_sum = ulong2(0);

  for (uint chunk = task.begin; chunk < task.end; chunk += 256) {
    uint chunk_rows = min(256u, task.end - chunk);
    uint row_nnz = 0;
    if (tid < chunk_rows) {
      uint row = row_ids[chunk + tid];
      long2 g = gradients[ulong(row) * p.groups + grad_group];
      thread_node_sum += as_type<ulong2>(g);
      uint rb = csr_row_ptr[row], re = csr_row_ptr[row + 1];
      row_nnz = re - rb;
      s_row_begin[tid] = rb;
      s_grads[tid] = as_type<uint4>(g);
    } else {
      s_row_begin[tid] = 0;
      s_grads[tid] = uint4(0);
    }

    uint warp_excl = simd_prefix_exclusive_sum(row_nnz);
    uint warp_tot = simd_sum(row_nnz);
    if ((tid & 31u) == 0u) s_warp_nnz[tid >> 5] = warp_tot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_excl = 0, total_nnz = 0;
    for (uint w = 0; w < 8; ++w) {
      if (w < (tid >> 5)) block_excl += s_warp_nnz[w];
      total_nnz += s_warp_nnz[w];
    }
    s_row_off[tid] = block_excl + warp_excl;
    if (tid == 0) s_row_off[256] = total_nnz;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < total_nnz; k += 256) {
      // Binary search for largest r_local in [0, chunk_rows - 1] with s_row_off[r_local] <= k.
      uint lo = 0, hi = chunk_rows;
      while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (s_row_off[mid] <= k) lo = mid;
        else hi = mid;
      }
      uint e = s_row_begin[lo] + (k - s_row_off[lo]);
      uint feature = csr_col_idx[e];
      uint bin = load_bin(csr_bins, e, p.bins);
      if (bin != 0xffffffffu) {
        long2 grad = as_type<long2>(s_grads[lo]);
        ulong dst = hist_base + (ulong(feature) * stride + bin) * 4;
        add64(hist + dst, grad.x);
        add64(hist + dst + 2, grad.y);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Exact modulo-2^64 reduction of thread_node_sum across the 256 threads in this tile.
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    uint4 other = simd_shuffle_xor(as_type<uint4>(thread_node_sum), mask);
    thread_node_sum += as_type<ulong2>(other);
  }
  if ((tid & 31u) == 0u) s_grads[tid >> 5] = as_type<uint4>(thread_node_sum);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    ulong2 tile_total = ulong2(0);
    for (uint w = 0; w < 8; ++w) tile_total += as_type<ulong2>(s_grads[w]);
    long2 signed_total = as_type<long2>(tile_total);
    ulong tot_dst = (ulong(node) * p.padding1 + target) * 4;
    add64(node_totals + tot_dst, signed_total.x);
    add64(node_totals + tot_dst + 2, signed_total.y);
  }
}

// Compute exact missing-bin sum: hist[p.bins] = node_total - sum_{b=0..count-1} hist[b] (mod 2^64).
kernel void histogram_csr_finalize_missing(device uint* hist [[buffer(0)]],
                                           device const uint* node_totals [[buffer(1)]],
                                           device const uint* cut_counts [[buffer(2)]],
                                           device const uint* node_counts [[buffer(3)]],
                                           constant SparseParams& p [[buffer(4)]],
                                           uint3 block [[threadgroup_position_in_grid]],
                                           uint lane [[thread_index_in_threadgroup]]) {
  uint feature = block.x, node = block.y, target = block.z;
  if (feature >= p.cols) return;
  uint actual_node = p.flags ? (p.flags - 1u) : node;
  if (p.subtract) {
    uint first = actual_node & ~1u;
    uint small = node_counts[first] <= node_counts[first + 1] ? first : first + 1;
    if (actual_node != small) return;
  }
  ulong stride = ulong(p.bins) + 1;
  ulong feat_base = ((ulong(actual_node) * p.targets + target) * p.cols + feature) * stride * 4;
  uint count = cut_counts[feature];
  ulong2 present = ulong2(0);
  for (uint b = lane; b < count; b += 32) {
    present += as_type<ulong2>(load_hist_int(hist, feat_base + ulong(b) * 4));
  }
  for (ushort mask = 16; mask > 0; mask >>= 1) {
    uint4 other = simd_shuffle_xor(as_type<uint4>(present), mask);
    present += as_type<ulong2>(other);
  }
  if (lane == 0) {
    ulong2 total = as_type<ulong2>(load_hist_int(node_totals, (ulong(actual_node) * p.targets + target) * 4));
    ulong2 missing = total - present;
    ulong m_off = feat_base + ulong(p.bins) * 4;
    hist[m_off] = uint(missing.x);
    hist[m_off + 1] = uint(missing.x >> 32);
    hist[m_off + 2] = uint(missing.y);
    hist[m_off + 3] = uint(missing.y >> 32);
  }
}

// Sparse CSR Partition Kernels (binary search in csr_col_idx per row).
kernel void partition_two_ended_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                    device const uint* csr_col_idx [[buffer(1)]],
                                    device const float* csr_val [[buffer(2)]],
                                    device const uint* rows [[buffer(3)]],
                                    device const Split* splits [[buffer(4)]],
                                    device const PartitionTask* tasks [[buffer(5)]],
                                    device const PartitionSegment* segments [[buffer(6)]],
                                    device atomic_uint* cursors [[buffer(7)]],
                                    device uint* output [[buffer(8)]],
                                    device const uint* categories [[buffer(9)]],
                                    constant PartitionParams& p [[buffer(10)]],
                                    uint block [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[8], bases[2];
  PartitionTask task = tasks[block];
  uint index = task.begin + tid, row = 0;
  bool valid = index < task.end, left = false;
  if (valid) {
    Split split = splits[task.node];
    row = rows[index];
    CsrRow r{csr_col_idx, csr_val, csr_row_ptr[row], csr_row_ptr[row + 1]};
    float value = r.get(uint(split.feature));
    left = split_left(value, split.threshold, split.default_left, split.split_type,
                      split.category_begin, split.category_count, categories);
  }
  uint prefix = simd_prefix_exclusive_sum(uint(left));
  uint sum = simd_sum(uint(left));
  if ((tid & 31) == 0) sums[tid / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint total = 0;
  for (uint i = 0; i < 8; ++i) {
    if (i < tid / 32) prefix += sums[i];
    total += sums[i];
  }
  PartitionSegment segment = segments[task.segment];
  uint right = task.end - task.begin - total;
  if (tid == 0) {
    bases[0] = total ? atomic_fetch_add_explicit(cursors + segment.left, total, memory_order_relaxed) : 0;
    bases[1] = right ? atomic_fetch_add_explicit(cursors + segment.right, right, memory_order_relaxed) : 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!valid) return;
  output[left ? segment.begin + bases[0] + prefix
              : segment.end - bases[1] - right + (tid - prefix)] = row;
}

kernel void partition_segment_flags_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                        device const uint* csr_col_idx [[buffer(1)]],
                                        device const float* csr_val [[buffer(2)]],
                                        device const uint* rows [[buffer(3)]],
                                        device const Split* splits [[buffer(4)]],
                                        device const PartitionTask* tasks [[buffer(5)]],
                                        device uint* ranks [[buffer(6)]],
                                        device uint* counts [[buffer(7)]],
                                        device const uint* categories [[buffer(8)]],
                                        constant PartitionParams& p [[buffer(9)]],
                                        uint block [[threadgroup_position_in_grid]],
                                        uint tid [[thread_index_in_threadgroup]]) {
  threadgroup uint sums[8];
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  bool left = false;
  if (index < task.end) {
    Split split = splits[task.node];
    uint row = rows[index];
    CsrRow r{csr_col_idx, csr_val, csr_row_ptr[row], csr_row_ptr[row + 1]};
    float value = r.get(uint(split.feature));
    left = split_left(value, split.threshold, split.default_left, split.split_type,
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

kernel void partition_atomic_count_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                       device const uint* csr_col_idx [[buffer(1)]],
                                       device const float* csr_val [[buffer(2)]],
                                       device const uint* rows [[buffer(3)]],
                                       device const Split* splits [[buffer(4)]],
                                       device const PartitionTask* tasks [[buffer(5)]],
                                       device uint* ranks [[buffer(6)]],
                                       device atomic_uint* counts [[buffer(7)]],
                                       device const uint* categories [[buffer(8)]],
                                       constant PartitionParams& p [[buffer(9)]],
                                       uint block [[threadgroup_position_in_grid]],
                                       uint tid [[thread_index_in_threadgroup]]) {
  PartitionTask task = tasks[block];
  uint index = task.begin + tid;
  bool valid = index < task.end, left = false;
  if (valid) {
    Split split = splits[task.node];
    uint row = rows[index];
    CsrRow r{csr_col_idx, csr_val, csr_row_ptr[row], csr_row_ptr[row + 1]};
    float value = r.get(uint(split.feature));
    left = split_left(value, split.threshold, split.default_left, split.split_type,
                      split.category_begin, split.category_count, categories);
    ranks[index] = uint(left);
  }
  if (p.padding) {
    uint sum = simd_sum(uint(valid && left));
    if (simd_is_first() && sum) atomic_fetch_add_explicit(counts + 2 * task.segment, sum, memory_order_relaxed);
  } else if (valid && left) atomic_fetch_add_explicit(counts + 2 * task.segment, 1, memory_order_relaxed);
}

kernel void vector_leaf_sums_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                 device const uint* csr_col_idx [[buffer(1)]],
                                 device const float* csr_val [[buffer(2)]],
                                 device const Node* nodes [[buffer(3)]],
                                 device const uint* categories [[buffer(4)]],
                                 device const long2* gradient [[buffer(5)]],
                                 device atomic_uint* sums [[buffer(6)]],
                                 device const float* category_bounds [[buffer(7)]],
                                 constant VectorLeafParams& p [[buffer(8)]],
                                 uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.targets) return;
  CsrRow r{csr_col_idx, csr_val, csr_row_ptr[id.x], csr_row_ptr[id.x + 1]};
  uint index = 0;
  Node node = nodes[index];
  while (node.left >= 0) {
    float value = r.get(uint(node.feature));
    bool numeric_category = node.split_type && p.padding;
    bool left = split_left(value, numeric_category ? category_bounds[node.feature] : node.threshold,
                           node.default_left, numeric_category ? 0 : node.split_type,
                           node.category_begin, node.category_count, categories);
    index = left ? uint(node.left) : uint(node.right);
    node = nodes[index];
  }
  auto pair = gradient[ulong(id.x) * p.targets + id.y];
  auto offset = (ulong(index) * p.targets + id.y) * 4;
  add64(sums + offset, pair.x);
  add64(sums + offset + 2, pair.y);
}

// Sparse CSR Prediction, Leaf Index, TreeSHAP & Approximate Contributions.
kernel void predict_csr(device const uint* csr_row_ptr [[buffer(0)]],
                        device const uint* csr_col_idx [[buffer(1)]],
                        device const float* csr_val [[buffer(2)]],
                        device const Node* nodes [[buffer(3)]],
                        device const TreeMeta* trees [[buffer(4)]],
                        device const float* base_score [[buffer(5)]],
                        device const float* base_margin [[buffer(6)]],
                        device float* output [[buffer(7)]],
                        device const uint* categories [[buffer(8)]],
                        device const float* weights [[buffer(9)]],
                        device const float* leaf_weights [[buffer(10)]],
                        device const float* category_bounds [[buffer(11)]],
                        constant PredictParams& p [[buffer(12)]],
                        uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  ulong idx = ulong(id.x) * p.groups + id.y;
  float sum = p.has_margin ? base_margin[ulong(id.x) * p.margin_groups + id.y] : base_score[id.y];
  CsrRow row{csr_col_idx, csr_val, csr_row_ptr[id.x], csr_row_ptr[id.x + 1]};
  for (uint t = p.tree_begin; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if ((tree.outputs == 1 && tree.group != id.y) || tree.round < p.begin || tree.round >= p.end) continue;
    uint node = leaf_index(row, nodes, tree.begin, categories, p.bin_partition ? category_bounds : nullptr);
    float leaf = tree.outputs > 1 ? leaf_weights[tree.leaf_begin + ulong(nodes[tree.begin + node].right) * tree.outputs + id.y]
                                  : nodes[tree.begin + node].leaf;
    sum += leaf * (p.has_tree_weights ? weights[t] : 1.0f);
  }
  output[idx] = sum;
}

kernel void predict_csr_tree_cached(device const uint* csr_row_ptr [[buffer(0)]],
                                    device const uint* csr_col_idx [[buffer(1)]],
                                    device const float* csr_val [[buffer(2)]],
                                    device const Node* nodes [[buffer(3)]],
                                    device const TreeMeta* trees [[buffer(4)]],
                                    device const float* base_score [[buffer(5)]],
                                    device const float* base_margin [[buffer(6)]],
                                    device float* output [[buffer(7)]],
                                    device const uint* categories [[buffer(8)]],
                                    device const float* weights [[buffer(9)]],
                                    device const float* leaf_weights [[buffer(10)]],
                                    device const float* category_bounds [[buffer(11)]],
                                    constant PredictParams& p [[buffer(12)]],
                                    uint2 block [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup Node s_nodes[256];
  uint row_id = block.x * 256 + tid, group = block.y;
  if (group >= p.groups) return;
  bool valid = row_id < p.rows;
  float sum = valid ? (p.has_margin ? base_margin[ulong(row_id) * p.margin_groups + group] : base_score[group]) : 0.0f;
  CsrRow row{csr_col_idx, csr_val, valid ? csr_row_ptr[row_id] : 0u, valid ? csr_row_ptr[row_id + 1] : 0u};
  device const float* cat_bounds = p.bin_partition ? category_bounds : nullptr;
  for (uint t = p.tree_begin; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if ((tree.outputs == 1 && tree.group != group) || tree.round < p.begin || tree.round >= p.end) continue;
    float tw = p.has_tree_weights ? weights[t] : 1.0f;
    if (tree.count <= 256) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid < tree.count) s_nodes[tid] = nodes[tree.begin + tid];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (valid) {
        uint node = leaf_index_tg(row, s_nodes, categories, cat_bounds);
        float leaf = tree.outputs > 1 ? leaf_weights[tree.leaf_begin + ulong(s_nodes[node].right) * tree.outputs + group]
                                      : s_nodes[node].leaf;
        sum += leaf * tw;
      }
    } else if (valid) {
      uint node = leaf_index(row, nodes, tree.begin, categories, cat_bounds);
      float leaf = tree.outputs > 1 ? leaf_weights[tree.leaf_begin + ulong(nodes[tree.begin + node].right) * tree.outputs + group]
                                    : nodes[tree.begin + node].leaf;
      sum += leaf * tw;
    }
  }
  if (valid) output[ulong(row_id) * p.groups + group] = sum;
}

kernel void predict_leaf_csr(device const uint* csr_row_ptr [[buffer(0)]],
                             device const uint* csr_col_idx [[buffer(1)]],
                             device const float* csr_val [[buffer(2)]],
                             device const Node* nodes [[buffer(3)]],
                             device const TreeMeta* trees [[buffer(4)]],
                             device float* output [[buffer(5)]],
                             device const uint* categories [[buffer(6)]],
                             constant PredictParams& p [[buffer(7)]],
                             uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.trees) return;
  TreeMeta tree = trees[id.y];
  CsrRow row{csr_col_idx, csr_val, csr_row_ptr[id.x], csr_row_ptr[id.x + 1]};
  output[ulong(id.x) * p.trees + id.y] = float(leaf_index(row, nodes, tree.begin, categories));
}

kernel void shap_contributions_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                   device const uint* csr_col_idx [[buffer(1)]],
                                   device const float* csr_val [[buffer(2)]],
                                   device const ShapPath* paths [[buffer(3)]],
                                   device const ShapLeaf* leaves [[buffer(4)]],
                                   device const float* base [[buffer(5)]],
                                   device const float* margin [[buffer(6)]],
                                   device float* output [[buffer(7)]],
                                   device const uint* categories [[buffer(8)]],
                                   device const float* weights [[buffer(9)]],
                                   constant PredictParams& p [[buffer(10)]],
                                   uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  bool interactions = p.type == 4;
  ulong columns = ulong(p.cols) + 1;
  ulong elements = interactions ? columns * columns : columns;
  ulong offset = (ulong(id.x) * p.groups + id.y) * elements;
  for (ulong i = 0; i < elements; ++i) output[offset + i] = 0;
  float bias = p.has_margin ? margin[ulong(id.x) * p.groups + id.y] : base[id.y];
  CsrRow row{csr_col_idx, csr_val, csr_row_ptr[id.x], csr_row_ptr[id.x + 1]};
  for (uint leaf_id = 0; leaf_id < p.trees; ++leaf_id) {
    ShapLeaf leaf = leaves[leaf_id];
    if (leaf.group != id.y || leaf.round < p.begin || leaf.round >= p.end) continue;
    float weight = p.has_tree_weights ? weights[leaf.tree] : 1.0f;
    bias += leaf.expectation * weight;
    eval_shap_leaf_global(row, leaf, weight, interactions, columns, paths, categories, output + offset, p.bin_partition != 0);
  }
  output[offset + elements - 1] = bias;
}

kernel void shap_contributions_csr_fast(device const uint* csr_row_ptr [[buffer(0)]],
                                        device const uint* csr_col_idx [[buffer(1)]],
                                        device const float* csr_val [[buffer(2)]],
                                        device const ShapPath* paths [[buffer(3)]],
                                        device const ShapLeaf* leaves [[buffer(4)]],
                                        device const float* base [[buffer(5)]],
                                        device const float* margin [[buffer(6)]],
                                        device float* output [[buffer(7)]],
                                        device const uint* categories [[buffer(8)]],
                                        device const float* weights [[buffer(9)]],
                                        constant PredictParams& p [[buffer(10)]],
                                        uint2 block [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float s_x[32][33];
  threadgroup float s_out[32][33];
  uint base_row = block.x * 32, group = block.y;
  if (group >= p.groups || base_row >= p.rows) return;
  uint valid_rows = min(32u, p.rows - base_row);
  if (lane < valid_rows) {
    uint row_id = base_row + lane;
    for (uint c = 0; c < p.cols; ++c) {
      s_x[lane][c] = NAN;
      s_out[lane][c] = 0.0f;
    }
    uint beg = csr_row_ptr[row_id], end = csr_row_ptr[row_id + 1];
    for (uint e = beg; e < end; ++e) {
      uint col = csr_col_idx[e];
      if (col < p.cols) s_x[lane][col] = csr_val[e];
    }
    float bias = p.has_margin ? margin[ulong(row_id) * p.groups + group] : base[group];
    SramRow row_view{s_x[lane]};
    for (uint leaf_id = 0; leaf_id < p.trees; ++leaf_id) {
      ShapLeaf leaf = leaves[leaf_id];
      if (leaf.group != group || leaf.round < p.begin || leaf.round >= p.end) continue;
      float weight = p.has_tree_weights ? weights[leaf.tree] : 1.0f;
      bias += leaf.expectation * weight;
      eval_shap_leaf_sram(row_view, leaf, weight, paths, categories, s_out[lane]);
    }
    s_out[lane][p.cols] = bias;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0; r < valid_rows; ++r) {
    ulong offset = (ulong(base_row + r) * p.groups + group) * (ulong(p.cols) + 1);
    if (lane < p.cols) output[offset + lane] = s_out[r][lane];
    if (lane == 0) output[offset + p.cols] = s_out[r][p.cols];
  }
}

kernel void approximate_contributions_csr(device const uint* csr_row_ptr [[buffer(0)]],
                                          device const uint* csr_col_idx [[buffer(1)]],
                                          device const float* csr_val [[buffer(2)]],
                                          device const Node* nodes [[buffer(3)]],
                                          device const TreeMeta* trees [[buffer(4)]],
                                          device const float* means [[buffer(5)]],
                                          device const float* base [[buffer(6)]],
                                          device const float* margin [[buffer(7)]],
                                          device float* output [[buffer(8)]],
                                          device const uint* categories [[buffer(9)]],
                                          device const float* weights [[buffer(10)]],
                                          constant PredictParams& p [[buffer(11)]],
                                          uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows || id.y >= p.groups) return;
  bool interactions = p.type == 5;
  ulong columns = ulong(p.cols) + 1;
  ulong elements = interactions ? columns * columns : columns;
  ulong offset = (ulong(id.x) * p.groups + id.y) * elements;
  for (ulong i = 0; i < elements; ++i) output[offset + i] = 0;
  float bias = p.has_margin ? margin[ulong(id.x) * p.groups + id.y] : base[id.y];
  CsrRow row{csr_col_idx, csr_val, csr_row_ptr[id.x], csr_row_ptr[id.x + 1]};
  for (uint t = 0; t < p.trees; ++t) {
    TreeMeta tree = trees[t];
    if (tree.group != id.y || tree.round < p.begin || tree.round >= p.end) continue;
    uint index = 0;
    float weight = p.has_tree_weights ? weights[t] : 1.0f;
    bias += means[tree.begin] * weight;
    while (nodes[tree.begin + index].left >= 0) {
      Node node = nodes[tree.begin + index];
      float value = row.get(uint(node.feature));
      bool left = split_left(value, node.threshold, node.default_left, node.split_type,
                             node.category_begin, node.category_count, categories);
      uint next = uint(left ? node.left : node.right);
      ulong dst = interactions ? ulong(node.feature) * columns + uint(node.feature) : uint(node.feature);
      output[offset + dst] += (means[tree.begin + next] - means[tree.begin + index]) * weight;
      index = next;
    }
  }
  output[offset + elements - 1] = bias;
}

// Sparse CSC/CSR gblinear kernels (O(nnz) instead of O(rows * cols)).
kernel void linear_partial_csc(device const uint* csc_col_ptr [[buffer(0)]],
                               device const uint* csc_row_idx [[buffer(1)]],
                               device const float* csc_val [[buffer(2)]],
                               device const Pair* gradient [[buffer(3)]],
                               device SurvivalSum* partial [[buffer(4)]],
                               device const uint* feature_order [[buffer(5)]],
                               constant LinearParams& p [[buffer(6)]],
                               uint2 block [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
  #pragma clang fp contract(off)
  threadgroup SurvivalSum scratch[256];
  uint feature = (p.all_features & 1u) ? block.y : ((p.all_features & 2u) ? feature_order[ulong(p.group) * p.cols + p.feature] : p.feature);
  uint tile_rows = (p.has_weight >> 1) ? (p.has_weight >> 1) : kLinearRowsPerTile;
  bool bias = feature == p.cols;
  auto sum = survival_zero();
  if (bias) {
    ulong end = min(ulong(p.rows), (ulong(block.x) + 1) * tile_rows);
    for (ulong row = ulong(block.x) * tile_rows + tid; row < end; row += 256) {
      auto pair = gradient[row * p.groups + p.group];
      if (pair.h >= 0) sum = sum + SurvivalSum{sr(pair.g), sr(pair.h)};
    }
  } else {
    uint col_begin = csc_col_ptr[feature], col_end = csc_col_ptr[feature + 1];
    uint col_len = col_end - col_begin;
    ulong end = min(ulong(col_len), (ulong(block.x) + 1) * tile_rows);
    for (ulong k = ulong(block.x) * tile_rows + tid; k < end; k += 256) {
      uint row = csc_row_idx[col_begin + k];
      float value = csc_val[col_begin + k];
      auto pair = gradient[ulong(row) * p.groups + p.group];
      if (isnan(value) || pair.h < 0) continue;
      float g = pair.g * value, h = (pair.h * value) * value;
      sum = sum + SurvivalSum{sr(g), sr(h)};
    }
  }
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) partial[ulong(block.y) * p.tiles + block.x] = total;
}

kernel void linear_residual_csc(device const uint* csc_col_ptr [[buffer(0)]],
                                device const uint* csc_row_idx [[buffer(1)]],
                                device const float* csc_val [[buffer(2)]],
                                device Pair* gradient [[buffer(3)]],
                                device const LinearUpdate* update [[buffer(4)]],
                                constant LinearParams& p [[buffer(5)]],
                                uint idx [[thread_position_in_grid]]) {
  #pragma clang fp contract(off)
  if (update->value == 0) return;
  if (update->feature == p.cols) {
    if (idx >= p.rows) return;
    ulong index = ulong(idx) * p.groups + p.group;
    auto pair = gradient[index];
    if (pair.h < 0) return;
    gradient[index].g = pair.g + pair.h * update->value;
  } else {
    uint begin = csc_col_ptr[update->feature], end = csc_col_ptr[update->feature + 1];
    if (idx >= end - begin) return;
    uint row = csc_row_idx[begin + idx];
    float value = csc_val[begin + idx];
    if (isnan(value)) return;
    ulong index = ulong(row) * p.groups + p.group;
    auto pair = gradient[index];
    if (pair.h < 0) return;
    gradient[index].g = pair.g + (pair.h * value) * update->value;
  }
}

kernel void linear_coord_fused_csc(device const uint* csc_col_ptr [[buffer(0)]],
                                   device const uint* csc_row_idx [[buffer(1)]],
                                   device const float* csc_val [[buffer(2)]],
                                   device Pair* gradient [[buffer(3)]],
                                   device float* weights [[buffer(4)]],
                                   device const SurvivalReal* mass [[buffer(5)]],
                                   device const uint* feature_order [[buffer(6)]],
                                   constant LinearParams& p [[buffer(7)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]]) {
  #pragma clang fp contract(off)
  threadgroup SurvivalSum scratch[256];
  threadgroup float s_dw;
  if (group >= p.groups) return;
  float alpha = extended_result(sr(p.reg_alpha) * mass[0]);
  float lambda = extended_result(sr(p.reg_lambda) * mass[0]);
  uint steps = p.feature;
  for (uint step = 0; step < steps; ++step) {
    uint feature = feature_order[ulong(group) * p.cols + step];
    uint begin = csc_col_ptr[feature], end = csc_col_ptr[feature + 1];
    auto sum = survival_zero();
    for (uint e = begin + tid; e < end; e += 256) {
      float value = csc_val[e];
      uint row = csc_row_idx[e];
      auto pair = gradient[ulong(row) * p.groups + group];
      if (isnan(value) || pair.h < 0) continue;
      float g = pair.g * value, h = (pair.h * value) * value;
      sum = sum + SurvivalSum{sr(g), sr(h)};
    }
    auto total = survival_block_reduce(sum, scratch, tid);
    if (!tid) {
      ulong w_idx = ulong(feature) * p.groups + group;
      auto delta = linear_delta(total, weights[w_idx], alpha, lambda);
      float dw = extended_result(sr(p.learning_rate) * delta);
      weights[w_idx] += dw;
      s_dw = dw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dw = s_dw;
    if (dw != 0) {
      for (uint e = begin + tid; e < end; e += 256) {
        float value = csc_val[e];
        if (isnan(value)) continue;
        ulong idx = ulong(csc_row_idx[e]) * p.groups + group;
        auto pair = gradient[idx];
        if (pair.h < 0) continue;
        gradient[idx].g = pair.g + (pair.h * value) * dw;
      }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

kernel void linear_predict_csr(device const uint* csr_row_ptr [[buffer(0)]],
                               device const uint* csr_col_idx [[buffer(1)]],
                               device const float* csr_val [[buffer(2)]],
                               device const float* weights [[buffer(3)]],
                               device const float* base [[buffer(4)]],
                               device const float* margins [[buffer(5)]],
                               device float* output [[buffer(6)]],
                               constant PredictParams& p [[buffer(7)]],
                               uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows) return;
  ulong index = ulong(id.x) * p.groups + id.y;
  float bias = weights[ulong(p.trees) * p.groups + id.y] + (p.has_margin ? margins[index] : base[0]);
  uint begin = csr_row_ptr[id.x], end = csr_row_ptr[id.x + 1];
  if (p.type == 2 || p.type == 3) {
    ulong base_out = index * (ulong(p.trees) + 1);
    for (uint f = 0; f < p.trees; ++f) output[base_out + f] = 0;
    for (uint e = begin; e < end; ++e) {
      uint f = csr_col_idx[e];
      if (f < p.trees) {
        float value = csr_val[e];
        if (!isnan(value)) output[base_out + f] = value * weights[ulong(f) * p.groups + id.y];
      }
    }
    output[base_out + p.trees] = bias;
  } else {
    float sum = bias;
    for (uint e = begin; e < end; ++e) {
      uint f = csr_col_idx[e];
      if (f < p.cols) {
        float value = csr_val[e];
        if (!isnan(value)) sum = fma(value, weights[ulong(f) * p.groups + id.y], sum);
      }
    }
    output[index] = sum;
  }
}
