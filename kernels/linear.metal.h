// Coordinate descent follows the CUDA updater's resident column/residual layout.
// The CPU reference sums float32 products in double. Extended GPU sums retain that
// range and extra precision instead of inheriting CUDA's float32 reductions.
kernel void linear_columns(device const float* data [[buffer(0)]], device float* columns [[buffer(1)]],
                             constant LinearParams& p [[buffer(2)]], uint2 block [[threadgroup_position_in_grid]],
                             uint2 lane [[thread_position_in_threadgroup]]) {
  threadgroup float tile[32][33];
  uint row = block.x * 32 + lane.y, feature = block.y * 32 + lane.x;
  for (uint j = 0; j < 32; j += 8)
    tile[lane.y + j][lane.x] = row + j < p.rows && feature < p.cols ? data[ulong(row + j) * p.cols + feature] : NAN;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row = block.x * 32 + lane.x; feature = block.y * 32 + lane.y;
  for (uint j = 0; j < 32; j += 8)
    if (row < p.rows && feature + j < p.cols) columns[ulong(feature + j) * p.rows + row] = tile[lane.x][lane.y + j];
}

inline SurvivalReal linear_delta(SurvivalSum sum, float weight, float alpha, float lambda) {
  if (survival_less(sum.weight, sr(1.0e-5f))) return sr(0);
  auto gradient = sum.value + sr(lambda) * sr(weight);
  auto hessian = sum.weight + sr(lambda);
  auto tentative = sr(weight) - gradient / hessian;
  auto delta = -(gradient + sr(tentative.significand.x >= 0 ? alpha : -alpha)) / hessian;
  auto limit = sr(-weight);
  return tentative.significand.x >= 0 ? (survival_less(delta, limit) ? limit : delta)
                                      : (survival_less(limit, delta) ? limit : delta);
}

kernel void linear_weight_partial(device const float* weights [[buffer(0)]],
                                   device SurvivalSum* partial [[buffer(1)]],
                                   constant LinearParams& p [[buffer(2)]],
                                   uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum scratch[256];
  auto sum = survival_zero();
  ulong end = min(ulong(p.rows), (ulong(block) + 1) * kLinearRowsPerTile);
  for (ulong row = ulong(block) * kLinearRowsPerTile + tid; row < end; row += 256)
    sum.value = sum.value + sr(p.has_weight ? weights[row] : 1.0f);
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) partial[block] = total;
}
kernel void linear_weight_total(device const SurvivalSum* partial [[buffer(0)]],
                                 device SurvivalReal* mass [[buffer(1)]],
                                 constant LinearParams& p [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum scratch[256];
  auto sum = survival_zero();
  for (uint i = tid; i < p.tiles; i += 256) sum = sum + partial[i];
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) mass[0] = total.value;
}
kernel void linear_partial(device const float* data [[buffer(0)]], device const Pair* gradient [[buffer(1)]],
                            device SurvivalSum* partial [[buffer(2)]], constant LinearParams& p [[buffer(3)]],
                            uint2 block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  #pragma clang fp contract(off)
  threadgroup SurvivalSum scratch[256];
  uint feature = p.all_features ? block.y : p.feature;
  bool bias = feature == p.cols;
  auto sum = survival_zero();
  ulong end = min(ulong(p.rows), (ulong(block.x) + 1) * kLinearRowsPerTile);
  for (ulong row = ulong(block.x) * kLinearRowsPerTile + tid; row < end; row += 256) {
    auto pair = gradient[row * p.groups + p.group];
    if (bias) {
      if (pair.h >= 0) sum = sum + SurvivalSum{sr(pair.g), sr(pair.h)};
    } else {
      float value = data[ulong(feature) * p.rows + row];
      if (isnan(value) || pair.h < 0) continue;
      float g = pair.g * value, h = (pair.h * value) * value;
      sum = sum + SurvivalSum{sr(g), sr(h)};
    }
  }
  auto total = survival_block_reduce(sum, scratch, tid);
  if (!tid) partial[ulong(block.y) * p.tiles + block.x] = total;
}
kernel void linear_reduce(device const SurvivalSum* partial [[buffer(0)]], device float* weights [[buffer(1)]],
                           device const SurvivalReal* mass [[buffer(2)]], device SurvivalReal* deltas [[buffer(3)]],
                           device float* scores [[buffer(4)]], device LinearUpdate* update [[buffer(5)]],
                           constant LinearParams& p [[buffer(6)]],
                           uint block [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup SurvivalSum scratch[256];
  auto sum = survival_zero();
  for (uint i = tid; i < p.tiles; i += 256) sum = sum + partial[ulong(block) * p.tiles + i];
  auto total = survival_block_reduce(sum, scratch, tid);
  if (tid) return;
  uint feature = p.all_features ? block : p.feature;
  auto delta = sr(0);
  if (feature == p.cols) {
    delta = -total.value / total.weight;
    if (!isfinite(delta.significand.x)) delta = sr(0);
  } else {
    float alpha = extended_result(sr(p.reg_alpha) * mass[0]);
    float lambda = extended_result(sr(p.reg_lambda) * mass[0]);
    delta = linear_delta(total, weights[ulong(feature) * p.groups + p.group], alpha, lambda);
  }
  if (p.all_features) {
    ulong index = ulong(p.group) * p.cols + feature;
    deltas[index] = delta;
    scores[index] = extended_result(delta);
  } else {
    float dw = extended_result(sr(p.learning_rate) * delta);
    weights[ulong(feature) * p.groups + p.group] += dw;
    *update = LinearUpdate{dw, feature};
  }
}
kernel void linear_pick(device const SurvivalReal* deltas [[buffer(0)]], device const float* scores [[buffer(1)]],
                         device float* weights [[buffer(2)]], device LinearUpdate* update [[buffer(3)]],
                         constant LinearParams& p [[buffer(4)]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float best_score[256];
  threadgroup uint best_feature[256];
  float score = 0; uint feature = 0;
  for (uint f = tid; f < p.cols; f += 256) {
    float candidate = fabs(scores[ulong(p.group) * p.cols + f]);
    if (candidate > score) { score = candidate; feature = f; }
  }
  best_score[tid] = score; best_feature[tid] = feature;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 128; step; step >>= 1) {
    if (tid < step) {
      auto other = tid + step;
      if (best_score[other] > best_score[tid] ||
          (best_score[other] == best_score[tid] && best_feature[other] < best_feature[tid])) {
        best_score[tid] = best_score[other]; best_feature[tid] = best_feature[other];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!tid) {
    feature = best_feature[0];
    float dw = extended_result(sr(p.learning_rate) * deltas[ulong(p.group) * p.cols + feature]);
    weights[ulong(feature) * p.groups + p.group] += dw;
    *update = LinearUpdate{dw, feature};
  }
}
kernel void linear_residual(device const float* data [[buffer(0)]], device Pair* gradient [[buffer(1)]],
                             device const LinearUpdate* update [[buffer(2)]], constant LinearParams& p [[buffer(3)]],
                             uint row [[thread_position_in_grid]]) {
  #pragma clang fp contract(off)
  if (row >= p.rows || update->value == 0) return;
  ulong index = ulong(row) * p.groups + p.group;
  auto pair = gradient[index];
  if (pair.h < 0) return;
  float change;
  if (update->feature == p.cols) change = pair.h * update->value;
  else {
    float value = data[ulong(update->feature) * p.rows + row];
    if (isnan(value)) return;
    change = (pair.h * value) * update->value;
  }
  gradient[index].g = pair.g + change;
}
kernel void linear_convergence(device const float* weights [[buffer(0)]], device float* previous [[buffer(1)]],
                                device uint* converged [[buffer(2)]], constant LinearParams& p [[buffer(3)]],
                                uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[256];
  float largest = 0;
  for (ulong i = tid; i < (ulong(p.cols) + 1) * p.groups; i += 256) {
    largest = max(largest, fabs(weights[i] - previous[i]));
    previous[i] = weights[i];
  }
  scratch[tid] = largest;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 128; step; step >>= 1) {
    if (tid < step) scratch[tid] = max(scratch[tid], scratch[tid + step]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!tid) *converged = scratch[0] <= p.tolerance;
}
kernel void linear_predict(device const float* data [[buffer(0)]], device const float* weights [[buffer(1)]],
                            device const float* base [[buffer(2)]], device const float* margins [[buffer(3)]],
                            device float* output [[buffer(4)]], constant PredictParams& p [[buffer(5)]],
                            uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.rows) return;
  ulong index = ulong(id.x) * p.groups + id.y;
  // p.trees carries the model feature count; a DMatrix may have fewer columns.
  float bias = weights[ulong(p.trees) * p.groups + id.y] + (p.has_margin ? margins[index] : base[0]);
  if (p.type == 2 || p.type == 3) {
    for (uint f = 0; f < p.trees; ++f) {
      float value = f < p.cols ? data[ulong(id.x) * p.cols + f] : NAN;
      output[index * (ulong(p.trees) + 1) + f] = isnan(value) ? 0 : value * weights[ulong(f) * p.groups + id.y];
    }
    output[index * (ulong(p.trees) + 1) + p.trees] = bias;
  } else {
    float sum = bias;
    for (uint f = 0; f < p.cols; ++f) {
      float value = data[ulong(id.x) * p.cols + f];
      if (!isnan(value)) sum = fma(value, weights[ulong(f) * p.groups + id.y], sum);
    }
    output[index] = sum;
  }
}
