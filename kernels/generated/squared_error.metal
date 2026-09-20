// Weighted squared-error gradient and Hessian; one row per thread.
#include <metal_stdlib>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup(256)]]
void main_kernel(device float* __restrict gradient [[buffer(0)]],
                 device const float* __restrict label [[buffer(1)]],
                 device const float* __restrict prediction [[buffer(2)]],
                 device const float* __restrict weight [[buffer(3)]],
                 uint row [[thread_position_in_grid]]) {
  if (row >= 131071u) return;

  const float row_weight = weight[row];
  const ulong offset = 2ul * row;
  gradient[offset] = (prediction[row] - label[row]) * row_weight;
  gradient[offset + 1] = row_weight;
}
