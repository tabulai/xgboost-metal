"""Generate a weighted squared-error Metal kernel.

Source generation uses only the Python standard library. The optional GPU check
uses NumPy and PyTorch; PyTorch is only a validation dependency.
"""
from pathlib import Path
import argparse
import hashlib
import json


def metal_source(rows, block):
    return f"""// Weighted squared-error gradient and Hessian; one row per thread.
#include <metal_stdlib>
using namespace metal;

[[kernel, max_total_threads_per_threadgroup({block})]]
void main_kernel(device float* __restrict gradient [[buffer(0)]],
                 device const float* __restrict label [[buffer(1)]],
                 device const float* __restrict prediction [[buffer(2)]],
                 device const float* __restrict weight [[buffer(3)]],
                 uint row [[thread_position_in_grid]]) {{
  if (row >= {rows}u) return;

  const float row_weight = weight[row];
  const ulong offset = 2ul * row;
  gradient[offset] = (prediction[row] - label[row]) * row_weight;
  gradient[offset + 1] = row_weight;
}}
"""


def validate(source, rows, block):
    import numpy as np
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("An Apple Metal GPU is required for validation.")
    launch_rows = ((rows + block - 1) // block) * block
    padded_rows = launch_rows + 1
    rng = np.random.default_rng(15)
    inputs = [rng.normal(size=padded_rows).astype(np.float32) for _ in range(2)]
    inputs.append(rng.uniform(0, 2, size=padded_rows).astype(np.float32))
    inputs[0][0] = inputs[1][0]
    if rows > 1:
        inputs[2][rows // 2] = 0
    prediction, label, weight = (torch.from_numpy(value).to("mps") for value in inputs)
    sentinel = 123456.0
    gradient = torch.full((padded_rows, 2), sentinel, device="mps", dtype=torch.float32)
    shader = torch.mps.compile_shader(source)
    shader.main_kernel(gradient, label, prediction, weight,
                       threads=launch_rows, group_size=block)
    result = gradient.cpu().numpy()
    np.testing.assert_array_equal(result[rows:], np.float32(sentinel),
                                  err_msg="Kernel wrote beyond the requested rows.")
    actual = result[:rows]
    expected = np.column_stack(((inputs[0][:rows] - inputs[1][:rows]) * inputs[2][:rows],
                                inputs[2][:rows]))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    return {
        "gpu_validated": True,
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "padding_untouched": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=131071)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("kernels/generated/squared_error.metal"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.block not in (64, 128, 256, 512):
        parser.error("Use positive rows and a block size of 64, 128, 256 or 512.")
    source = metal_source(args.rows, args.block)
    manifest = {
        "generator": "Native Metal template", "tool": "tools/generate_squared_error.py",
        "target": "metal", "rows": args.rows, "threads": args.block,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "gpu_validated": False,
        "role": "Standalone squared-error kernel; the training runtime uses kernels/boost.metal.",
    }
    if args.check:
        manifest.update(validate(source, args.rows, args.block))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
