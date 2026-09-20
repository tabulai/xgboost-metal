# Generated Metal kernels

`squared_error.metal` computes weighted squared-error gradients and Hessians.
Its buffers are interleaved gradient/Hessian output, labels, predictions, and
weights, in that order. Threads beyond the configured row count do not write.

Regenerate the source and its JSON manifest from the repository root:

```sh
python3 tools/generate_squared_error.py
```

Generation uses the Python standard library. The defaults are 131,071 rows and
256 threads per group; `--rows`, `--block`, and `--output` select other values.
The manifest records the source SHA-256 and validation status.

Add `--check` in an Apple Metal environment with NumPy and a PyTorch version
providing `torch.mps.compile_shader` to check arithmetic and row boundaries on
the GPU. PyTorch is optional and is not a package runtime dependency.

This standalone kernel is separate from the training entry point in
[`kernels/boost.metal`](../boost.metal).
