# Best Optimization Strategy

## Policy

Only mathematically equivalent optimizations are accepted as best-version optimizations.
Activation substitutions are not valid best-version optimizations. RepNeXt's original GELU must not be replaced by ReLU for a correctness-preserving result.

## Current Best Valid Path

- Use the original RepNeXt checkpoint and architecture semantics.
- Keep GELU semantics unchanged for any result reported as best-version correct.
- Use sparse-equivalent downsample export when needed because it preserves the original downsample computation while making the graph easier to lower.
- Use fixed 512x512 input specialization only as a shape constraint, not as a model behavior change.

## TFLite / Edge TPU Status

- The previous ReLU-based RepNeXt artifacts are deprecated because they changed the activation math.
- The tanh-GELU export path is useful for conversion experiments, but it is not exactly equivalent to exact GELU and should not be reported as the strict best version unless accepted as a separate model variant.
- The latest tanh-GELU TFLite CPU artifact runs, but Edge TPU compilation fails with an internal compiler error.

## Equivalent Conversion Patches

- `--sparse-equiv-downsample` is accepted as an equivalent graph rewrite.
- ONNX `kernel_shape` metadata repair is accepted when it only restores missing static metadata required by converters.
- TFLite depthwise convolution opcode-version patching is accepted only when the FlatBuffer operator semantics remain unchanged.

## Risk Items

- GELU-to-ReLU is a non-equivalent approximation and is rejected.
- Exact GELU to tanh-GELU is also an approximation, not a mathematically equivalent optimization.
- TPU-friendly downsample rewrites that alter padding, pooling, convolution order, or activation placement are rejected unless equivalence is proven for the fixed exported shape.
