# Best Optimization Strategy

## Policy

Prioritize mathematically equivalent optimizations for best-version reporting.
Activation substitutions should be treated as accuracy-sensitive changes. RepNeXt's original GELU should be kept for correctness-preserving results, while ReLU or tanh-GELU variants should be labeled clearly in benchmark notes.

## Current Best Valid Path

- Use the original RepNeXt checkpoint and architecture semantics.
- Keep GELU semantics unchanged for any result reported as strictly correctness-preserving.
- Use sparse-equivalent downsample export when needed because it preserves the original downsample computation while making the graph easier to lower.
- Use fixed 512x512 input specialization only as a shape constraint, not as a model behavior change.

## TFLite / Edge TPU Status

- The previous ReLU-based RepNeXt artifacts should be reported with caution because they changed the activation math.
- The tanh-GELU export path is useful for conversion experiments, but it is not exactly equivalent to exact GELU and should be labeled as an activation-approximate variant when compared.
- The latest tanh-GELU TFLite CPU artifact runs, but Edge TPU compilation fails with an internal compiler error.

## Equivalent Conversion Patches

- `--sparse-equiv-downsample` is accepted as an equivalent graph rewrite.
- ONNX `kernel_shape` metadata repair is accepted when it only restores missing static metadata required by converters.
- TFLite depthwise convolution opcode-version patching is accepted only when the FlatBuffer operator semantics remain unchanged.

## Risk Items

- GELU-to-ReLU is a non-equivalent approximation and needs accuracy validation before being compared as a candidate.
- Exact GELU to tanh-GELU is also an approximation, so benchmark reports should label it explicitly.
- TPU-friendly downsample rewrites that alter padding, pooling, convolution order, or activation placement need equivalence proof or separate accuracy validation for the fixed exported shape.
