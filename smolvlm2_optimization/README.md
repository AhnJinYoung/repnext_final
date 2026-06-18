# SmolVLM2 2.2B A100 TVM Optimization Pipeline

This directory contains the server-side pipeline for benchmarking
`HuggingFaceTB/SmolVLM2-2.2B-Instruct` against an optimized compiler path on an
NVIDIA A100 80 GB machine.

## Why This Shape

SmolVLM2 accepts video, image, multi-image, and text inputs and generates text.
Hugging Face's model card describes it as an Idefics3-based
image/multi-image/video/text model. The server pipeline decodes each mp4 with
`imageio-ffmpeg`, samples frames in time order, and feeds them as multi-image
inputs. This avoids making PyAV/FFmpeg development headers a hard server
dependency.

The autoregressive language-generation loop is dynamic, but the visual encoder
has fixed tensor shapes after preprocessing. The pipeline therefore uses:

- **TVM Relay CUDA compilation** for the fixed-shape SmolVLM2 vision tower.
- **Native PyTorch** as the baseline end-to-end VLM generation path.
- **TorchInductor via `torch.compile`** as the end-to-end compiler path, with
  fallback to native if the server environment cannot compile the full model.

This keeps TVM as the main DL compiler artifact while avoiding brittle attempts
to compile the entire dynamic `generate()` loop.

## One-command Server Run

Run this on the A100 server from the repository root:

```bash
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

The script performs all required stages:

1. Creates a Python virtual environment under
   `smolvlm2_optimization/runs/a100_smolvlm2_2b/.venv`.
2. Installs PyTorch CUDA, Transformers, video IO, TVM, and optional FlashAttention.
3. Downloads `HuggingFaceTB/SmolVLM2-2.2B-Instruct`.
4. Compiles the vision tower with TVM for CUDA `sm_80`.
5. Runs native and optimized generation on the three final demo videos.
6. Writes latency/accuracy JSON, a summary markdown file, and concat videos.

Useful overrides:

```bash
WORK_DIR=/data/smolvlm2_run \
MAX_NEW_TOKENS=64 \
BENCH_ITERS=10 \
DEMO_SECONDS=10 \
VIDEO_SAMPLE_FRAMES=8 \
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

## Outputs

The default output directory is:

```text
smolvlm2_optimization/runs/a100_smolvlm2_2b/
```

Important files:

```text
benchmark_results.json                      full benchmark and generated text
SUMMARY.md                                  human-readable result table
artifacts/smolvlm2_vision_tvm_cuda_sm80.so  TVM compiled vision tower
demo_outputs/*_source_native_optimized.mp4  Source | Native | Optimized demos
```

The lightweight accuracy metric is a keyword-hit score for the three known demo
videos. It is intended as a reproducible smoke benchmark, not a replacement for
large public VLM benchmarks such as Video-MME, MLVU, or MVBench.

## Sources

- Hugging Face model card:
  `https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- Transformers SmolVLM documentation:
  `https://huggingface.co/docs/transformers/en/model_doc/smolvlm`
- Hugging Face SmolVLM2 blog:
  `https://huggingface.co/blog/smolvlm2`
