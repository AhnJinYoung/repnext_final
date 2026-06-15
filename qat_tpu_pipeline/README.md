# QAT TPU Pipeline

Single-directory entrypoint for training the Coral TPU version and exporting the compiled binary.

The scripts in this directory call the repo's existing model/export code, but all QAT workflow commands are here:

```text
qat_tpu_pipeline/
  install_env.sh
  train.sh
  export_compile.sh
  run_all.sh
  train_qat_distill.py
  requirements.txt
```

## Quick Start on GPU Server

```bash
cd repnext-optimization
export ADE_ROOT=/path/to/ade20k
export TEACHER_WEIGHTS=repnext_m5_ade20k.pth

bash qat_tpu_pipeline/install_env.sh
source .venv_qat/bin/activate
bash qat_tpu_pipeline/train.sh
bash qat_tpu_pipeline/export_compile.sh
```

The final compiled binary will be:

```text
build/w48_192_qat_edgetpu/*_edgetpu.tflite
```

## One Command

```bash
export ADE_ROOT=/path/to/ade20k
export TEACHER_WEIGHTS=repnext_m5_ade20k.pth
bash qat_tpu_pipeline/run_all.sh
```

## Useful Overrides

```bash
export TRAIN_LIMIT=8000
export VAL_LIMIT=500
export BATCH=24
export DISTILL_EPOCHS=30
export QAT_EPOCHS=15
export OUT=training/checkpoints/w48_192_distill_qat_latest.pth
export CKPT=training/checkpoints/w48_192_distill_qat_latest_best.pth
export OUT_DIR=build/w48_192_qat_edgetpu
```

## What It Produces

Training:

```text
training/checkpoints/w48_192_distill_qat_latest.pth
training/checkpoints/w48_192_distill_qat_latest_best.pth
```

Export/compile:

```text
build/w48_192_qat_edgetpu/*_full_integer_quant_dwpatched.tflite
build/w48_192_qat_edgetpu/*_edgetpu.tflite
logs/w48_192_qat_edgetpu/compile_192.log
```

## Accuracy Check

After export, run:

```bash
python benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend tflite \
  --model build/w48_192_qat_edgetpu/*_full_integer_quant_dwpatched.tflite \
  --size 192 \
  --limit 200 \
  --threads 4 \
  --normalize zero-one \
  --out benchmark/results/w48_192_qat_val200_tflite.json
```

The previous untrained w48 INT8 result was about `0.0031 mIoU`; the goal of this QAT/distillation workflow is to recover accuracy while preserving Edge TPU compilation.
