#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <video-name> <source-mp4>" >&2
  exit 2
fi

NAME="$1"
SOURCE="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/demo/video_runs_rpi_only/$NAME"

RPI_HOST="${RPI_HOST:-rpi5@192.168.0.200}"
RPI_PASS="${RPI_PASS:-pi1234}"
RPI_ENV="${RPI_ENV:-/home/rpi5/coral-env/bin/python}"
RPI_WORK="~/repnext-rpi-demo-$NAME"

FPS="${FPS:-24}"
FRAMES="${FRAMES:-120}"
NATIVE_FRAMES="${NATIVE_FRAMES:-8}"
DEMO_SECONDS="${DEMO_SECONDS:-16}"

VIDEO_PY="${VIDEO_PY:-$ROOT/.venv-video/bin/python}"
INTEL_PY="${INTEL_PY:-/workspace/tvm/local-convert-env/bin/python3}"
NATIVE_WEIGHTS="${NATIVE_WEIGHTS:-/workspace/tvm/handoff/repnext_m5_ade20k.pth}"
INTEL_MODEL="${INTEL_MODEL:-$ROOT/artifacts/full_repnext_192_target/onnx2tf_tanhgelu_192_logits/repnext_m5_tanhgelu_real_full_192_logits_dynamic_range_quant.tflite}"
RPI_CPU_MODEL="${RPI_CPU_MODEL:-/home/rpi5/repnext-pipeline/lowres_fixed_20260613/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite}"

rm -rf "$OUT"
mkdir -p "$OUT"
cp "$SOURCE" "$OUT/source.mp4"

"$VIDEO_PY" "$ROOT/demo/video_segmentation_demo.py" extract \
  --input-video "$OUT/source.mp4" \
  --output-frames "$OUT/input_frames" \
  --stride 1 \
  --max-frames "$FRAMES"

python3 "$ROOT/demo/video_segmentation_demo.py" run \
  --input-frames "$OUT/input_frames" \
  --output-frames "$OUT/native512_frames" \
  --metrics "$OUT/native512_metrics.json" \
  --name "Native PyTorch 512" \
  --backend pytorch \
  --weights "$NATIVE_WEIGHTS" \
  --size 512 \
  --threads 4 \
  --fps "$FPS" \
  --max-frames "$NATIVE_FRAMES"

"$INTEL_PY" "$ROOT/demo/video_segmentation_demo.py" run \
  --input-frames "$OUT/input_frames" \
  --output-frames "$OUT/litert192_frames" \
  --metrics "$OUT/litert192_metrics.json" \
  --name "Intel CPU LiteRT 192" \
  --backend tflite \
  --model "$INTEL_MODEL" \
  --size 192 \
  --threads 4 \
  --fps "$FPS" \
  --max-frames "$FRAMES"

sshpass -p "$RPI_PASS" ssh -o StrictHostKeyChecking=no "$RPI_HOST" "rm -rf $RPI_WORK && mkdir -p $RPI_WORK/input_frames"
sshpass -p "$RPI_PASS" scp -o StrictHostKeyChecking=no "$ROOT/demo/video_segmentation_demo.py" "$RPI_HOST:$RPI_WORK/"
sshpass -p "$RPI_PASS" scp -o StrictHostKeyChecking=no "$OUT"/input_frames/*.png "$RPI_HOST:$RPI_WORK/input_frames/"

sshpass -p "$RPI_PASS" ssh -o StrictHostKeyChecking=no "$RPI_HOST" "
cd $RPI_WORK && $RPI_ENV video_segmentation_demo.py run \
  --input-frames input_frames \
  --output-frames rpi5_cpu_frames \
  --metrics rpi5_cpu_metrics.json \
  --name 'RPi5 CPU LiteRT 256' \
  --backend tflite \
  --model '$RPI_CPU_MODEL' \
  --size 256 \
  --threads 4 \
  --fps '$FPS' \
  --max-frames '$FRAMES'
"

mkdir -p "$OUT/rpi5_cpu_frames"
sshpass -p "$RPI_PASS" scp -o StrictHostKeyChecking=no -r "$RPI_HOST:$RPI_WORK/rpi5_cpu_frames/." "$OUT/rpi5_cpu_frames/"
sshpass -p "$RPI_PASS" scp -o StrictHostKeyChecking=no "$RPI_HOST:$RPI_WORK/rpi5_cpu_metrics.json" "$OUT/"

"$VIDEO_PY" "$ROOT/demo/make_threepanel_wallclock_video.py" \
  --root "$OUT" \
  --output "$OUT/${NAME}_rpi_only_3panel_24fps.mp4" \
  --fps "$FPS" \
  --duration "$DEMO_SECONDS" \
  --panel-width 426 \
  --panel-height 240

python3 "$ROOT/demo/video_segmentation_demo.py" summarize \
  "$OUT/native512_metrics.json" \
  "$OUT/litert192_metrics.json" \
  "$OUT/rpi5_cpu_metrics.json" \
  --out "$OUT/${NAME}_rpi_only_3panel_summary.md"
