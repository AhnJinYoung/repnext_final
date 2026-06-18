프로젝트 재현 및 실행 매뉴얼

이 문서는 RepNeXt-M5 ADE20K semantic segmentation 최적화 프로젝트를 재현하기 위한 실행 절차를 정리한 것이다. 범위는 환경 준비, 모델 변환, 컴파일, 정확도 및 latency benchmark, 영상 demo 생성까지이다.


1. 프로젝트 개요

본 프로젝트는 RepNeXt-M5 segmentation 모델을 edge 환경에서 더 빠르게 실행하기 위해 여러 compiler 기반 최적화를 적용했다.

주요 대상 환경은 다음과 같다.

Intel CPU
Raspberry Pi 5 ARM CPU
Raspberry Pi 5 + Google Coral USB Edge TPU

사용한 주요 도구는 다음과 같다.

PyTorch baseline
PyTorch compile
ONNX export
onnx2tf
LiteRT / TFLite
Edge TPU compiler
QAT / distillation


2. 저장소 준비

저장소를 clone한다.

    git clone https://github.com/AhnJinYoung/repnext_final.git
    cd repnext_final

이미 clone한 경우 최신 상태로 맞춘다.

    git pull


3. 필요한 데이터와 모델

재현에는 ADE20K dataset과 RepNeXt-M5 checkpoint가 필요하다.

예시:

    export ADE_ROOT=/data/ADEChallengeData2016
    export REPNEXT_WEIGHTS=/path/to/repnext_m5_ade20k.pth

이후 모든 명령은 repository root에서 실행한다고 가정한다.


4. Python 환경 설치

기본 실행 환경을 만든다.

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

requirements.txt에는 Python package뿐 아니라 Edge TPU compiler, flatc, Raspberry Pi / Coral runtime 등 재현에 필요한 비-Python 의존성도 주석으로 정리되어 있다.


5. Native PyTorch baseline 측정

최적화 전 기준 latency와 accuracy를 측정한다.

    python3 benchmark/ade20k_accuracy_benchmark.py \
      --ade-root "$ADE_ROOT" \
      --backend pytorch \
      --weights "$REPNEXT_WEIGHTS" \
      --activation gelu \
      --size 512 \
      --limit 40 \
      --out benchmark/results/reproduce_native_pytorch_512.json

결과 JSON에서 확인할 항목은 latency.avg_ms, latency.p50_ms, latency.p95_ms, miou, pixel_accuracy이다.


6. ONNX export

compiler toolchain에 넣기 위해 PyTorch checkpoint를 ONNX로 export한다.

    python3 conversion/export_onnx.py \
      --ckpt "$REPNEXT_WEIGHTS" \
      --activation tanh-gelu \
      --size 512 \
      --out artifacts/full_repnext/repnext_m5_tanhgelu_512.onnx

TPU 또는 low-resolution 실험용 logits graph는 고정 입력 크기로 export한다.

    python3 conversion/export_full_tile_logits_onnx.py \
      --ckpt "$REPNEXT_WEIGHTS" \
      --activation tanh-gelu \
      --size 192 \
      --out artifacts/full_repnext/repnext_m5_tanhgelu_192_logits.onnx


7. Calibration data 생성

TFLite quantization에는 실제 ADE20K 이미지 기반 calibration tensor가 필요하다.

    python3 conversion/make_real_calib.py \
      --ade-root "$ADE_ROOT" \
      --split training \
      --size 256 \
      --samples 200 \
      --out calibration_image_sample_data_200x256x256x3_float32.npy


8. LiteRT / TFLite 변환

low-resolution 고정 입력 모델을 TFLite로 변환한다.

    python3 conversion/build_lowres_fixed.py \
      --sizes 256 \
      --activation tanh-gelu \
      --weights "$REPNEXT_WEIGHTS" \
      --calib-dir . \
      --out-dir build/lowres_fixed \
      --log-dir logs/lowres_fixed

이 단계에서 수행되는 일은 ONNX export, onnx2tf 변환, TFLite quantization, depthwise op patch, Edge TPU compiler 실행이다.

주요 산출물은 build/lowres_fixed 아래에 생성된다.


9. TFLite 모델 benchmark

변환된 TFLite 모델의 accuracy와 latency를 측정한다.

    python3 benchmark/ade20k_accuracy_benchmark.py \
      --ade-root "$ADE_ROOT" \
      --backend tflite \
      --model build/lowres_fixed/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite \
      --size 256 \
      --limit 40 \
      --threads 4 \
      --normalize zero-one \
      --out benchmark/results/reproduce_litert_256.json


10. Edge TPU compile

Edge TPU binary를 만들려면 Edge TPU compiler가 설치되어 있어야 한다.

QAT 또는 distillation 없이 checkpoint에서 바로 export와 compile을 실행할 경우:

    bash scripts/export_compile_edgetpu.sh \
      --ckpt "$REPNEXT_WEIGHTS" \
      --ade-root "$ADE_ROOT" \
      --size 192 \
      --out-dir build/qat_distill_edgetpu \
      --log-dir logs/qat_distill_edgetpu

주요 산출물은 다음과 같다.

    build/qat_distill_edgetpu/*_full_integer_quant_dwpatched.tflite
    build/qat_distill_edgetpu/*_edgetpu.tflite
    logs/qat_distill_edgetpu/*.log

compile log에서 Edge TPU에 mapping된 op 수를 확인한다.

    tail -80 logs/qat_distill_edgetpu/*.log


11. QAT / distillation 파이프라인

Full INT8 quantization은 segmentation accuracy를 크게 떨어뜨릴 수 있다. TPU accuracy를 회복하려면 QAT / distillation 파이프라인을 사용한다.

GPU 서버에서 실행한다.

    export ADE_ROOT=/path/to/ade20k
    export TEACHER_WEIGHTS=/path/to/repnext_m5_ade20k.pth
    bash qat_tpu_pipeline/run_all.sh

단계별 실행도 가능하다.

    bash qat_tpu_pipeline/install_env.sh
    source .venv_qat/bin/activate
    bash qat_tpu_pipeline/train.sh
    bash qat_tpu_pipeline/export_compile.sh

자주 바꾸는 옵션은 다음과 같다.

    export TRAIN_LIMIT=8000
    export VAL_LIMIT=500
    export BATCH=24
    export DISTILL_EPOCHS=30
    export QAT_EPOCHS=15
    export OUT=training/checkpoints/w48_192_distill_qat_latest.pth
    export CKPT=training/checkpoints/w48_192_distill_qat_latest_best.pth
    export OUT_DIR=build/w48_192_qat_edgetpu

최종 산출물은 다음 위치에 생성된다.

    training/checkpoints/
    build/w48_192_qat_edgetpu/
    logs/w48_192_qat_edgetpu/


12. Edge TPU 모델 benchmark

Raspberry Pi 5 + Coral USB Edge TPU 환경에서 실행한다.

    python3 benchmark/ade20k_accuracy_benchmark.py \
      --ade-root "$ADE_ROOT" \
      --backend tflite \
      --model build/w48_192_qat_edgetpu/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched_edgetpu.tflite \
      --delegate edgetpu \
      --size 192 \
      --limit 40 \
      --threads 4 \
      --normalize zero-one \
      --out benchmark/results/reproduce_edgetpu_192.json


13. 영상 demo 생성

먼저 source video에서 frame을 추출한다.

    python3 demo/video_segmentation_demo.py extract \
      --input-video demo/video_sources_eye_new/source_busy_city_street.mp4 \
      --output-frames demo/video_runs_rpi_only/busy_city_street_10s/input_frames \
      --max-frames 240

Native PyTorch overlay frame을 만든다.

    python3 demo/video_segmentation_demo.py run \
      --input-frames demo/video_runs_rpi_only/busy_city_street_10s/input_frames \
      --output-frames demo/video_runs_rpi_only/busy_city_street_10s/native_realtime_sparse_frames \
      --metrics demo/video_runs_rpi_only/busy_city_street_10s/native_realtime_sparse_metrics.json \
      --name "Native PyTorch 512" \
      --backend pytorch \
      --weights "$REPNEXT_WEIGHTS" \
      --activation gelu \
      --size 512 \
      --max-frames 5

RPi5 CPU LiteRT overlay frame을 만든다.

    python3 demo/video_segmentation_demo.py run \
      --input-frames demo/video_runs_rpi_only/busy_city_street_10s/input_frames \
      --output-frames demo/video_runs_rpi_only/busy_city_street_10s/rpi5_cpu_frames \
      --metrics demo/video_runs_rpi_only/busy_city_street_10s/rpi5_cpu_metrics.json \
      --name "RPi5 CPU LiteRT 256" \
      --backend tflite \
      --model /path/to/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite \
      --size 256 \
      --threads 4

Source, Native, RPi5 결과를 하나의 3-panel video로 합친다.

    python3 demo/make_source_native_rpi5_realtime_video.py \
      --root demo/video_runs_rpi_only/busy_city_street_10s \
      --output demo/video_runs_rpi_only/source_native_rpi5_3panel/city_street_source_native_rpi5_24fps.mp4 \
      --fps 24

최종 demo video는 다음 디렉토리에 모은다.

    demo/video_runs_rpi_only/source_native_rpi5_3panel/


14. 그래프 재생성

보고서용 latency / accuracy graph를 다시 생성한다.

    python3 demo/runtime_graph_viz.py
    python3 demo/seg_compare_viz.py

출력 위치는 다음과 같다.

    demo/runtime_graphs/
    demo/seg_compare/


15. 최종 결과 확인

최소한 다음 파일들이 생성되면 기본 재현이 완료된 것이다.

    benchmark/results/reproduce_native_pytorch_512.json
    benchmark/results/reproduce_litert_256.json
    build/lowres_fixed/*.tflite
    logs/**/*.log
    demo/runtime_graphs/*.png
    demo/video_runs_rpi_only/source_native_rpi5_3panel/*.mp4


16. 산출물 관리

대용량 산출물은 기본적으로 git에 넣지 않는다. 모델 checkpoint, TFLite binary, Edge TPU binary, mp4 등은 별도 압축 파일이나 Google Drive로 전달하는 것을 권장한다.

예시:

    tar -czf repnext_reproduction_outputs.tar.gz \
      benchmark/results \
      build \
      logs \
      demo/video_runs_rpi_only/source_native_rpi5_3panel

git에 꼭 포함해야 할 경우에만 git add -f를 사용한다.


17. 문제 해결

PyAV 또는 FFmpeg build 오류가 나면 imageio-ffmpeg 기반 video path를 사용하거나, 먼저 frame을 추출한 뒤 frame directory mode로 실행한다.

Edge TPU compiler가 실패하면 compile log에서 CPU fallback op 또는 unsupported op를 확인한다. 이 프로젝트에서는 depthwise patch와 low-resolution logits export를 통해 Edge TPU compiler가 가능한 형태로 graph를 맞추었다.

정확도가 낮게 나오면 calibration data, activation 설정, input size, normalize 옵션이 기존 실험과 같은지 먼저 확인한다.
