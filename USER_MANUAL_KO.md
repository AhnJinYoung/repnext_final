# 프로젝트 재현 및 실행 매뉴얼

## 1. 문서 목적

본 문서는 `repnext-optimization` 저장소의 전체 실험을 다른 사용자가 재현할 수 있도록 정리한 실행 매뉴얼이다. 단순히 최종 데모를 실행하는 방법만이 아니라, 환경 구성, 모델 최적화, 컴파일, 정확도 및 지연시간 벤치마크, 영상 데모 생성, 산출물 확인 절차를 하나의 파이프라인 관점에서 설명한다.

프로젝트는 두 축으로 구성된다.

1. **RepNeXt-M5 semantic segmentation 최적화**
   - 대상: ADE20K semantic segmentation
   - 주요 하드웨어: Intel CPU, Raspberry Pi 5 ARM CPU, Raspberry Pi 5 + Coral Edge TPU
   - 주요 컴파일/최적화 도구: PyTorch compile, OpenVINO, ONNX, onnx2tf, LiteRT/TFLite, Edge TPU compiler

2. **SmolVLM2 2.2B video-language 최적화**
   - 대상: 3개 영상에 대한 video understanding text generation
   - 주요 하드웨어: NVIDIA A100 80 GB
   - 주요 컴파일/최적화 도구: TVM Relay, TVM auto-tuning, TorchInductor

## 2. 저장소 구조

주요 디렉토리와 파일은 다음과 같다.

```text
README.md                                  프로젝트 요약 및 주요 결과
USER_MANUAL_KO.md                          본 한글 실행 매뉴얼
requirements.txt                           RepNeXt 재현용 Python 및 비-Python 의존성
report.tex                                 최종 보고서 원문
BENCHMARK_RESULTS_AND_METHODS.md           최종 수치와 방법 요약

conversion/                                RepNeXt ONNX/TFLite 변환 코드
benchmark/ade20k_accuracy_benchmark.py     ADE20K 정확도/지연시간 벤치마크
scripts/export_compile_edgetpu.sh          checkpoint -> TFLite -> Edge TPU 컴파일
qat_tpu_pipeline/                          QAT/distillation 기반 TPU 정확도 회복 파이프라인

demo/video_segmentation_demo.py            영상/프레임 segmentation 실행기
demo/make_source_native_rpi5_realtime_video.py
                                           Source | Native | RPi5 데모 concat 생성기
demo/runtime_graph_viz.py                  latency/accuracy 그래프 재생성

smolvlm2_optimization/                     SmolVLM2 A100/TVM 파이프라인
```

## 3. 공통 준비 사항

### 3.1 저장소 받기

```bash
git clone https://github.com/AhnJinYoung/repnext_final.git
cd repnext_final
```

이미 받은 저장소라면 최신 상태로 맞춘다.

```bash
git pull
```

### 3.2 데이터와 모델 파일

RepNeXt 재현에는 다음 파일이 필요하다.

```text
ADE20K dataset root
RepNeXt-M5 ADE20K checkpoint
```

예시 경로:

```bash
export ADE_ROOT=/data/ADEChallengeData2016
export REPNEXT_WEIGHTS=/workspace/tvm/handoff/repnext_m5_ade20k.pth
```

SmolVLM2 파이프라인은 Hugging Face에서 모델을 자동 다운로드한다. 서버가 Hugging Face 접근 권한을 요구하는 경우에는 먼저 로그인한다.

```bash
huggingface-cli login
```

### 3.3 RepNeXt 기본 Python 환경

로컬 CPU/변환/그래프 재생성 환경은 다음과 같이 만든다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`에는 pip 패키지뿐 아니라 Edge TPU compiler, `flatc`, SSH 도구, Raspberry Pi/Coral 런타임 등 프로젝트 재현에 필요한 비-Python 의존성도 주석으로 정리되어 있다.

## 4. RepNeXt 최적화 파이프라인

### 4.1 Native PyTorch baseline 측정

Native PyTorch baseline은 최적화 전 기준점이다. ADE20K validation subset에 대해 지연시간과 mIoU를 측정한다.

```bash
python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend pytorch \
  --weights "$REPNEXT_WEIGHTS" \
  --activation gelu \
  --size 512 \
  --limit 40 \
  --out benchmark/results/pytorch_gelu_512_reproduce.json
```

출력 JSON에는 다음 항목이 포함된다.

```text
latency.avg_ms
latency.p50_ms
latency.p95_ms
miou
pixel_accuracy
```

### 4.2 ONNX export

RepNeXt 모델을 compiler toolchain에 넣기 위해 ONNX로 export한다.

```bash
python3 conversion/export_onnx.py \
  --ckpt "$REPNEXT_WEIGHTS" \
  --activation tanh-gelu \
  --size 512 \
  --out artifacts/full_repnext/repnext_m5_tanhgelu_512.onnx
```

TPU 및 low-resolution 실험에서는 전체 logits graph를 고정 해상도로 export한다.

```bash
python3 conversion/export_full_tile_logits_onnx.py \
  --ckpt "$REPNEXT_WEIGHTS" \
  --activation tanh-gelu \
  --size 192 \
  --out artifacts/full_repnext/repnext_m5_tanhgelu_192_logits.onnx
```

### 4.3 LiteRT/TFLite 변환 및 quantization

TFLite/LiteRT 변환에는 real-image calibration tensor가 필요하다.

```bash
python3 conversion/make_real_calib.py \
  --ade-root "$ADE_ROOT" \
  --split training \
  --size 256 \
  --samples 200 \
  --out calibration_image_sample_data_200x256x256x3_float32.npy
```

low-resolution 고정 입력 모델을 빌드한다.

```bash
python3 conversion/build_lowres_fixed.py \
  --sizes 256 \
  --activation tanh-gelu \
  --weights "$REPNEXT_WEIGHTS" \
  --calib-dir . \
  --out-dir build/lowres_fixed \
  --log-dir logs/lowres_fixed
```

이 과정은 ONNX export, onnx2tf 변환, TFLite quantization, 필요한 depthwise op patch, Edge TPU compiler 호출을 포함한다.

### 4.4 Edge TPU 컴파일

이미 생성된 checkpoint에서 Edge TPU binary까지 한 번에 만들려면 다음 스크립트를 사용한다.

```bash
bash scripts/export_compile_edgetpu.sh \
  --ckpt "$REPNEXT_WEIGHTS" \
  --ade-root "$ADE_ROOT" \
  --size 192 \
  --out-dir build/qat_distill_edgetpu \
  --log-dir logs/qat_distill_edgetpu
```

핵심 산출물:

```text
build/qat_distill_edgetpu/*_full_integer_quant_dwpatched.tflite
build/qat_distill_edgetpu/*_edgetpu.tflite
logs/qat_distill_edgetpu/*.log
```

Edge TPU compiler 로그에서 전체 op가 TPU에 mapping되었는지 확인한다.

```bash
tail -80 logs/qat_distill_edgetpu/*.log
```

### 4.5 QAT/distillation 기반 TPU 정확도 회복

Full INT8 quantization은 RepNeXt segmentation 정확도를 크게 손상시킨다. 이를 완화하기 위해 QAT/distillation 파이프라인을 사용한다.

GPU 서버에서 다음을 실행한다.

```bash
cd repnext_final
export ADE_ROOT=/path/to/ade20k
export TEACHER_WEIGHTS=/path/to/repnext_m5_ade20k.pth

bash qat_tpu_pipeline/run_all.sh
```

세부 단계로 나누어 실행할 수도 있다.

```bash
bash qat_tpu_pipeline/install_env.sh
source .venv_qat/bin/activate
bash qat_tpu_pipeline/train.sh
bash qat_tpu_pipeline/export_compile.sh
```

주요 override:

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

최종 산출물:

```text
training/checkpoints/w48_192_distill_qat_latest.pth
training/checkpoints/w48_192_distill_qat_latest_best.pth
build/w48_192_qat_edgetpu/*_edgetpu.tflite
logs/w48_192_qat_edgetpu/compile_192.log
```

## 5. RepNeXt 벤치마크 재현

### 5.1 PyTorch 모델 정확도/지연시간

```bash
python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend pytorch \
  --weights "$REPNEXT_WEIGHTS" \
  --activation tanh-gelu \
  --size 512 \
  --limit 40 \
  --out benchmark/results/ade20k_val40_pytorch_tanhgelu_512_reproduce.json
```

### 5.2 TFLite/LiteRT 모델 정확도/지연시간

```bash
python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend tflite \
  --model build/lowres_fixed/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite \
  --size 256 \
  --limit 40 \
  --threads 4 \
  --normalize zero-one \
  --out benchmark/results/ade20k_val40_litert256_reproduce.json
```

### 5.3 Edge TPU 모델 벤치마크

Raspberry Pi 5 + Coral USB Edge TPU 환경에서 실행한다.

```bash
python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend tflite \
  --model build/w48_192_qat_edgetpu/repnext_m5_tanhgelu_real_full_192_logits_full_integer_quant_dwpatched_edgetpu.tflite \
  --delegate edgetpu \
  --size 192 \
  --limit 40 \
  --threads 4 \
  --normalize zero-one \
  --out benchmark/results/ade20k_val40_w48_192_edgetpu_reproduce.json
```

## 6. RepNeXt 영상 데모 재현

### 6.1 영상 프레임 추출

LiteRT/RPi 환경에서 video I/O가 부족할 수 있으므로, 먼저 host에서 프레임을 추출하는 방식이 안정적이다.

```bash
python3 demo/video_segmentation_demo.py extract \
  --input-video demo/video_sources_eye_new/source_busy_city_street.mp4 \
  --output-frames demo/video_runs_rpi_only/busy_city_street_10s/input_frames \
  --max-frames 240
```

### 6.2 Native PyTorch overlay 생성

```bash
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
```

### 6.3 RPi5 CPU LiteRT overlay 생성

Raspberry Pi 5 또는 동일한 LiteRT runtime이 있는 환경에서 실행한다.

```bash
python3 demo/video_segmentation_demo.py run \
  --input-frames demo/video_runs_rpi_only/busy_city_street_10s/input_frames \
  --output-frames demo/video_runs_rpi_only/busy_city_street_10s/rpi5_cpu_frames \
  --metrics demo/video_runs_rpi_only/busy_city_street_10s/rpi5_cpu_metrics.json \
  --name "RPi5 CPU LiteRT 256" \
  --backend tflite \
  --model /path/to/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite \
  --size 256 \
  --threads 4
```

### 6.4 Source | Native | RPi5 concat 영상 생성

```bash
python3 demo/make_source_native_rpi5_realtime_video.py \
  --root demo/video_runs_rpi_only/busy_city_street_10s \
  --output demo/video_runs_rpi_only/source_native_rpi5_3panel/city_street_source_native_rpi5_24fps.mp4 \
  --fps 24
```

최종 3개 데모 영상은 다음 위치에 모은다.

```text
demo/video_runs_rpi_only/source_native_rpi5_3panel/
```

## 7. 그래프와 보고서용 그림 재생성

벤치마크 JSON을 바탕으로 보고서용 그래프를 다시 생성한다.

```bash
python3 demo/runtime_graph_viz.py
python3 demo/seg_compare_viz.py
```

출력 위치:

```text
demo/runtime_graphs/
demo/seg_compare/
```

## 8. SmolVLM2 2.2B A100/TVM 파이프라인

### 8.1 목적

SmolVLM2는 video, image, multi-image, text를 입력으로 받아 텍스트를 생성하는 VLM이다. 전체 `generate()` 루프는 autoregressive token generation과 KV cache 때문에 동적이다. 따라서 TVM 최적화 대상은 고정 shape 입력을 갖는 **vision tower**로 설정한다.

vision tower는 영상에서 샘플링한 프레임을 visual embedding으로 변환하는 encoder 부분이다. 본 파이프라인은 다음 비교를 수행한다.

```text
Native PyTorch end-to-end generation
TorchInductor end-to-end generation
TVM Relay/auto-tuned vision tower microbenchmark
```

### 8.2 A100 서버 원샷 실행

NVIDIA A100 80 GB 서버에서 저장소 root 기준으로 실행한다.

```bash
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

스크립트가 수행하는 단계:

1. `smolvlm2_optimization/runs/a100_smolvlm2_2b/.venv` 생성
2. PyTorch CUDA, Transformers, 영상 I/O 패키지 설치
3. TVM Relay/AutoTVM 호환 버전 확인
4. pip wheel이 없으면 TVM `v0.14.0` source build
5. SmolVLM2 2.2B 모델 다운로드
6. 세 개 demo 영상을 multi-image input으로 변환
7. native PyTorch generation 벤치마크
8. TorchInductor generation 벤치마크
9. TVM vision tower compile/tuning/benchmark
10. Source | Native | TorchInductor concat mp4 생성

### 8.3 SmolVLM2 주요 옵션

```bash
WORK_DIR=/data/smolvlm2_run \
MAX_NEW_TOKENS=64 \
BENCH_ITERS=10 \
DEMO_SECONDS=10 \
VIDEO_SAMPLE_FRAMES=8 \
TVM_TUNING_TRIALS=64 \
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

TVM source build를 명시적으로 고정하려면 다음과 같이 실행한다.

```bash
TVM_GIT_REF=v0.14.0 \
TVM_SOURCE_DIR=/data/tvm-v0.14.0-src \
TVM_TUNING_TRIALS=64 \
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

현재 스크립트는 서버 패키지 drift를 피하기 위해 TVM source build를 다음 설정으로 수행한다.

```text
USE_CUDA=ON
USE_LLVM=OFF
USE_CUBLAS=ON
USE_CUDNN=OFF
USE_GTEST=OFF
BUILD_TESTING=OFF
```

`USE_LLVM=OFF`이므로 `LLVMConfig.cmake`가 없어도 빌드가 가능하다. Python 쪽 compile target은 CUDA device code와 C host target을 사용한다.

### 8.4 SmolVLM2 산출물

기본 출력 위치:

```text
smolvlm2_optimization/runs/a100_smolvlm2_2b/
```

주요 파일:

```text
benchmark_results.json
SUMMARY.md
artifacts/smolvlm2_vision_tvm_*.so
artifacts/autotvm_smolvlm2_vision_cuda_sm80.log
artifacts/meta_schedule_smolvlm2_vision_cuda_sm80/
demo_outputs/*_source_native_optimized.mp4
```

결과 확인:

```bash
cat smolvlm2_optimization/runs/a100_smolvlm2_2b/SUMMARY.md
python -m json.tool smolvlm2_optimization/runs/a100_smolvlm2_2b/benchmark_results.json | less
ls -lh smolvlm2_optimization/runs/a100_smolvlm2_2b/demo_outputs/
```

### 8.5 SmolVLM2 결과 해석

`SUMMARY.md`에서 반드시 확인할 항목:

```text
TVM Vision Tower
Status: ok
Tuning backend: autotvm 또는 meta_schedule
Mean latency
Artifact
```

`Status`가 `ok`가 아니면 TVM 최적화 결과로 보고하면 안 된다. 이 경우 에러 메시지를 확인하고 TVM build/runtime 문제를 먼저 해결해야 한다.

## 9. 산출물 관리와 Google Drive 전달

실험 결과 디렉토리는 `.gitignore`에 포함되어 있으므로 기본적으로 git에 올라가지 않는다. 이는 모델, mp4, TVM build 결과가 크기 때문이다.

결과를 전달하려면 압축 파일로 묶는 방식을 권장한다.

```bash
tar -czf smolvlm2_a100_results.tar.gz \
  smolvlm2_optimization/runs/a100_smolvlm2_2b
```

RepNeXt 데모 결과도 동일하게 묶을 수 있다.

```bash
tar -czf repnext_video_demos.tar.gz \
  demo/video_runs_rpi_only/source_native_rpi5_3panel
```

git에 강제로 포함해야 할 경우에는 `git add -f`를 사용한다. 단, mp4와 compiled binary는 repository 크기를 크게 만들 수 있으므로 권장하지 않는다.

```bash
git add -f smolvlm2_optimization/runs/a100_smolvlm2_2b/SUMMARY.md
git add -f smolvlm2_optimization/runs/a100_smolvlm2_2b/benchmark_results.json
```

## 10. 재현 체크리스트

최소 재현 성공 기준은 다음과 같다.

```text
[ ] requirements.txt 설치 완료
[ ] ADE_ROOT 설정 완료
[ ] RepNeXt checkpoint 경로 설정 완료
[ ] Native PyTorch ADE20K benchmark JSON 생성
[ ] LiteRT/TFLite 변환 산출물 생성
[ ] TFLite ADE20K benchmark JSON 생성
[ ] Edge TPU compiler log 생성
[ ] 영상 overlay frame 및 metrics JSON 생성
[ ] Source | Native | RPi5 concat mp4 생성
[ ] SmolVLM2 A100 SUMMARY.md 생성
[ ] SmolVLM2 TVM Vision Tower Status가 ok
[ ] SmolVLM2 demo_outputs mp4 생성
```

## 11. 문제 해결 기준

### 11.1 `tvm.relay`가 없는 경우

`apache-tvm 0.25.0rc*`가 설치된 상태일 가능성이 높다. 이 버전은 본 파이프라인의 Relay/AutoTVM 코드와 맞지 않는다. 스크립트는 이를 감지하면 제거하고, legacy wheel 또는 source build로 전환한다.

수동 초기화:

```bash
smolvlm2_optimization/runs/a100_smolvlm2_2b/.venv/bin/python \
  -m pip uninstall -y apache-tvm tvm
```

### 11.2 TVM source build가 이전 cache를 재사용하는 경우

```bash
rm -rf smolvlm2_optimization/runs/a100_smolvlm2_2b/apache-tvm-src/build
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

### 11.3 GTest 또는 LLVM CMake 에러

현재 스크립트는 다음 옵션으로 해당 문제를 회피한다.

```text
USE_GTEST=OFF
BUILD_TESTING=OFF
USE_LLVM=OFF
```

최신 코드를 받았는지 먼저 확인한다.

```bash
git pull
```

### 11.4 PyAV/FFmpeg build 에러

SmolVLM2 파이프라인은 `av`를 필수 의존성으로 사용하지 않는다. 영상은 `imageio-ffmpeg`로 decode하고, 모델에는 sampled multi-image 입력을 전달한다. 따라서 `av` 설치 실패가 발생하면 오래된 requirements를 사용 중인지 확인한다.

```bash
git pull
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

## 12. 최종 재현 순서 요약

전체 프로젝트를 처음부터 끝까지 재현하는 가장 짧은 순서는 다음과 같다.

```bash
git clone https://github.com/AhnJinYoung/repnext_final.git
cd repnext_final

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export ADE_ROOT=/path/to/ade20k
export REPNEXT_WEIGHTS=/path/to/repnext_m5_ade20k.pth

python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend pytorch \
  --weights "$REPNEXT_WEIGHTS" \
  --activation tanh-gelu \
  --size 512 \
  --limit 40 \
  --out benchmark/results/reproduce_native.json

python3 conversion/make_real_calib.py \
  --ade-root "$ADE_ROOT" \
  --split training \
  --size 256 \
  --samples 200 \
  --out calibration_image_sample_data_200x256x256x3_float32.npy

python3 conversion/build_lowres_fixed.py \
  --sizes 256 \
  --activation tanh-gelu \
  --weights "$REPNEXT_WEIGHTS" \
  --calib-dir . \
  --out-dir build/lowres_fixed \
  --log-dir logs/lowres_fixed

python3 benchmark/ade20k_accuracy_benchmark.py \
  --ade-root "$ADE_ROOT" \
  --backend tflite \
  --model build/lowres_fixed/repnext_m5_tanhgelu_real_full_256_logits_dynamic_range_quant.tflite \
  --size 256 \
  --limit 40 \
  --threads 4 \
  --normalize zero-one \
  --out benchmark/results/reproduce_litert256.json

python3 demo/runtime_graph_viz.py
```

SmolVLM2는 A100 서버에서 별도로 실행한다.

```bash
git pull
TVM_TUNING_TRIALS=64 \
bash smolvlm2_optimization/run_smolvlm2_a100_pipeline.sh
```

최종적으로 다음 파일들이 존재하면 재현 파이프라인이 정상적으로 완료된 것이다.

```text
benchmark/results/reproduce_native.json
benchmark/results/reproduce_litert256.json
demo/runtime_graphs/*.png
smolvlm2_optimization/runs/a100_smolvlm2_2b/SUMMARY.md
smolvlm2_optimization/runs/a100_smolvlm2_2b/benchmark_results.json
smolvlm2_optimization/runs/a100_smolvlm2_2b/demo_outputs/*.mp4
```
