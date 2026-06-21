# SmolVLM2 A100 TVM 최적화 — 실행 런북 & 사후분석

> 대상: 이 디렉토리(`smolvlm2_optimization/`)의 파이프라인을 **다른 사람이 안전하게 다시 돌릴 수 있도록** 하기 위한 문서.
> 최종 갱신: 2026-06-21

---

## 1. 목표 (What this does)

A100 GPU 환경에서 SmolVLM2-2.2B-Instruct 모델을:
- **TVM Relay + AutoTVM/MetaSchedule** 로 (고정 shape인) **비전 타워**를 CUDA `sm_80` 타겟으로 컴파일·튜닝하고,
- **native PyTorch** vs **TorchInductor(`torch.compile`)** end-to-end 생성 latency를 비교하고,
- 비전 타워에 대해 **TVM vs PyTorch** 마이크로벤치를 측정하고,
- 3개 데모 영상에 대해 `source | native | optimized` 3분할 데모 mp4를 생성한다.

결과물은 `runs/a100_smolvlm2_2b/` 아래:
- `benchmark_results.json`, `SUMMARY.md`
- `demo_outputs/*.mp4`
- `artifacts/` (TVM 튜닝 로그, 컴파일된 `.so` 등)

> 참고: `generate()`는 동적 autoregressive 루프라 전체 모델을 TVM으로 통째로 컴파일하지 않는다. **고정 shape인 비전 타워만** TVM 컴파일 대상이고, end-to-end는 native vs TorchInductor로 비교한다. 이건 의도된 설계다(`smolvlm2_tvm_pipeline.py` 상단 docstring 참고).

---

## 2. ⚠️ 가장 중요한 사후분석 — 왜 pod이 exit code 137로 죽었나

### 증상
파이프라인 실행 중 pod이 **exit code 137 (= 128 + 9 = SIGKILL)** 로 죽었다. CPU/RAM 사용량이 높지도 않았는데 죽었다.

### 결론: **우리 컨테이너의 메모리 한도 초과(cgroup OOM)가 아니다. 호스트(노드) 레벨 압박에 의한 외부 SIGKILL이다.**

근거(모두 직접 확인함):
- `/sys/fs/cgroup/memory.events` → `oom_kill 0`, `oom 0`, `max 0`. **cgroup 한도 OOM이 발생했다면 이 카운터가 0이 아니다.**
- `/sys/fs/cgroup/memory.pressure` → `some/full` 모두 `total=0`. 메모리 압박이 cgroup 내부에는 전혀 없었음.
- 로그는 `Loading checkpoint shards: 100%` 직후 **깔끔하게 끊김** — Python traceback도, memguard 트립 로그도 없음 → 외부에서 온 SIGKILL.

### 핵심 메커니즘 (왜 "사용량이 낮은데도" 죽는가)
- 이 컨테이너 cgroup 한도: **RAM 128 GiB**, CPU quota 64 core (`cpu.max = 6400000 100000`).
- 그런데 호스트 전체는 1 TiB RAM을 **여러 유저가 공유**한다. 다른 유저들이 대략 **70 vCPU, 640 GB RAM** 을 쓰고 있다.
- 즉 우리 cgroup 한도(128 GiB) > **호스트의 물리적 여유 RAM** 인 상황이 생긴다 (측정 시점 `MemFree ≈ 84–88 GiB`).
- 이 상태에서 우리가(또는 이웃이) RAM을 좀만 더 잡으면 **호스트 전체가 물리 RAM 고갈** → **커널의 global OOM killer** 또는 **kubelet node-pressure eviction** 이 발동.
- global OOM killer는 badness score(주로 RSS 크기)로 희생자를 고르며, **이 kill은 우리 cgroup의 `memory.events`를 증가시키지 않는다.** → 그래서 "우리 사용량은 낮았는데 죽었다"로 보인다.

### 그래서 바뀐 점
- 이전 memguard는 **cgroup 128 GiB 기준 110 GiB**에서 트립하도록 되어 있었다 → **이번 실패 모드에서는 무의미**(우리는 그 근처도 안 갔다).
- 새 memguard(`run_with_memguard.sh`)는 **호스트 `MemAvailable`도 함께 감시**하고, 더 자주(3초) 샘플링하며, 위험 시 **깨끗하게 종료**하고 상태를 파일로 남긴다. 자세한 건 §4.

> 한 줄 요약: **이건 noisy-neighbor(이웃 테넌트) 때문에 생기는 호스트 압박 문제다. 우리 코드가 메모리를 많이 써서가 아니다. 100% 막을 수는 없고, "우리 footprint를 작게 유지 + 재실행을 싸게 + 호스트 상태를 보고 깨끗이 빠지기"로 위험을 줄인다.**

---

## 3. 적용한 안전장치 (코드 변경 요약)

`run_smolvlm2_a100_pipeline.sh`:
- 모든 병렬도 노브를 호스트 코어 수가 아니라 **명시적 소수**로 고정 (가장 중요한 가드 중 하나):
  - `TVM_AUTOTVM_N_PARALLEL=4` (AutoTVM `LocalBuilder`의 기본값은 `multiprocessing.cpu_count()`=256 → cgroup quota 무시. 그대로 두면 빌드 워커가 256개까지 fork됨.)
  - `TORCHINDUCTOR_COMPILE_THREADS=4`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `TVM_NUM_THREADS=8`
  - 소스 빌드 fallback의 `cmake --build --parallel`을 `nproc` 대신 `TVM_BUILD_PARALLEL_JOBS=8`로 캡.
- `flash-attn` 설치를 **opt-in**으로 변경 (`INSTALL_FLASH_ATTN=1`). 휠이 없으면 소스 컴파일로 빠지면서 `MAX_JOBS`가 호스트 코어 수를 따라가 RAM을 터뜨릴 수 있어서. 기본 attention은 `eager`라 애초에 불필요.
- 파이프라인에 `--allow-tvm-failure` 추가 → TVM 단계가 실패해도 native / TorchInductor 비교와 데모 생성은 끝까지 진행.

`smolvlm2_tvm_pipeline.py`:
- `autotvm.LocalBuilder(timeout=20, n_parallel=4)` 로 빌드 병렬도 명시 제한.

`run_with_memguard.sh` (새 가드, §2·§4 참고).

> TVM(v0.14.0, Relay+AutoTVM+CUDA)은 이미 `runs/a100_smolvlm2_2b/apache-tvm-src/build`에 빌드되어 있어 **재빌드 불필요**. PyPI 최신 `apache-tvm==0.25.0`은 `tvm.relay`/`tvm.autotvm`이 제거되어 못 쓴다 — 그래서 소스 빌드를 쓰는 것. 이미 있는 빌드를 건드리지 말 것.

---

## 4. 실행 방법 (How to run)

### 권장: 메모리 가드로 백그라운드 실행

```bash
cd /data/repnext_final
# 보수적인 첫 실행: TVM 튜닝 trial 수를 낮게
TVM_TUNING_TRIALS=8 nohup bash smolvlm2_optimization/run_with_memguard.sh \
  > /dev/null 2>&1 & disown
```

진행 상황 확인:
```bash
D=/data/repnext_final/smolvlm2_optimization/runs/a100_smolvlm2_2b
tail -f "$D/run.log"        # 파이프라인 출력
tail -f "$D/memguard.log"   # 3초마다 self/host 메모리 heartbeat
cat   "$D/memguard.status"  # RUNNING / DONE / ABORTED <이유>
```

### 튜닝 가드 환경변수 (`run_with_memguard.sh`)
| 변수 | 기본값 | 의미 |
| --- | --- | --- |
| `SELF_LIMIT_GIB` | 50 | 우리 cgroup 사용량이 이걸 넘으면 = 우리 쪽 runaway → 트립 kill |
| `HOST_FLOOR_GIB` | 25 | 호스트 `MemAvailable`이 이 아래로 떨어지면 → 깨끗하게 abort (SIGKILL 당하기 전에) |
| `INTERVAL` | 3 | 샘플링 주기(초) |

### 파이프라인 튜닝 환경변수 (`run_smolvlm2_a100_pipeline.sh`)
| 변수 | 기본값 | 비고 |
| --- | --- | --- |
| `TVM_TUNING_TRIALS` | 64 | **첫 실행은 8~20 권장.** 호스트 한가할 때만 올릴 것 |
| `TVM_AUTOTVM_N_PARALLEL` | 4 | 절대 호스트 코어 수로 올리지 말 것 |
| `TORCHINDUCTOR_COMPILE_THREADS` | 4 | |
| `INSTALL_FLASH_ATTN` | 0 | 1로 켜면 소스 컴파일 위험 — 비권장 |

### 종료 코드
- `0` 정상 완료
- `99` memguard가 트립해서 중단 (`memguard.status`에 `ABORTED <이유>`)
- 그 외 → 파이프라인 자체 실패 (`run.log` 확인)

---

## 5. 다시 돌릴 때 유의사항 (Cautions)

1. **먼저 호스트 상태를 보고 시작하라.** `grep MemAvailable /proc/meminfo`. `MemAvailable`이 작으면(예: < 60 GiB) 이웃이 바쁜 것 — 잠시 기다렸다 시작하는 게 안전하다.
2. **TVM 튜닝 trial을 한 번에 크게 잡지 마라.** 8 → 20 → 64 식으로 점증. trial이 클수록 호스트 CPU/RAM·시간이 늘고 호스트 압박에 노출되는 시간이 길어진다.
3. **병렬도 노브를 `nproc`/`cpu_count()` 기준으로 되돌리지 마라.** 이 환경에서 `nproc`=256은 **거짓말**이다(실제 quota는 64). AutoTVM/Inductor/cmake/flash-attn 모두 이걸 따라가면 워커 폭발 → 호스트 OOM.
4. **재실행은 대부분 저렴하다.** HF 모델·샘플 프레임·TVM 튜닝 로그는 디스크에 캐시된다. 외부 SIGKILL로 죽어도 그냥 다시 돌리면 된다(`benchmark_results.json`/`demo_outputs`는 마지막에 한 번에 쓰여서 부분 손상 위험 낮음).
5. **TVM 빌드 디렉토리(`apache-tvm-src/build`)를 지우지 마라.** 재빌드는 무겁고 위험하다. 이미 정상 동작하는 v0.14.0 빌드가 있다.
6. **이미 정상 결과(`benchmark_results.json`, `demo_outputs/*.mp4`)가 2026-06-18자로 존재한다.** 재실행은 검증/갱신 목적이며, 결과가 안 나와도 기존 산출물은 보존됨.
7. memguard가 `HOST_PRESSURE`로 abort하면 **우리 잘못이 아니다.** 호스트가 한가해질 때까지 기다렸다 재시도하라.

---

## 6. 환경 사실 (Reference)

- GPU: A100-SXM4-80GB (실행 시점 idle, 메모리 ~1 MiB 사용).
- cgroup: `memory.max = 128 GiB`, `cpu.max = 64 core`, `pids.max = max`, 컨테이너는 cgroup-namespace root(`0::/`)로 보임.
- 호스트: RAM 1 TiB / 256 visible core, **다른 유저 ~70 vCPU·640 GB 사용 중** (공유).
- ⚠️ `nproc` = 256, `multiprocessing.cpu_count()` = 256 이지만 **실제 CPU quota는 64**. 도구들이 256을 따라가는 게 이 환경의 핵심 함정.
- TVM: 0.14.0 (Relay+AutoTVM+MetaSchedule+CUDA), 소스 빌드, `runs/a100_smolvlm2_2b/apache-tvm-src`.
- Python venv: `runs/a100_smolvlm2_2b/.venv` (torch 2.5.1+cu121).
