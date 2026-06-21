#!/usr/bin/env bash
# Run the SmolVLM2 A100 pipeline under an active memory guard.
#
# WHY THIS EXISTS / POST-MORTEM
# -----------------------------
# A previous run died with exit code 137 (SIGKILL). It was NOT a cgroup OOM:
#   /sys/fs/cgroup/memory.events showed oom_kill=0, oom=0, max=0, and
#   memory.pressure was flat zero. The pipeline log cut off cleanly right after
#   "Loading checkpoint shards: 100%" with no Python traceback.
#
# Root cause: this container's cgroup memory limit (128 GiB) is LARGER than the
# host's physically-free RAM, and other tenants on the node use ~640 GB. When
# the node runs short, the HOST-level (global) OOM killer / kubelet node-pressure
# eviction picks a victim by badness score and can SIGKILL us even though our own
# usage is small. A global OOM does not increment our cgroup's memory.events,
# which is exactly the "usage was low but it still died" signature we saw.
#
# Therefore guarding only our own cgroup usage (the old 110 GiB threshold, set
# against the 128 GiB cgroup limit) is useless here. This guard watches BOTH:
#   1. our own cgroup memory.current  -> trip if WE run away
#   2. host MemAvailable              -> abort CLEANLY if the NODE gets dangerous,
#                                        so we exit and write status instead of
#                                        being SIGKILLed mid-write.
# Keeping our footprint small also lowers our OOM badness score, which makes the
# kernel prefer a larger neighbor process over us if the node does OOM.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOL_DIR="${ROOT_DIR}/smolvlm2_optimization"
WORK_DIR="${WORK_DIR:-${SMOL_DIR}/runs/a100_smolvlm2_2b}"
LOG_FILE="${1:-${WORK_DIR}/run.log}"
GUARD_LOG="${GUARD_LOG:-${WORK_DIR}/memguard.log}"
STATUS_FILE="${STATUS_FILE:-${WORK_DIR}/memguard.status}"

# Our own ceiling. A 2B bf16 model lives on the GPU and the TVM host-side
# Relay build stays well under 20 GiB, so 50 GiB is a generous runaway tripwire.
SELF_LIMIT_GIB="${SELF_LIMIT_GIB:-50}"
# Host floor. If node-wide available RAM drops below this, abort cleanly rather
# than risk being SIGKILLed by the global OOM killer.
HOST_FLOOR_GIB="${HOST_FLOOR_GIB:-25}"
# Sampling interval (seconds). The previous death happened between 10 s samples,
# so sample faster.
INTERVAL="${INTERVAL:-3}"

SELF_LIMIT=$((SELF_LIMIT_GIB * 1024 * 1024 * 1024))
MEM_CUR="/sys/fs/cgroup/memory.current"

mkdir -p "${WORK_DIR}"

ts() { date '+%Y-%m-%dT%H:%M:%S'; }
host_avail_gib() { awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo; }

setsid bash "${SMOL_DIR}/run_smolvlm2_a100_pipeline.sh" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "[memguard $(ts)] started pid=${PID} self_limit=${SELF_LIMIT_GIB}GiB host_floor=${HOST_FLOOR_GIB}GiB interval=${INTERVAL}s log=${LOG_FILE}" | tee -a "${GUARD_LOG}"
echo "RUNNING pid=${PID} started=$(ts)" >"${STATUS_FILE}"

reason=""
while kill -0 "${PID}" 2>/dev/null; do
  cur=$(cat "${MEM_CUR}" 2>/dev/null || echo 0)
  cur_gib=$((cur / 1024 / 1024 / 1024))
  havail_gib=$(host_avail_gib)
  echo "[memguard $(ts)] self=${cur_gib}GiB host_avail=${havail_gib}GiB" >>"${GUARD_LOG}"
  if [ "${cur}" -gt "${SELF_LIMIT}" ]; then
    reason="SELF_RUNAWAY self=${cur_gib}GiB > ${SELF_LIMIT_GIB}GiB"
    break
  fi
  if [ -n "${havail_gib}" ] && [ "${havail_gib}" -lt "${HOST_FLOOR_GIB}" ]; then
    reason="HOST_PRESSURE host_avail=${havail_gib}GiB < ${HOST_FLOOR_GIB}GiB"
    break
  fi
  sleep "${INTERVAL}"
done

if [ -n "${reason}" ]; then
  echo "[memguard $(ts)] TRIP: ${reason} -> stopping pipeline pgid=${PID}" | tee -a "${GUARD_LOG}"
  echo "ABORTED ${reason} at=$(ts)" >"${STATUS_FILE}"
  kill -TERM -- "-${PID}" 2>/dev/null
  sleep 5
  kill -KILL -- "-${PID}" 2>/dev/null
  exit 99
fi

wait "${PID}"
status=$?
echo "[memguard $(ts)] pipeline exited status=${status}" | tee -a "${GUARD_LOG}"
echo "DONE status=${status} at=$(ts)" >"${STATUS_FILE}"
exit "${status}"
