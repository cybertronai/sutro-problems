#!/bin/bash
# Container entrypoint for a wikitext submission run.
#
# Stages WikiText-103 if missing, verifies NVML, then runs the user's
# submission via run_eval.py. On success writes /results/result.json
# (atomic mv from .tmp). On failure touches /results/FAIL and dumps
# the relevant log to /results/error.log so the orchestrator can show it.
#
# Two sentinel files (result.json or FAIL) terminate the orchestrator's
# blocking SSH wait. /results/ is bind-mounted from the host.

mkdir -p /results

fail() {
    echo "[entrypoint] FAIL: $1" >&2
    touch /results/FAIL
    exit 1
}

# Stage data (no-op if already present).
if [ ! -f /data/wiki.train.raw ]; then
    echo "[entrypoint] fetching WikiText-103 ..."
    python3 fetch_data.py /data 2>/results/error.log || fail "data fetch failed; see error.log"
fi

# NVML probe — bail before burning training cycles if the energy
# counter isn't exposed on this host.
echo "[entrypoint] verifying NVML energy counter ..."
python3 verify_nvml.py > /results/nvml.log 2>&1 || {
    cp /results/nvml.log /results/error.log
    fail "verify_nvml.py failed; see nvml.log"
}
# verify_nvml prints a JSON summary as its final stdout line.
tail -1 /results/nvml.log > /results/nvml.json

# Pull task-pinned values from task.py (single source of truth — submitters
# do not get to vary these on the way in).
TEST_CHARS=$(python3 -c "import task; print(task.TEST_CHARS)")
E_MAX=$(python3 -c "import task; print(task.E_MAX_JOULES if task.E_MAX_JOULES is not None else '')")

EVAL_ARGS=(--max-test-chars "$TEST_CHARS")
[ -n "$E_MAX" ] && EVAL_ARGS+=(--e-max-joules "$E_MAX")

# Train + eval.
echo "[entrypoint] running submission (task: TEST_CHARS=$TEST_CHARS E_MAX=${E_MAX:-unset}) ..."
python3 run_eval.py \
    --data-dir /data \
    --submission submission.py \
    --results-json /results/result.json.tmp \
    "${EVAL_ARGS[@]}" \
    > /results/run.log 2>&1 || {
        cp /results/run.log /results/error.log
        fail "run_eval.py failed; see run.log"
    }

sync && mv /results/result.json.tmp /results/result.json
echo "[entrypoint] done."
