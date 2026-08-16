#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
ELLM_HF_HOME="${ELLM_HF_HOME:-/tmp/ellm-hf-cache}"
export HF_HOME="${ELLM_HF_HOME}"

GPU_IDS="${GPU_IDS:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
STORE_CACHE_LAYERS="${STORE_CACHE_LAYERS:-0.5}"
FLATTEN_LAYERS="${FLATTEN_LAYERS:-4}"
SCHEDULING_POLICY="${SCHEDULING_POLICY:-fcfs}"
PREEMPTION_MODE="${PREEMPTION_MODE:-recompute}"

DATASET_PATH="${DATASET_PATH:-/data/wenyan/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
REQUEST_RATES="${REQUEST_RATES:-1}"
SHAREGPT_OUTPUT_LEN="${SHAREGPT_OUTPUT_LEN:-4}"
PORT="${PORT:-8080}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-180}"

if [[ ! -f "${DATASET_PATH}" ]]; then
    echo "ShareGPT dataset not found: ${DATASET_PATH}" >&2
    exit 1
fi

MODEL_NAME="${MODEL//\//_}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/dataset/slo/ellm_ds/sharegpt}"
RESULT_DIR="${RESULT_DIR:-${REPO_ROOT}/results/sharegpt}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${HF_HOME}"

SERVER_LOG="${LOG_DIR}/${MODEL_NAME}_server_${NUM_PROMPTS}_${TENSOR_PARALLEL_SIZE}gpu.log"
CLIENT_LOG="${LOG_DIR}/${MODEL_NAME}_client_${NUM_PROMPTS}_${TENSOR_PARALLEL_SIZE}gpu.log"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local deadline=$((SECONDS + SERVER_TIMEOUT))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "vLLM server exited before becoming ready" >&2
            tail -n 80 "${SERVER_LOG}" >&2 || true
            return 1
        fi
        if curl --fail --silent "http://127.0.0.1:${PORT}/health" \
                >/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "Timed out waiting for vLLM server on port ${PORT}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    return 1
}

echo "Starting ${MODEL} on GPU(s) ${GPU_IDS}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --trust-remote-code \
    --enforce-eager \
    --load-format "${LOAD_FORMAT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --store-cache-layers "${STORE_CACHE_LAYERS}" \
    --flatten-layers "${FLATTEN_LAYERS}" \
    --scheduling-policy "${SCHEDULING_POLICY}" \
    --preemption-mode "${PREEMPTION_MODE}" \
    --disable-log-requests >"${SERVER_LOG}" 2>&1 &
server_pid=$!

wait_for_server
echo "Server ready; running ${NUM_PROMPTS} ShareGPT requests"

python "${REPO_ROOT}/benchmarks/benchmark_serving_dynamic.py" \
    --backend vllm \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path "${DATASET_PATH}" \
    --model "${MODEL}" \
    --trust-remote-code \
    --request-rates "${REQUEST_RATES}" \
    --num-prompts "${NUM_PROMPTS}" \
    --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}" \
    --save-result \
    --result-dir "${RESULT_DIR}" 2>&1 | tee "${CLIENT_LOG}"

echo "Server log: ${SERVER_LOG}"
echo "Client log: ${CLIENT_LOG}"
echo "Results: ${RESULT_DIR}"
