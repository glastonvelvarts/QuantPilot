#!/usr/bin/env bash
# Start vLLM once per setting, keep its start-up log, stop it when it serves (Linux or WSL2).
#   bash scripts/vllm_check.sh <results folder> [model]
# Each run also saves what QuantPilot predicted (ocq feasibility) next to vLLM's log.
# VENV=<folder> uses the ocq and vllm of that virtual environment; ONLY=<name> runs one setting;
# SHARE=<0.xx> is the --gpu-memory-utilization of the runs after the default one (0.8).
set -u
[ -n "${VENV:-}" ] && PATH="$VENV/bin:$PATH"
# HF_TOKEN for gated models and faster downloads, from the repo's .env when there is one.
[ -f .env ] && { set -a; . ./.env; set +a; }
out="$1/vllm"
model="${2:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"
share="${SHARE:-0.8}"
util="util${share#0.}"
mkdir -p "$out"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,compute_cap --format=csv > "$out/gpu.csv"
start_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
# Without the CUDA toolkit (nvcc), FlashInfer cannot build its kernels: vLLM's own sampler, and
# Triton attention for an FP8 cache (FlashAttention 2 has none before Hopper).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

run() {
    name="$1"; shift
    [ -n "${ONLY:-}" ] && [ "$name" != "$ONLY" ] && return
    COLUMNS=160 ocq feasibility "$model" "${@:2}" > "$out/$name-ocq.txt" 2>&1
    nvidia-smi --query-gpu=memory.free --format=csv,noheader >> "$out/$name-ocq.txt"
    vllm serve "$model" --port 8299 $1 > "$out/$name.log" 2>&1 &
    pid=$!
    for _ in $(seq 600); do
        grep -q "Application startup complete\|Error\|error:" "$out/$name.log" && break
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    # After a failed start the API server ignores SIGTERM.
    pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f "VLLM::EngineCore" 2>/dev/null
    wait $pid 2>/dev/null
    # The next prediction needs the VRAM back.
    for _ in $(seq 180); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -ge $((start_free - 200)) ] && break
        sleep 1
    done
    echo "== $name"
    grep -E "Free memory|Model loading took|memory profiling|Available KV cache|KV cache size|Maximum concurrency|CUDA graph|startup complete|Error" "$out/$name.log" | cut -c1-260
}

run default-4k "--max-model-len 4096" -c 4096
run $util-4k "--max-model-len 4096 --gpu-memory-utilization $share" -c 4096
run $util-fp8-kv-32k "--max-model-len 32768 --gpu-memory-utilization $share --kv-cache-dtype fp8 --attention-backend TRITON_ATTN" -c 32768 --kv-cache q8_0
run $util-f16-kv-32k "--max-model-len 32768 --gpu-memory-utilization $share --attention-backend TRITON_ATTN" -c 32768
