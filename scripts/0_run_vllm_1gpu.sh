#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# cd ../
# bash compile.sh
# cd scripts
# pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
gpu_id=0,1,2,3
tensor_parallel_size=4
gpu_memory_utilizations=(0.9)
preemption_mode=recompute
scheduling_policy=fcfs
wt_weight=1.0
flatten_layers=4
store_cache_layers=0.1

models=(deepseek-ai/DeepSeek-V2)
models=(meta-llama/Llama-2-70b-chat-hf)
max_num_seqs=512
data_name=sharegpt
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json

req_rates_csv='/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week_count.csv'  
n=60
scale=1

if (( n > 0 )); then
    lines=$(sed -n "2,$((n+1))p" "$req_rates_csv")
else
    lines=$(tail -n +2 "$req_rates_csv")
fi
request_rates_str=$(echo "$lines" | cut -d',' -f2 \
                    | awk -v s="$scale" '{print int($1*s+0.5)}' \
                    | paste -sd ',' -)
echo "after scaled: request_rates_str=${request_rates_str}"                    

num_prompt=$(echo "$lines" | cut -d',' -f2 \
             | awk -v s="$scale" '{sum+=int($1*s+0.5)} END{print sum}')

echo "num_prompt: $num_prompt"

log_path="${SCRIPT_DIR}/dataset/slo/ellm_ds/${data_name}"
if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi

wait_for_server() {
    local port=$1
    while true; do
        if netstat -tulnp | grep -q "${port}"; then
            echo "server is running on port ${port}"
            break
        else
            echo "server is not running on port ${port}"
            sleep 5
        fi
    done
}

for run in {1..1}; do
    for model_idx in "${!models[@]}"; do
        model="${models[$model_idx]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$model_idx]}"
        model_name=$(echo "$model" | tr '/' '_')

        CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
            --model ${model} \
            --port 8080 \
            --tensor-parallel-size ${tensor_parallel_size} \
            --trust-remote-code \
            --enforce-eager \
            --worker-use-ray \
            --load-format dummy \
            --gpu-memory-utilization ${gpu_memory_utilization} \
            --max-num-seqs ${max_num_seqs} \
            --disable-log-requests > "${log_path}/${model_name}_server_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.log" & 
        pid=$!

        if ! wait_for_server 8080; then
            kill -9 $pid
            exit 1
        fi

        sleep 1
        # sm_log="${log_path}/${model_name}_sm_util_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.csv"
        # bash "${SCRIPT_DIR}/collect_sm_utilization.sh" \
        #     --pid "$pid" \
        #     --gpu-id "$gpu_id" \
        #     --out "$sm_log" &
        # sm_pid=$!

        python3 ../benchmarks/benchmark_serving_dynamic.py \
            --model ${model} \
            --port 8080 \
            --dataset ${dataset_path} \
            --request-rates "${request_rates_str}" \
            --num-prompts ${num_prompt} \
            --result-dir results/swap_recompute \
            --endpoint /v1/completions >> "${log_path}/${model_name}_client_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.log"

        kill $pid || kill -9 $pid
        # wait $sm_pid || true
        sleep 5
    done
done
