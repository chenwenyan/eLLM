#!/bin/bash

cd ../
bash compile.sh
cd scripts
pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

gpu_id=0
tensor_parallel_size=1
gpu_memory_utilizations=(0.6)
preemption_mode=recompute
scheduling_policy=fcfs
wt_weight=1.0
flatten_layers=4
store_cache_layers=0.1

models=(meta-llama/Llama-2-13b-chat-hf)
max_num_seqs=512
data_name=paper_assistant
# dataset_path=/nfs/dataset/LEval/LEval-data/Open-ended-tasks/${data_name}_transformed.json
dataset_path=/nfs/dataset/${data_name}_transformed.json

req_rates_csv='/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week_count.csv'  
n=60

log_path='/root/workspace/vllm-dynamic/scripts/dataset/slo/ellm/'${data_name}
if [ ! -d "${log_path}" ]; then
    mkdir -p ${log_path}
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

scales=(0.5 1 1.5 2 2.5 3)

# 2. 在 for run ... 之前再套一层 for scale in ...
for scale in "${scales[@]}"; do
    echo "========= running with scale = $scale ========="

    request_rates_str=$(sed -n "2,$((n+1))p" "$req_rates_csv" | cut -d',' -f2 | paste -sd ',' -)
    echo "$request_rates_str"

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

    for run in {1..1}; do
        for model_idx in "${!models[@]}"; do
            model="${models[$model_idx]}"
            gpu_memory_utilization="${gpu_memory_utilizations[$model_idx]}"
            model_name=$(echo "$model" | tr '/' '_')

            GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0 CUDA_LAUNCH_BLOCKING=1 RAY_DEDUP_LOGS=0 CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                --model ${model} \
                --port 8081 \
                --distributed-executor-backend=ray \
                --tensor-parallel-size ${tensor_parallel_size} \
                --swap-space 10 \
                --enforce-eager \
                --gpu-memory-utilization ${gpu_memory_utilization} \
                --max-num-seqs ${max_num_seqs} \
                --store-cache-layers ${store_cache_layers} \
                --preemption-mode ${preemption_mode} \
                --scheduling-policy ${scheduling_policy} \
                --wt-weight ${wt_weight} \
                --preemption-mode ${preemption_mode} \
                --flatten-layers ${flatten_layers} \
                --disable-log-requests > "${log_path}/${model_name}_server_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.log" & 
            pid=$!

            if ! wait_for_server 8081; then
                kill -9 $pid
                exit 1
            fi

            sleep 1

            python3 ../benchmarks/benchmark_serving_dynamic.py \
                --model ${model} \
                --port 8081 \
                --dataset ${dataset_path} \
                --request-rates "${request_rates_str}" \
                --num-prompts ${num_prompt} \
                --result-dir results/swap_recompute \
                --endpoint /v1/completions >> "${log_path}/${model_name}_client_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.log"

            kill $pid || kill -9 $pid
            sleep 5
        done
    done
done    