#!/bin/bash

cd ../
bash compile.sh
cd scripts
pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

gpu_id=0,1,2,3
tensor_parallel_size=4
gpu_memory_utilizations=(0.7)
preemption_mode=recompute
scheduling_policy=fcfs
wt_weight=1.0
flatten_layers=8
store_cache_layers=0.1

models=(meta-llama/Llama-2-70b-chat-hf)
max_num_seqs=512
data_name=sharegpt
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json

duration=60
req_rates_csv='/nfs/dataset/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week_milliseconds.csv'  
request_rates=(
    $(tail -n +2 "$req_rates_csv" | cut -d',' -f4 |
    awk -F, '{print int($1/2000)}' |
    sort -n | uniq -c |
    awk '{print $1}')
)
request_rates=("${request_rates[@]:0:$duration}")
echo "request_rates=(${request_rates[@]})"
# ，拼接request_rates为字符串
request_rates_str=$(printf "%s," "${request_rates[@]}")
request_rates_str=${request_rates_str%,} # 去掉最后一个逗号

num_prompt=0
for ((i=0; i<${#request_rates[@]}; i++)); do
    num_prompt=$((num_prompt + request_rates[i]))
done
echo "num_prompt: $num_prompt"

log_path='/root/workspace/vllm-dynamic/scripts/dataset/submodule/ellm/disable_overlap/'${data_name}
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

for run in {1..3}; do
    for model_idx in "${!models[@]}"; do
        model="${models[$model_idx]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$model_idx]}"
        model_name=$(echo "$model" | tr '/' '_')

        CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
            --model ${model} \
            --port 8080 \
            --tensor-parallel-size ${tensor_parallel_size} \
            --swap-space 40 \
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

        if ! wait_for_server 8080; then
            kill -9 $pid
            exit 1
        fi

        sleep 1

        python3 ../benchmarks/benchmark_serving_dynamic.py \
            --model ${model} \
            --port 8080 \
            --dataset ${dataset_path} \
            --request-rates "${request_rates_str}" \
            --num-prompts ${num_prompt} \
            --result-dir results/swap_recompute \
            --endpoint /v1/completions >> "${log_path}/${model_name}_client_${num_prompt}_${preemption_mode}_${tensor_parallel_size}gpu.log"

        kill $pid || kill -9 $pid
        sleep 5
    done
done