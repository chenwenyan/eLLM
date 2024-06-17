# cd ../
# bash compile.sh
# cd examples

# pgrep -f 'api_server' | xargs kill -9

#!/bin/bash

preemption_mode1=recompute # 1: swap 2: recomputation
preemption_mode2=recompute
gpu_id=2
gpu_memory_utilization=0.3 # Use a single variable for both models
gpu_memory_utilization_2=0.6

# model1=facebook/opt-6.7b
# model2=meta-llama/Llama-2-13b-hf
model1=facebook/opt-2.7b
model2=facebook/opt-13b
request_rates=(5 10 15 20 25 30 35 40 45 50)
num_prompts=(100)
port1=8070
port2=8071
dataset=../dataset/ShareGPT_V3_unfiltered_cleaned_split.json
LOG_DIR=logs/1_colorun_llm_1gpu/colo_swap_recompute_various_req_rate_models

# Define the models array
models=($model1 $model2)

# Ensure models array has at least two elements
if [ ${#models[@]} -lt 2 ]; then
    echo "Error: models array must contain at least two elements."
    exit 1
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

IFS='/' read -ra parts1 <<< "$model1"
model1_name=${parts1[-1]}
IFS='/' read -ra parts2 <<< "$model2"
model2_name=${parts2[-1]}

# Start the first model's server
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model ${model1} \
    --port ${port1} \
    --tensor-parallel-size 1 \
    --swap-space 16 \
    --preemption-mode ${preemption_mode1} \
    --gpu-memory-utilization ${gpu_memory_utilization} \
    --disable-log-requests > ${LOG_DIR}/${model1_name}-${model2_name}/${model1_name}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode1}_${preemption_mode2}.log & 
pid1=$!
sleep 10
wait_for_server ${port1}


# Start the second model's server
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model ${model2} \
    --port ${port2} \
    --tensor-parallel-size 1 \
    --swap-space 16 \
    --preemption-mode ${preemption_mode2} \
    --gpu-memory-utilization ${gpu_memory_utilization_2} \
    --disable-log-requests > ${LOG_DIR}/${model1_name}-${model2_name}/${model2_name}_server_${gpu_memory_utilization_2}_${request_rate}_${num_prompt}_${preemption_mode2}_${preemption_mode1}.log & 
pid2=$!    
sleep 10
wait_for_server ${port2}


for num_prompt in ${num_prompts[@]}; do
    for request_rate in ${request_rates[@]}; do
        python3 ../benchmarks/benchmark_serving.py \
            --model ${model1} \
            --port ${port1} \
            --dataset ${dataset} \
            --request-rate ${request_rate} \
            --num-prompts ${num_prompt} \
            --result-dir results/1_colorun_llm_1gpu \
            --endpoint /v1/completions > ${LOG_DIR}/${model1_name}-${model2_name}/${model1_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode1}_${preemption_mode2}.log &
        pid3=$!    

        # Run the benchmark for the second model
        python3 ../benchmarks/benchmark_serving.py \
            --model ${model2} \
            --port ${port2} \
            --dataset ${dataset} \
            --request-rate ${request_rate} \
            --num-prompts ${num_prompt} \
            --result-dir results/1_colorun_llm_1gpu \
            --endpoint /v1/completions > ${LOG_DIR}/${model1_name}-${model2_name}/${model2_name}_client_${gpu_memory_utilization_2}_${request_rate}_${num_prompt}_${preemption_mode2}_${preemption_mode1}.log &
        pid4=$!    

        wait $pid3 $pid4
        sleep 5
    done    
done

kill -9 $pid1
kill -9 $pid2