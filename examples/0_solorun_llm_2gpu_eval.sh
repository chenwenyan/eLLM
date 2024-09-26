cd ../
bash compile.sh
cd examples

pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# pgrep -f 'api_server' | xargs kill -9

preemption_mode=swap # 1: swap 2: recomputation
gpu_id=2,3
tensor_parallel_size=2
# gpu_memory_utilizations=(0.9 0.2 0.4)
gpu_memory_utilizations=(1.0)
store_cache_layerss=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# store_cache_layerss=(0.1)

# models=(facebook/opt-30b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf)
models=(meta-llama/Llama-2-70b-hf)
request_rates=(20)
num_prompts=(300)
max_num_seqs=256
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json

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

for i in "${!models[@]}"; do
    model="${models[$i]}"
    gpu_memory_utilization="${gpu_memory_utilizations[$i]}"
    model_name=$(echo "$model" | tr '/' '_')
    for request_rate in ${request_rates[@]}; do
        for num_prompt in ${num_prompts[@]}; do
            for store_cache_layers in ${store_cache_layerss[@]}; do
                CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                    --model ${model} \
                    --port 8080 \
                    --tensor-parallel-size ${tensor_parallel_size} \
                    --swap-space 4 \
                    --gpu-memory-utilization ${gpu_memory_utilization} \
                    --store-cache-layers ${store_cache_layers} \
                    --max-num-seqs ${max_num_seqs} \
                    --preemption-mode ${preemption_mode} --disable-log-requests > server.log 2>&1 &
                    # > logs/swap_recompute/${model_name}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${tensor_parallel_size}gpu.log & 
                    
                    # 
                pid=$!    
                wait_for_server 8080
                sleep 5

                python3 ../benchmarks/benchmark_serving.py \
                    --model ${model} \
                    --port 8080 \
                    --dataset ${dataset_path} \
                    --request-rate ${request_rate} \
                    --num-prompts ${num_prompt} \
                    --save-result \
                    --result-dir results/swap_recompute \
                    --endpoint /v1/completions > client.log
                    # > logs/swap_recompute/${model_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${tensor_parallel_size}gpu.log 
                    # 
                    # 
                kill -9 $pid 
                sleep 10
            done      
        done
    done
done