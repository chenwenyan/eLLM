cd ../
bash compile.sh
cd examples

pip uninstall -y vllm-flash-attn

# sleep 4000

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# pgrep -f 'api_server' | xargs kill -9

preemption_mode=recompute # 1: swap 2: recomputation
preemption_mode=recompute # 1: swap 2: recomputation
gpu_id=0,1,2,3
tensor_parallel_size=4
# gpu_memory_utilizations=(0.9 0.2 0.4)
gpu_memory_utilizations=(0.5)
# gpu_memory_utilizations=(0.9)
store_cache_layerss=(0.025 0.05 0.1 0.125 0.25 0.5 1.0)
# store_cache_layerss=(0.1)

# models=(facebook/opt-30b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf)
models=(meta-llama/Llama-2-70b-hf)
request_rates=(300) 
num_prompts=(300)
max_num_seqs=512
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
scheduling_policy=fcfs
wt_weights=(1.0)

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

# 循环10次
for i in {1..5}; do
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$i]}"
        model_name=$(echo "$model" | tr '/' '_')
        for request_rate in ${request_rates[@]}; do
            for num_prompt in ${num_prompts[@]}; do
                for store_cache_layers in ${store_cache_layerss[@]}; do
                    for wt_weight in ${wt_weights[@]}; do
                        CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                            --model ${model} \
                            --port 8080 \
                            --tensor-parallel-size ${tensor_parallel_size} \
                            --swap-space 0 \
                            --gpu-memory-utilization ${gpu_memory_utilization} \
                            --store-cache-layers ${store_cache_layers} \
                            --max-num-seqs ${max_num_seqs} \
                            --preemption-mode ${preemption_mode} \
                            --scheduling-policy ${scheduling_policy} \
                            --wt-weight ${wt_weight} \
                            --disable-log-requests > logs/dllm_org_128_hfusion_64/${model_name}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${scheduling_policy}_wt${wt_weight}.log & 
                            # > server.log 2>&1 &
                            # 
                        pid=$!    
                        wait_for_server 8080
                        sleep 1

                        python3 ../benchmarks/benchmark_serving.py \
                            --model ${model} \
                            --port 8080 \
                            --dataset ${dataset_path} \
                            --request-rate ${request_rate} \
                            --num-prompts ${num_prompt} \
                            --endpoint /v1/completions >> logs/dllm_org_128_hfusion_64/${model_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${scheduling_policy}_wt${wt_weight}.log 
                            # > client.log
                            # 
                        kill -9 $pid 
                        sleep 1
                    done    
                done      
            done
        done
    done
done    
