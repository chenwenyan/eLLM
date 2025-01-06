cd ../
bash compile.sh
cd examples

pip uninstall -y vllm-flash-attn

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=on
export CUDA_LAUNCH_BLOCKING_WAIT=1
export VLLM_LOGGING_LEVEL=DEBUG
# export NCCL_DEBUG=TRACE
# export VLLM_TRACE_FUNCTION=1


# pgrep -f 'api_server' | xargs kill -9

preemption_mode=swap # 1: swap 2: recomputation
gpu_id=3
# gpu_memory_utilizations=(0.1)
# gpu_memory_utilizations=(0.2)
gpu_memory_utilizations=(0.6)
# gpu_memory_utilizations=(0.7)
# gpu_memory_utilizations=(0.9)
# store_cache_layerss=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# store_cache_layerss=(0.0625 0.125 0.25 0.5) # 32层 for llama2-7B
# store_cache_layerss=(0.05 0.1 0.2 0.25 0.5) # 40层 for llama2-13B
store_cache_layerss=(0.5)


# models=(facebook/opt-30b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf)
# models=(facebook/opt-2.7b)
# models=(meta-llama/Llama-2-7b-hf)
models=(meta-llama/Llama-2-13b-hf)
# request_rates=(50 100 150 200 250 300)
request_rates=(5)
num_prompts=(100)
seeds=(11 12 13 14 15 16 17 18 19 20)
max_num_seqs=512
# max_num_seqs=1024
dataset_path=/nfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
scheduling_policy=fcfs
wt_weights=(1.0)
# wt_weights=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)


rm server.log
rm client.log

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

threads=32
path_dir=logs/dllm_org_128_multi_hfusion_${threads}
mkdir -p $path_dir
# for i in {1..1}; do
for seed in ${seeds[@]}; do
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        gpu_memory_utilization="${gpu_memory_utilizations[$i]}"
        model_name=$(echo "$model" | tr '/' '_')
        for request_rate in ${request_rates[@]}; do
            for num_prompt in ${num_prompts[@]}; do
                for store_cache_layers in ${store_cache_layerss[@]}; do
                    for wt_weight in ${wt_weights[@]}; do
                        CUDA_VISIBLE_DEVICES=${gpu_id} taskset -c 2-3 python3 -m vllm.entrypoints.openai.api_server \
                            --model ${model} \
                            --port 8081 \
                            --tensor-parallel-size 1 \
                            --swap-space 4 \
                            --gpu-memory-utilization ${gpu_memory_utilization} \
                            --store-cache-layers ${store_cache_layers} \
                            --max-num-seqs ${max_num_seqs} \
                            --enforce-eager \
                            --scheduling-policy ${scheduling_policy} \
                            --wt-weight ${wt_weight} \
                            --preemption-mode ${preemption_mode} \
                            --disable-log-requests >> server.log 2>&1 &
                            # > ${path_dir}/${model_name}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${scheduling_policy}_wt${wt_weight}.log & 
                        # > server.log 2>&1 &
                        pid=$!    
                        wait_for_server 8081
                        sleep 1

                        python3 ../benchmarks/benchmark_serving.py \
                            --model ${model} \
                            --port 8081 \
                            --dataset ${dataset_path} \
                            --request-rate ${request_rate} \
                            --num-prompts ${num_prompt} \
                            --seed ${seed} \
                            --endpoint /v1/completions > client.log 
                            # >> ${path_dir}/${model_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}_${store_cache_layers}_${scheduling_policy}_wt${wt_weight}.log    
                        # > client.log     
                        kill -9 $pid 
                        sleep 1
                    done
                done      
            done
        done
    done
done    