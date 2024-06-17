cd ../
bash compile.sh
cd examples

# pgrep -f 'api_server' | xargs kill -9

preemption_mode=2 # 1: swap 2: recomputation
gpu_id=2
gpu_memory_utilization=0.9

models=(facebook/opt-30b)
request_rates=(10)
num_prompts=(100)

for model in ${models[@]}; do
    for request_rate in ${request_rates[@]}; do
        for num_prompt in ${num_prompts[@]}; do
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                --model ${model} \
                --port 8080 \
                --tensor-parallel-size 1 \
                --swap-space 16 \
                --gpu-memory-utilization ${gpu_memory_utilization} \
                --preemption-mode ${preemption_mode} --disable-log-requests > logs/0_solorun_llm_1gpu/${model}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}.log & 
            pid=$!    
            sleep 800
            python3 ../benchmarks/benchmark_serving.py \
                --model ${model} \
                --port 8080 \
                --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
                --request-rate ${request_rate} \
                --num-prompts ${num_prompt} \
                --save-result \
                --result-dir results/0_solorun_llm_1gpu \
                --endpoint /v1/completions > logs/0_solorun_llm_1gpu/${model}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_${preemption_mode}.log 
            kill -9 $pid   
        done
    done
done
