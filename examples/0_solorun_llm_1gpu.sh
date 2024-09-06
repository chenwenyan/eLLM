# cd ../
# bash compile.sh
# cd examples

pgrep -f 'api_server' | xargs kill -9

preemption_mode=2 # 1: swap 2: recomputation
gpu_id=2
gpu_memory_utilization=0.2

models=(facebook/opt-6.7b JackFram/llama-68m meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf)
models=(facebook/opt-2.7b)
request_rates=(20)
num_prompts=(100)

for model in ${models[@]}; do
    for request_rate in ${request_rates[@]}; do
        for num_prompt in ${num_prompts[@]}; do
            CUDA_VISIBLE_DEVICES=${gpu_id} taskset -c 12-13 python3 -m vllm.entrypoints.openai.api_server \
                --model ${model} \
                --port 8080 \
                --tensor-parallel-size 1 \
                --swap-space 4 \
                --gpu-memory-utilization ${gpu_memory_utilization} \
                --preemption-mode ${preemption_mode} --disable-log-requests > logs/0_solorun_llm_1gpu/${model}_server_${gpu_memory_utilization}_${request_rate}_${num_prompt}_test.log & 
            pid=$!    
            sleep 100
            python3 ../benchmarks/benchmark_serving.py \
                --model ${model} \
                --port 8080 \
                --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
                --request-rate ${request_rate} \
                --num-prompts ${num_prompt} \
                --save-result \
                --result-dir results/0_solorun_llm_1gpu \
                --endpoint /v1/completions >> logs/0_solorun_llm_1gpu/${model}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}_test.log 
            kill -9 $pid   
        done
    done
done
