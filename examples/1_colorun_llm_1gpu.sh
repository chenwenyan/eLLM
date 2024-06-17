# cd ../
# bash compile.sh
# cd examples

# pgrep -f 'api_server' | xargs kill -9

preemption_mode=2 # 1: swap 2: recomputation
gpu_id=3
gpu_memory_utilization1=0.6
gpu_memory_utilization2=0.3

models=(facebook/opt-6.7b JackFram/llama-68m meta-llama/Llama-2-13b-hf meta-llama/Llama-2-7b-hf )
request_rates=(1 2 5 10)
num_prompts=(50 100 150 200)

# 确保 models 数组至少有两个元素
if [ ${#models[@]} -lt 2 ]; then
    echo "Error: models array must contain at least two elements."
    exit 1
fi

for i in {0..2}; do
    for request_rate in ${request_rates[@]}; do
        for num_prompt in ${num_prompts[@]}; do
            model1=${models[$i]}
            model2=${models[$((i+1))]}  # 确保取出的不是同一个模型

            # 启动第一个模型的 server
            IFS='/' read -ra parts1 <<< "$model1"
            IFS='/' read -ra parts2 <<< "$model2"
            model1_name=${parts1[-1]}
            model2_name=${parts2[-1]}

            CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                --model ${model1} \
                --port 8090 \
                --tensor-parallel-size 1 \
                --swap-space 4 \
                --gpu-memory-utilization ${gpu_memory_utilization2} \
                --preemption-mode ${preemption_mode} --disable-log-requests > logs/1_colorun_llm_1gpu/${model1}_${model2_name}_server_${gpu_memory_utilization1}_${request_rate}_${num_prompt}.log & 
            pid1=$!
            sleep 50

            # 启动第二个模型的 server
            CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
                --model ${model2} \
                --port 8091 \
                --tensor-parallel-size 1 \
                --swap-space 4 \
                --gpu-memory-utilization ${gpu_memory_utilization2} \
                --preemption-mode ${preemption_mode} --disable-log-requests > logs/1_colorun_llm_1gpu/${model2}_${model1_name}_server_${gpu_memory_utilization2}_${request_rate}_${num_prompt}.log & 
            pid2=$!    

            sleep 100  # 根据实际情况调整等待时间

            # 启动第一个模型的 client
            python3 ../benchmarks/benchmark_serving.py \
                --model ${model1} \
                --port 8090 \
                --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
                --request-rate ${request_rate} \
                --num-prompts ${num_prompt} \
                --save-result \
                --result-dir results/1_colorun_llm_1gpu \
                --endpoint /v1/completions > logs/1_colorun_llm_1gpu/${model1}_${model2_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}.log & 

            # 启动第二个模型的 client
            python3 ../benchmarks/benchmark_serving.py \
                --model ${model2} \
                --port 8091 \
                --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
                --request-rate ${request_rate} \
                --num-prompts ${num_prompt} \
                --save-result \
                --result-dir results/1_colorun_llm_1gpu \
                --endpoint /v1/completions > logs/1_colorun_llm_1gpu/${model2}_${model1_name}_client_${gpu_memory_utilization}_${request_rate}_${num_prompt}.log & 

            # 等待足够的时间以便完成测试
            sleep 400       

            # 杀死所有 api_server 进程
            # pgrep -f 'api_server' | xargs kill -9 
            kill -9 ${pid1} ${pid2}   
        done
    done
done
