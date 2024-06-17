# cd ../
# bash compile.sh
# cd examples
preemption_mode=2 # 1: swap 2: recomputation
gpu_id=2

# CUDA_VISIBLE_DEVICES=${gpu_id} 
python3 ../benchmarks/benchmark_serving.py \
    --model facebook/opt-6.7b \
    --port 8080 \
    --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 50 \
    --endpoint /v1/completions > logs/1_colorun_llm_1gpu/opt-6.7b_client.log & 

python3 ../benchmarks/benchmark_serving.py \
    --model JackFram/llama-68m \
    --port 8081 \
    --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 50 \
    --endpoint /v1/completions > logs/1_colorun_llm_1gpu/llama-68m_client.log &    