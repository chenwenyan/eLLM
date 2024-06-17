gpu_id=3
# CUDA_VISIBLE_DEVICES=${gpu_id} 
python3 ../benchmarks/benchmark_serving.py \
    --model facebook/opt-13b \
    --port 8080 \
    --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 50 \
    --endpoint /v1/completions > 2_multi_gpu_opt_30b_client.log & 

python3 ../benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-2-13b-hf \
    --port 8081 \
    --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 50 \
    --endpoint /v1/completions > 2_multi_gpu_Llama-2-13b-hf_client.log &    