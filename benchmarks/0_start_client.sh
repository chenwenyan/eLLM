gpu_id=3
CUDA_VISIBLE_DEVICES=${gpu_id} python3 benchmark_serving.py \
    --model meta-llama/Llama-2-70b-chat-hf \
    --dataset ../dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 10 \
    --num-prompts 1000 
    # --endpoint /v1/completions 