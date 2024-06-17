gpu_id=3
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf --swap-space 16 --disable-log-requests &