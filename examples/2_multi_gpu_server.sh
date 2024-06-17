gpu_id=2,3
preemption_mode=2 # 1: swap 2: recomputation
pgrep -f 'api_server' | xargs kill -9
CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-30b \
    --port 8080 \
    --tensor-parallel-size 2 \
    --swap-space 16 \
    # --gpu-memory-utilization 0.5 \
    --preemption-mode ${preemption_mode} --disable-log-requests > 2_multi_gpu_opt_30b_server.log & 

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-13b-hf \
    --port 8081 \
    --tensor-parallel-size 2 \
    --swap-space 16 \
    # --gpu-memory-utilization 0.4 \
    --preemption-mode ${preemption_mode} --disable-log-requests > 2_multi_gpu_Llama-2-13b-hf_server.log & 