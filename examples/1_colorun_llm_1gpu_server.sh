# cd ../
# bash compile.sh
# cd examples
preemption_mode=2 # 1: swap 2: recomputation
gpu_id=2
pgrep -f 'api_server' | xargs kill -9

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-6.7b \
    --port 8080 \
    --tensor-parallel-size 1 \
    --swap-space 4 \
    --gpu-memory-utilization 0.6 \
    --preemption-mode ${preemption_mode} --disable-log-requests > logs/1_colorun_llm_1gpu/opt-6.7b_server.log & 

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m vllm.entrypoints.openai.api_server \
    --model JackFram/llama-68m \
    --port 8081 \
    --tensor-parallel-size 1 \
    --swap-space 4 \
    --gpu-memory-utilization 0.3 \
    --preemption-mode ${preemption_mode} --disable-log-requests > logs/1_colorun_llm_1gpu/llama-68m_server.log & 