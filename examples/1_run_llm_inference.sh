cd ../
bash compile.sh
cd examples
preemption_mode=2 # 1: swap 2: recomputation
gpu_id=0,1
# CUDA_VISIBLE_DEVICES=${gpu_id} python3 llm_engine_example.py --model facebook/opt-6.7b --preemption-mode ${preemption_mode} --gpu-memory-utilization 0.2 > 1_run_opt_6.7b_inference_${preemption_mode}.log 2>&1 &

CUDA_VISIBLE_DEVICES=${gpu_id} python3 llm_engine_example.py \
    --model facebook/opt-30b \
    --preemption-mode ${preemption_mode} --gpu-memory-utilization 1 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --swap-space 16 > 1_run_opt_30b_inference_${preemption_mode}_solo.log 