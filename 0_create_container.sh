docker create --name vllm --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /nfs/cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf:/root/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf vllm_build:v1.1
# docker create --name muxserve --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel 

docker create --name vllm --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/txu/.cache:/root/.cache -v /home/txu/workspace:/root/workspace -v /home/txu/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1

docker create --name vllm-spec --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /home/wychen/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1

docker create --name vllm_wychen --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /home/wychen/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1