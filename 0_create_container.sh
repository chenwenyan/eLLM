docker create --name vllm --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /nfs/cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf:/root/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-hf vllm_build:v1.1
# docker create --name muxserve --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel 

docker create --name vllm --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/txu/.cache:/root/.cache -v /home/txu/workspace:/root/workspace -v /home/txu/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1

docker create --name vllm-spec --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /home/wychen/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1

docker create --name vllm_wychen --ipc=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /home/wychen/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_build:v1.1

docker create --name vllm_dynamic --gpus=all --ipc=host --net=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /home/wychen/.cache/huggingface/hub:/root/.cache/huggingface/hub -v /nfs/cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf:/root/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf vllm_lucz:latest

docker create --name vllm_hcache --gpus=all --ipc=host --net=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_lucz:latest

docker create --name vllm_org --gpus=all --ipc=host --net=host -it -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_lucz:latest

docker create --name vllm_dynamic --gpus=all --ipc=host --net=host -it -e CUDA_DEVICE_ORDER=PCI_BUS_ID  -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_wychen:latest 

docker create --name vllm_spec --gpus=all --ipc=host --net=host -it -e CUDA_DEVICE_ORDER=PCI_BUS_ID  -v /nfs/dataset:/nfs/dataset -v /nfs/cache:/nfs/cache -v /home/wychen/.cache:/root/.cache -v /home/wychen/workspace:/root/workspace -v /root/.cache/huggingface/hub:/root/.cache/huggingface/hub vllm_wychen:latest 