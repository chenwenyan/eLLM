#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "dispatch_utils.h"
#include "reduction_utils.cuh"

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#else
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
#endif

namespace vllm {

  template <typename scalar_t>
  __global__ void hfused_mlp_kernel(
      scalar_t* __restrict__ last_out,      // [..., output_size]
      scalar_t* __restrict__ out,           // [..., output_size]
      scalar_t* __restrict__ last_input,     // [..., last_input_size]
      const scalar_t* __restrict__ input,    // [..., input_size]
      const scalar_t* __restrict__ weight_hidden,   // [hidden_size, input_size]
      const scalar_t* __restrict__ bias_hidden,     // [hidden_size]
      const int num_tokens, const int input_size, const int hidden_size, const int output_size) {
    // Check if the hidden size is larger than the maximum shared memory per block
    if (hidden_size > 1024) {
      // Handle error or reduce hidden_size
      return;
    }
  
    // Check if the output size is larger than the maximum shared memory per block
    if (output_size > 1024) {
      // Handle error or reduce output_size
      return;
    }
  
    __shared__ float s_hidden[1024];
    __shared__ float s_output[1024];
  
    int token_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;
  
    for (int idx = hidden_idx; idx < hidden_size; idx += blockDim.x) {
      float hidden = 0.0f;
      for (int j = 0; j < input_size; ++j) {
        hidden += input[token_idx * input_size + j] * weight_hidden[idx * input_size + j];
      }
      hidden += bias_hidden[idx];
      s_hidden[idx] = hidden;
    }
    __syncthreads();
  
    for (int idx = hidden_idx; idx < output_size; idx += blockDim.x) {
      float output = 0.0f;
      for (int j = 0; j < hidden_size; ++j) {
        output += s_hidden[j];
      }
      output += bias_hidden[idx];
      s_output[idx] = output;
    }
    __syncthreads();
  
    for (int idx = hidden_idx; idx < output_size; idx += blockDim.x) {
      out[token_idx * output_size + idx] = (scalar_t)s_output[idx];
    }
  }
}  // namespace vllm


void hfused_mlp(torch::Tensor& last_out, torch::Tensor& out, torch::Tensor& last_input, torch::Tensor& input,
                torch:: Tensor& weight, torch::Tensor& bias) {
  int input_size = input.size(-1);
  int hidden_size = input.size(-1);
  int output_size = input.size(-1);
  int num_tokens = input.numel() / input_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hfused_mlp_kernel", [&] {
    vllm::hfused_mlp_kernel<scalar_t><<<grid, block, 0, stream>>>(
      last_out.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
        last_input.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
        num_tokens, input_size, hidden_size, output_size);
  });
}

