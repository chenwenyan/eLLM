#include "cpu_types.hpp"

namespace vllm{
template <typename scalar_t>
void hfused_mlp_impl(scalar_t* __restrict__ last_input,
                      const scalar_t* __restrict__ input,
                      const scalar_t* __restrict__ weight,
                      const scalar_t* __restrict__ bias,
                      const int num_tokens, const int input_size,
                      const int hidden_size, const int output_size) {
  using scalar_vec_t = vec_op::vec_t<scalar_t>;
  constexpr int VEC_ELEM_NUM = scalar_vec_t::get_elem_num();
  TORCH_CHECK(input_size % VEC_ELEM_NUM == 0);
  TORCH_CHECK(hidden_size % VEC_ELEM_NUM == 0);
  TORCH_CHECK(output_size % VEC_ELEM_NUM == 0);

#pragma omp parallel for
  for (int i = 0; i < num_tokens; ++i) {
    auto input_p = input + i * input_size;
    auto output_p = out + i * output_size;

    // First linear layer
    for (int j = 0; j < hidden_size; j += VEC_ELEM_NUM) {
      scalar_vec_t hidden(0.0);
      for (int k = 0; k < input_size; k += VEC_ELEM_NUM) {
        scalar_vec_t x(input_p + k);
        scalar_vec_t w(j * input_size + k);
        hidden = hidden + x * w;
      }
      scalar_vec_t b(bias + j);
      hidden = hidden + b;
      hidden = vec_op::relu(hidden);  // Apply ReLU activation
      hidden.save(output_p + j);
    }

    // Second linear layer
    for (int j = 0; j < output_size; j += VEC_ELEM_NUM) {
      scalar_vec_t output(0.0);
      for (int k = 0; k < hidden_size; k += VEC_ELEM_NUM) {
        scalar_vec_t h(output_p + k);
        scalar_vec_t w(j * hidden_size + k);
        output = output + h * w;
      }
      scalar_vec_t b(bias + j);
      output = output + b;
      output.save(output_p + j);
    }
  }
}
}  // namespace

void hfused_mlp(torch::Tensor& last_out,torch::Tensor& out,  torch::Tensor& last_input, torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias) {
  int input_size = input.size(-1);
  int hidden_size = input.size(-1);
  int output_size = input.size(-1);
  int num_tokens = input.numel() / input_size;

  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hfused_mlp_impl", [&] {
    CPU_KERNEL_GUARD_IN(hfused_mlp_impl)
    hfused_mlp_impl(last_input.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                     num_tokens, input_size, hidden_size, output_size);
    CPU_KERNEL_GUARD_OUT(hfused_mlp_impl)
  });
}