from typing import Optional, Tuple, Type

import torch
import inspect
from vllm._C import ops as vllm_ops

# 获取 vllm_ops 中的所有成员
members = inspect.getmembers(vllm_ops)

# 过滤出函数
functions = [member for member in members if inspect.isfunction(member[1])]

# 输出函数名称
for name, func in functions:
    print(name)

# try:
from vllm._C import cache_ops as vllm_cache_ops
from vllm._C import ops as vllm_ops
# except ImportError:
#     pass


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.silu_and_mul(out, x)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_and_mul(out, x)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_tanh_and_mul(out, x)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_fast(out, x)


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    vllm_ops.gelu_new(out, x)


# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v1(out, query, key_cache, value_cache,
                                num_kv_heads, scale, block_tables, seq_lens,
                                block_size, max_seq_len, alibi_slopes,
                                kv_cache_dtype, kv_scale)

def fused_paged_attention_v1(
    last_out: torch.Tensor,
    last_query: torch.Tensor,
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.fused_paged_attention_v1(last_out, last_query, out, query, key_cache, value_cache,
                                num_kv_heads, scale, block_tables, seq_lens,
                                block_size, max_seq_len, alibi_slopes,
                                kv_cache_dtype, kv_scale)

def paged_attention_v1_with_dynamic_kv(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    
    from torch import nn    
    num_seqs, num_heads, head_size = query.shape
    print(f'num_seqs: {num_seqs}, num_heads: {num_heads}, head_size: {head_size}, num_kv_heads: {num_kv_heads}')
    # num_seqs: 96, num_heads: 32, head_size: 80, num_kv_heads: 32

    # The first method
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (head_size ** 0.5)
    mask = seq_lens.view(-1, 1, 1).expand_as(attn_scores) < max_seq_len
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    # # print(f'attn_scores.shape is: {attn_scores.shape}')
    # # attn_scores.shape is: torch.Size([96, 32, 32])
    # print(f'seq_lens.shape is: {seq_lens.shape}, max_seq_len is: {max_seq_len}')
    # # seq_lens.shape is: torch.Size([120]), max_seq_len is: 2048
    # attn_scores = torch.bmm(query, key.transpose(-2, -1)) / (head_size ** 0.5)
    # causal_mask =  torch.triu(seq_lens.view(-1, 1, 1).expand_as(attn_scores), diagonal=1).bool()
    # attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    

    # print(f'alibi_slopes is: {alibi_slopes}')
    # alibi_slopes 一种用于注意力分数偏置的方法,形状为 (nheads,) 或 (batch_size, nheads),数据类型为 fp32。例如,如果 nheads 为 4,则 alibi_slopes 可以是形状为 (4,) 的张量
    if alibi_slopes is not None:
        attn_scores += alibi_slopes.unsqueeze(1).unsqueeze(1)
    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    out.copy_(output.view(num_seqs, -1, head_size))

    # print(f'output: {output.shape}, out: {out.shape}')
    # output = output.view(num_seqs, num_heads, head_size)
    # out.copy_(output)

    # out.to('cpu')
    # print(f'out: {out.shape}')
    # print(f'out: {out[0, :2, :3]}')
    # out.to(query.device)

    # The second method
    # src_len = key.shape[1]
    # attn_scores = torch.bmm(query, key.transpose(1, 2))
    # attention_mask = seq_lens.view(-1, 1, 1).expand_as(attn_scores) < max_seq_len
    # attn_weights = attn_scores / (head_size ** 0.5)
    # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
    # attn_weights = attn_weights.view(num_seqs * num_heads, max_seq_len, src_len)
    # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    # attn_weights = attn_weights.view(num_seqs, num_heads, max_seq_len, src_len)
    # attn_output = torch.bmm(attn_weights, value)
    # # attn_weights = attn_weights.masked_fill(attention_mask, 0.0)
    # attn_output = attn_output.view(num_seqs, num_heads, max_seq_len, head_size)
    # attn_output = attn_output.transpose(1, 2)
    # hidden_size = num_heads * head_size
    # attn_output = attn_output.reshape(num_seqs, max_seq_len, hidden_size)
    # out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    # out = out_proj(attn_output)
    # out.copy_(out.view(num_seqs, -1, head_size))


def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query,
                                key_cache, value_cache, num_kv_heads, scale,
                                block_tables, seq_lens, block_size,
                                max_seq_len, alibi_slopes, kv_cache_dtype,
                                kv_scale)


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    vllm_ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache,
                              is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    vllm_ops.batched_rotary_embedding(positions, query, key, head_size,
                                      cos_sin_cache, is_neox, rot_dim,
                                      cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    vllm_ops.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    vllm_ops.fused_add_rms_norm(input, residual, weight, epsilon)

def hfused_add_rms_norm(last_input: torch.Tensor, input: torch.Tensor,
                        last_residual: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    vllm_ops.hfused_add_rms_norm(last_input, input, last_residual, residual, weight, epsilon)

def hfused_mlp(last_out: torch.Tensor, out: torch.Tensor, last_input: torch.Tensor, input: torch.Tensor, 
               weight: torch.Tensor, bias: torch.Tensor) -> None:
    vllm_ops.hfused_mlp(last_out, out, last_input, input, weight, bias)    

# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    return vllm_ops.awq_dequantize(qweight, scales, zeros, split_k_iters, thx,
                                   thy)


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
             scales: torch.Tensor, split_k_iters: int) -> torch.Tensor:
    return vllm_ops.awq_gemm(input, qweight, qzeros, scales, split_k_iters)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    return vllm_ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                              b_g_idx, use_exllama, bit)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    vllm_ops.gptq_shuffle(q_weight, q_perm, bit)


# squeezellm
def squeezellm_gemm(vec: torch.Tensor, mat: torch.Tensor, mul: torch.Tensor,
                    lookup_table: torch.Tensor) -> None:
    vllm_ops.squeezellm_gemm(vec, mat, mul, lookup_table)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    return vllm_ops.marlin_gemm(a, b_q_weight, b_scales, workspace, size_m,
                                size_n, size_k)


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, num_bits: int, size_m: int,
                        size_n: int, size_k: int) -> torch.Tensor:
    return vllm_ops.gptq_marlin_24_gemm(a, b_q_weight, b_meta, b_scales,
                                        workspace, num_bits, size_m, size_n,
                                        size_k)


# cutlass
def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
                         a_scales: torch.Tensor, b_scales: torch.Tensor,
                         out_dtype: Type[torch.dtype]) -> torch.Tensor:
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    vllm_ops.cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales)

    return out


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: torch.Tensor,
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    return vllm_ops.aqlm_gemm(input, codes, codebooks, scales,
                              codebook_partition_sizes, bias)


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: torch.Tensor) -> torch.Tensor:
    return vllm_ops.aqlm_dequant(codes, codebooks, codebook_partition_sizes)


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    return vllm_ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n,
                                       num_bits)


def gptq_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor, g_idx: torch.Tensor,
                     perm: torch.Tensor, workspace: torch.Tensor,
                     num_bits: int, size_m: int, size_n: int, size_k: int,
                     is_k_full: bool) -> torch.Tensor:
    return vllm_ops.gptq_marlin_gemm(a, b_q_weight, b_scales, g_idx, perm,
                                     workspace, num_bits, size_m, size_n,
                                     size_k, is_k_full)


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.zeros(1, device=input.device, dtype=torch.float32)
        vllm_ops.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        vllm_ops.static_scaled_fp8_quant(output, input, scale)
    return output, scale


# moe
def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    vllm_ops.moe_align_block_size(topk_ids, num_experts, block_size,
                                  sorted_token_ids, experts_ids,
                                  num_tokens_post_pad)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                     slot_mapping, kv_cache_dtype, kv_scale)


# fused reshape_and_cache
def fused_reshape_and_cache(
    last_key: torch.Tensor,
    last_value: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    kv_scale: float,
) -> None:
    vllm_cache_ops.fused_reshape_and_cache(last_key, last_value, key, value, key_cache, value_cache,
                                     slot_mapping, kv_cache_dtype, kv_scale)

def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
) -> None:
    vllm_cache_ops.reshape_and_cache_flash(key, value, key_cache, value_cache,
                                           slot_mapping, kv_cache_dtype)


def copy_blocks(key_caches: torch.Tensor, value_caches: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    vllm_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    vllm_cache_ops.swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    vllm_cache_ops.convert_fp8(output, input, scale, kv_dtype)


#TODO: cuda_utils, custom_ar
