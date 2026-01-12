# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata, HFusedAttention
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm, HFusedRMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear,
                                               ColumnParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_hip, print_warning_once
from vllm.logger import init_logger
from vllm import _custom_ops as ops
from contextlib import contextmanager
import time

import itertools
logger = init_logger(__name__)

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # with timer("gate_up_proj"):
        gate_up, _ = self.gate_up_proj(x)

        # with timer("act_fn"):    
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        
        # with timer("merged down_proj calls"):
        #     # 假设在批处理维度（dim=0）拼接；根据实际情况调整dim参数
        #     combined_input = torch.cat([x, x], dim=0)
        #     combined_output, _ = self.down_proj(combined_input)
        #     # 拆分结果
        #     split_size = x.size(0)
        #     last_x = combined_output[:split_size]
        #     x = combined_output[split_size:]    
        return x


@contextmanager
def timer(stage:str):
    st = torch.cuda.Event(enable_timing=True)
    et = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    st.record()
    try:
        yield
    finally:
        et.record()
        torch.cuda.synchronize()
        logger.info(f"{stage} used {st.elapsed_time(et)} ms")

    

class HFusedLlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=bias,
                                           quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, last_x, x):
        # fused kernel function
        out = torch.empty_like(x, device=x.device)
        last_out = torch.empty_like(last_x, device=x.device)
        # Load weights and bias from the model
        weight = torch.empty(x.shape[-1], x.shape[-1], dtype=torch.float16, device=x.device)
        bias = torch.empty(x.shape[-1], dtype=torch.float16, device=x.device)

        # logger.info("before mlp", out.shape, last_out.shape, x.shape, last_x.shape)
        # with timer("before fused mlp"):
        ops.hfused_mlp(last_out, out, last_x, x, weight, bias)
        # logger.info("after mlp",out.shape, last_out.shape, x.shape, last_x.shape)
        # torch.cuda.synchronize()
        return last_out, out
    
class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        sliding_window: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
        )

        self.o_proj_column = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
        )
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window,
                              cache_config=cache_config,
                              quant_config=quant_config)  
        self.hfused_attn = HFusedAttention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              sliding_window=sliding_window,
                              cache_config=cache_config,
                              quant_config=quant_config)  
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        fused: Optional[bool] = False,
        last_positions: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        last_kv_cache: Optional[torch.Tensor] = None,
        last_attn_metadata: Optional[AttentionMetadata] = None,
    ):
        if not fused:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
            output, _ = self.o_proj(attn_output)
            return output
        else:
            # with timer("before attention"):
            last_qkv, _ = self.qkv_proj(last_hidden_states)
            last_q, last_k, last_v = last_qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            last_q, last_k = self.rotary_emb(last_positions, last_q, last_k)
            # logger.info(f'last_q: {last_q.shape}, last_k: {last_k.shape}, last_v: {last_v.shape}')
    
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            
                # logger.info(f'q: {q.shape}, k: {k.shape}, v: {v.shape}')

                # attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
                # fused attention 
            # with timer("fused_attention"):
            last_attn_output, attn_output = self.hfused_attn(last_q, last_k, last_v, last_kv_cache, last_attn_metadata, q, k, v, kv_cache, attn_metadata)
            # logger.info(f'last_attn_output: {last_attn_output.shape}, attn_output: {attn_output.shape}')
            
            # org
            # with timer("org o_proj"):
            #     last_output, _ = self.o_proj(last_attn_output)
            #     output, _ = self.o_proj(attn_output)

                # last_output = self.o_proj_column(last_attn_output)
                # output = self.o_proj_column(attn_output)

            # with timer("merged o_proj calls"):
            # 假设在批处理维度（dim=0）拼接；根据实际情况调整dim参数
            combined_input = torch.cat([last_attn_output, attn_output], dim=0)
            combined_output, _ = self.o_proj(combined_input)
            # 拆分结果
            split_size = last_attn_output.size(0)
            last_output = combined_output[:split_size]
            output = combined_output[split_size:]

            # torch.cuda.synchronize()        
            return last_output, output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        sliding_window = getattr(config, "sliding_window", None)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            sliding_window=sliding_window,
            cache_config=cache_config,
        )

        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        # self.hfused_mlp = HFusedLlamaMLP(
        #     hidden_size=self.hidden_size,
        #     intermediate_size=config.intermediate_size,
        #     hidden_act=config.hidden_act,
        #     quant_config=quant_config,
        #     bias=getattr(config, "mlp_bias", False),
        # )
        self.hfused_post_attention_layernorm = HFusedRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        fused: Optional[bool] = False,
        last_positions: Optional[torch.Tensor] = None,
        last_hidden_states: Optional[torch.Tensor] = None,
        last_kv_cache: Optional[torch.Tensor] = None,
        last_attn_metadata: Optional[AttentionMetadata] = None,
        last_residual: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        if not fused:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )

            # Fully Connected
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
            # with timer("org mlp"):
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        
        else:
            if last_residual is None:
                last_residual = last_hidden_states
                last_hidden_states = self.input_layernorm(last_hidden_states)
            if last_residual is not None:
                last_hidden_states, last_residual = self.input_layernorm(
                    last_hidden_states, last_residual)      
        
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            if residual is not None:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual)      
            last_hidden_states, hidden_states = self.self_attn(
                    positions=positions,
                    hidden_states=hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    fused=True,
                    last_positions=last_positions,
                    last_hidden_states=last_hidden_states,
                    last_kv_cache=last_kv_cache,
                    last_attn_metadata=last_attn_metadata,
            )
            # last_hidden_states, last_residual, hidden_states, residual = self.hfused_post_attention_layernorm(
            #     last_hidden_states, hidden_states, last_residual, residual)
            
            # last_hidden_states, hidden_states = self.hfused_mlp(
            #     last_hidden_states, hidden_states)
            
            # last_hidden_states, hidden_states = self.self_attn(
            #     positions=positions,
            #     hidden_states=hidden_states,
            #     kv_cache=kv_cache,
            #     attn_metadata=attn_metadata,
            #     fused=True,
            #     last_positions=last_positions,
            #     last_hidden_states=last_hidden_states,
            #     last_kv_cache=last_kv_cache,
            #     last_attn_metadata=last_attn_metadata,
            # )

            # last_hidden_states, last_residual, hidden_states, residual = self.hfused_post_attention_layernorm(
            #     last_hidden_states, hidden_states, last_residual, residual)
            
            # last_hidden_states, hidden_states = self.hfused_mlp(
            #     last_hidden_states, hidden_states)

            return last_hidden_states, last_residual, hidden_states, residual

class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, cache_config, quant_config)
            for _ in range(config.num_hidden_layers)
        ])

        # logger.info(f'config.num_hidden_layers: {config.num_hidden_layers}')

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cache_config = cache_config

        # save last_hidden_states for token 0~9
        self.last_hidden_states = None
        self.last_residual = None
        self.last_attn_metadata = None
        self.last_positions = None

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_computation_cost(self, request_lens, recomputed_layers):
        self.FLOPS = 624 * 0.55 * 1000000000000 * get_tensor_model_parallel_world_size() * 1.0
        self.num_layers=self.layers
        self.hidden_size=self.config.hidden_size
        self.epsilon=20
        computation_cost = 0
        for _ in range(len(recomputed_layers)):
            computation_cost += (1 / self.FLOPS) * (
                24 * request_lens * self.hidden_size +
                4 * request_lens ** 2 +
                2 * request_lens * self.hidden_size * self.vocab_size +
                self.epsilon
            )
        return computation_cost

    def get_communication_cost(self, swapped_layers, attn_metadata):
        tokens = 0
        for i in range(len(swapped_layers)):
            tokens += self.num_prefill_tokens + attn_metadata.decode_metadata.num_decode_tokens
        communication_cost = (0.001085 * tokens + 0.1103)/1000 
        return communication_cost 

    def get_best_layer(self, request_lens, attn_metadata):
        # st = torch.cuda.Event(enable_timing=True)
        # et = torch.cuda.Event(enable_timing=True)
        # st.record()
        # 所有可能的 recomputed layers 和 swapped layers 的组合
        layers = [1, 2]
        best_cost = float('inf')
        best_recomputed_layers = []
        best_swapped_layers = []

        # 遍历所有可能的 recomputed layers 和 swapped layers 的组合
        for layer in range(len(layers)):
            recomputed_layers = [layer]
            # swapped layers 是剩余的层
            swapped_layers = [layer for layer in layers if layer not in recomputed_layers]

            # 计算总成本
            computation_cost = self.get_computation_cost(request_lens, recomputed_layers)
            communication_cost = self.get_communication_cost(swapped_layers, attn_metadata)
            cost_gap = abs(computation_cost - communication_cost)

            # 更新最佳组合
            if cost_gap < best_cost:
                best_cost = cost_gap
                best_recomputed_layers = recomputed_layers
                best_swapped_layers = swapped_layers

        # logger.info(f"Best recomputed layers: {best_recomputed_layers}")
        # logger.info(f"Best swapped layers: {best_swapped_layers}")
        # logger.info(f"Best cost gap: {best_cost}")
        # et.record()
        # torch.cuda.synchronize()
        # logger.info(f'get_best_layer time: {st.elapsed_time(et)} ms')
        return best_recomputed_layers, best_swapped_layers

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        seq_data_list: Optional[List] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # if inputs_embeds is not None:
        #     hidden_states = inputs_embeds
        # else:
        #     hidden_states = self.get_input_embeddings(input_ids)
        # residual = None

        # store_cache_layer_id=int(len(self.layers)*self.cache_config.store_cache_layers)-1
        # stream1 = torch.cuda.Stream()
        # stream2 = torch.cuda.Stream()
        # # 创建一个同步事件（用于跨流协调）
        # load_complete_event = torch.cuda.Event()
        # # logger.info(f'kv_caches_length: {len(kv_caches)}')
        # # 判断kv caches是否为空

        # if attn_metadata.prefill_metadata is not None:
        #     self.last_attn_metadata = attn_metadata
        #     self.last_positions = positions
        #     self.num_prefill_tokens = attn_metadata.prefill_metadata.num_prefill_tokens

        #     for i in range(int(len(self.layers)*self.cache_config.store_cache_layers)-1):
        #         layer = self.layers[i]
        #         hidden_states, residual = layer(
        #             positions,
        #             hidden_states,
        #             kv_caches[i],
        #             attn_metadata,
        #             residual,
        #         )
        #     self.last_hidden_states = hidden_states
        #     self.last_residual = residual

        #     if self.cache_config.store_cache_layers < 1:
        #         for i in range(int(len(self.layers)*self.cache_config.store_cache_layers)-1, len(self.layers)):
        #             # recomputing kv_caches for the layers that are not stored
        #             kv_cache = None
        #             hidden_states, residual = self.layers[i](
        #                 positions,
        #                 hidden_states,
        #                 kv_cache,
        #                 attn_metadata,
        #                 residual,
        #             )
        # else: 
        #     layer_indexs = [seq_data['layer_index'] for seq_data in seq_data_list]
        #     # 第一次prefilling时不需要并行，decoding阶段需要并行
        #     i = 0
        #     while i < len(self.layers) - 2:
        #         if i <= int(len(self.layers)*self.cache_config.store_cache_layers)-4:
        #             hidden_states, residual = self.layers[i](
        #                 positions,
        #                 hidden_states,
        #                 kv_caches[i],
        #                 attn_metadata,
        #                 residual,
        #             )  
        #         else:
        #             # logger.info(f'fused layer: {i}')
        #             if i in layer_indexs:
        #                 input_ids = seq_data_list[layer_indexs.index(i)]['input_ids']
        #                 if inputs_embeds is None:
        #                     hidden_states = self.get_input_embeddings(input_ids)
        #                 positions = seq_data_list[layer_indexs.index(i)]['positions']
        #                 attn_metadata = seq_data_list[layer_indexs.index(i)]['attn_metadata']

        #             total_seq_len = sum([seq_data['attn_metadata'].total_seq_len for seq_data in seq_data_list])
        #             swapped_layers, recomputed_layers = self.get_best_layer(total_seq_len, attn_metadata)
        #             with torch.cuda.stream(stream1):
        #                 # load kv caches from cpu to gpu
        #                 logger.info(f'load kv caches from cpu to gpu, layer: {i}')
        #                 sleep_time = self.get_communication_cost(swapped_layers, attn_metadata)
        #                 logger.info(f'sleep_time: {sleep_time}')
        #                 time.sleep(sleep_time)
        #                 # 记录事件：stream1完成加载
        #                 load_complete_event.record(stream=stream1)
                        
        #             i += len(swapped_layers)
        #             kv_cache = kv_caches[store_cache_layer_id]
        #             with torch.cuda.stream(stream2):    
        #                 load_complete_event.wait(stream=stream2)
        #                 last_hidden_states, last_residual, hidden_states, residual = self.layers[i](
        #                     positions,
        #                     hidden_states,
        #                     None,
        #                     attn_metadata,
        #                     residual,
        #                     True,
        #                     self.last_positions,
        #                     self.last_hidden_states,
        #                     kv_cache,
        #                     self.last_attn_metadata,
        #                     self.last_residual,
        #                 )

        #             self.last_hidden_states = last_hidden_states
        #             self.last_residual = last_residual
        #             # logger.info(f'after added, fused layer: {i}')
        #             i += 2 # skip two layers
        #             continue  # 跳过当前迭代，进入下一次迭代
        #         i += 1
            
        #     torch.cuda.synchronize()
        #     # The last layer is not fused
        #     # logger.info(f'torch cuda time: {st.elapsed_time(et)} ms, {input_ids.device}')
        #     hidden_states, residual = self.layers[-2](
        #         positions,
        #         hidden_states,
        #         None,
        #         attn_metadata,
        #         residual,
        #     )

        #     hidden_states, residual = self.layers[-1](
        #         positions,
        #         hidden_states,
        #         None,
        #         attn_metadata,
        #         residual,
        #     )
                    
        # hidden_states, _ = self.norm(hidden_states, residual)
        # return hidden_states


        # TODO: 优化代码
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        import os
        torch.cuda.device_count = lambda: len(list(os.environ["CUDA_VISIBLE_DEVICES"].split(",") if "CUDA_VISIBLE_DEVICES" in os.environ else []))
        print(f'kv_caches_length: {len(kv_caches)}, len(self.layers): {len(self.layers)}')
        for i in range(int(len(self.layers)*self.cache_config.store_cache_layers)):
        # for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        # for i in range(int(len(self.layers)*self.cache_config.store_cache_layers), len(self.layers)):
        for i in range(int(len(self.layers)*self.cache_config.store_cache_layers), 10):
            # recomputing kv_caches for the layers that are not stored
            kv_cache = None
            hidden_states, residual = self.layers[i](
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )     
  
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        seq_data_list: Optional[List] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, seq_data_list)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(
                        ".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        print_warning_once(
                            f"Found kv scale in the checkpoint (e.g. {name}), "
                            "but not found the expected name in the model "
                            f"(e.g. {remapped_kv_scale_name}). kv-scale is "
                            "not loaded.")
                        continue
                    else:
                        name = remapped_kv_scale_name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            layer_self_attn = self.model.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")