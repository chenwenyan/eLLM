# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
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
"""Inference-only DeepseekV2 model."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig
from vllm.core.ellm_cache import DEFAULT_LAYER_GROUP_SIZE
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import (fused_experts, fused_moe,
                                                  grouped_topk)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.worker.ellm_metadata import combine_partial_recompute_metadata


class DeepseekV2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}.")

        self.experts = nn.ModuleList([
            DeepseekV2MLP(hidden_size=config.hidden_size,
                          intermediate_size=config.moe_intermediate_size,
                          hidden_act=config.hidden_act,
                          quant_config=quant_config,
                          reduce_results=False)
            for idx in range(self.n_routed_experts)
        ])
        self.pack_params()

        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.n_routed_experts,
                                     bias=False,
                                     quant_config=None)

        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
            )

    def pack_params(self):
        w1 = []
        w2 = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)
            w2.append(expert.down_proj.weight)
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)

        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data

        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if self.config.n_shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        hidden_states = hidden_states.contiguous()
        num_expert_group = getattr(self.config, "n_group", 1)
        if num_expert_group > 1:
            topk_weights, topk_ids = grouped_topk(
                hidden_states,
                router_logits,
                self.top_k,
                renormalize=self.config.norm_topk_prob,
                num_expert_group=num_expert_group,
                topk_group=self.config.topk_group,
                scoring_func=getattr(self.config, "scoring_func", "softmax"),
                routed_scaling_factor=self.routed_scaling_factor)
            final_hidden_states = fused_experts(hidden_states,
                                                self.w1,
                                                self.w2,
                                                topk_weights,
                                                topk_ids,
                                                inplace=True)
        else:
            final_hidden_states = fused_moe(
                hidden_states,
                self.w1,
                self.w2,
                router_logits,
                self.top_k,
                renormalize=self.config.norm_topk_prob,
                inplace=True) * self.routed_scaling_factor
        if self.config.n_shared_experts is not None:
            final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)

    def forward_with_partial_recompute(
        self,
        recompute_hidden_states: torch.Tensor,
        decode_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch recompute and decode tokens in one expert dispatch.

        eLLM recomputes evicted prefix tokens while decoding a new token. For
        MoE layers, routing the two streams separately would turn the small
        decode/recompute matrices into two inefficient expert launches. The
        routing decision is token-local, so both streams can be concatenated,
        routed together, and split without changing model semantics.
        """
        if recompute_hidden_states.ndim != 2:
            raise ValueError("recompute_hidden_states must be a 2-D tensor")
        if decode_hidden_states.ndim != 2:
            raise ValueError("decode_hidden_states must be a 2-D tensor")
        if recompute_hidden_states.shape[1:] != decode_hidden_states.shape[1:]:
            raise ValueError("recompute and decode hidden sizes must match")
        if recompute_hidden_states.device != decode_hidden_states.device:
            raise ValueError("recompute and decode tensors must share a device")
        if recompute_hidden_states.dtype != decode_hidden_states.dtype:
            raise ValueError("recompute and decode tensors must share a dtype")

        num_recompute_tokens = recompute_hidden_states.shape[0]
        combined_hidden_states = torch.cat(
            (recompute_hidden_states, decode_hidden_states), dim=0)
        combined_output = self(combined_hidden_states)
        return (combined_output[:num_recompute_tokens],
                combined_output[num_recompute_tokens:])


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


class DeepseekV2Attention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        layer_idx=None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attn_head_dim = self._get_attn_head_dim(self.qk_head_dim)

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=quant_config)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config)
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config)

        self.kv_a_proj_with_mqa = ReplicatedLinear(self.hidden_size,
                                                   self.kv_lora_rank +
                                                   self.qk_rope_head_dim,
                                                   bias=False,
                                                   quant_config=quant_config)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config)
        # O projection.
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=quant_config)
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            rope_scaling['type'] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(self.num_local_heads,
                              self.attn_head_dim,
                              self.scaling,
                              num_kv_heads=self.num_local_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    @staticmethod
    def _get_attn_head_dim(head_dim: int) -> int:
        for supported_head_dim in (64, 80, 96, 112, 128, 256):
            if head_dim <= supported_head_dim:
                return supported_head_dim
        raise ValueError(f"Unsupported DeepSeek-V2 head size: {head_dim}.")

    def _pad_head_dim(self, tensor: torch.Tensor,
                      head_dim: int) -> torch.Tensor:
        if head_dim == self.attn_head_dim:
            return tensor
        return torch.nn.functional.pad(
            tensor, [0, self.attn_head_dim - head_dim], value=0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Union[AttentionMetadata, List[AttentionMetadata]],
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads,
                                         self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads,
                                                   self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a.contiguous())
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads,
                     self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank:]
        cos_sin = self.rotary_emb.cos_sin_cache.to(
            positions.device)[positions.view(-1)]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
        q_pe = q_pe * cos + _rotate_gptj(q_pe) * sin
        k_pe = k_pe * cos + _rotate_gptj(k_pe) * sin
        q[..., self.qk_nope_head_dim:] = q_pe
        k = torch.empty_like(q)
        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe
        q = self._pad_head_dim(q, self.qk_head_dim).view(
            -1, self.num_local_heads * self.attn_head_dim)
        k = self._pad_head_dim(k, self.qk_head_dim).view(
            -1, self.num_local_heads * self.attn_head_dim)
        v = self._pad_head_dim(v, self.v_head_dim).view(
            -1, self.num_local_heads * self.attn_head_dim)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        attn_output = attn_output.view(
            -1, self.num_local_heads,
            self.attn_head_dim)[..., :self.v_head_dim].reshape(
                -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_with_partial_recompute(
        self,
        recompute_positions: torch.Tensor,
        recompute_hidden_states: torch.Tensor,
        recompute_attn_metadata: AttentionMetadata,
        decode_positions: torch.Tensor,
        decode_hidden_states: torch.Tensor,
        decode_attn_metadata: AttentionMetadata,
        kv_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run recompute and decode through one MLA projection/attention.

        The backend writes the complete combined K/V tensor before launching
        either attention operation.  Consequently decode observes the prefix
        K/V restored by recompute while QKV and output projections are batched.
        """
        num_recompute_tokens = recompute_hidden_states.shape[0]
        combined_positions = torch.cat(
            (recompute_positions, decode_positions), dim=0)
        combined_hidden_states = torch.cat(
            (recompute_hidden_states, decode_hidden_states), dim=0)
        combined_metadata = combine_partial_recompute_metadata(
            recompute_attn_metadata, decode_attn_metadata)
        combined_output = self(
            positions=combined_positions,
            hidden_states=combined_hidden_states,
            kv_cache=kv_cache,
            attn_metadata=combined_metadata,
        )
        return (combined_output[:num_recompute_tokens],
                combined_output[num_recompute_tokens:])


class DeepseekV2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = DeepseekV2Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            layer_idx=layer_idx,
        )
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekV2MoE(config=config, quant_config=quant_config)
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
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
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_with_partial_recompute(
        self,
        recompute_positions: torch.Tensor,
        recompute_hidden_states: torch.Tensor,
        recompute_attn_metadata: Union[AttentionMetadata,
                                       List[AttentionMetadata]],
        recompute_residual: Optional[torch.Tensor],
        decode_positions: torch.Tensor,
        decode_hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        decode_attn_metadata: Union[AttentionMetadata, List[AttentionMetadata]],
        decode_residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance recompute and decode streams through one layer.

        One combined attention call writes both streams' K/V before computing
        prompt and paged-decode attention. The streams then share one MLP or
        MoE dispatch, realizing eLLM's horizontal batching for sparse FFNs.
        """
        if recompute_residual is None:
            recompute_residual = recompute_hidden_states
            recompute_hidden_states = self.input_layernorm(
                recompute_hidden_states)
        else:
            recompute_hidden_states, recompute_residual = (
                self.input_layernorm(recompute_hidden_states,
                                     recompute_residual))

        if decode_residual is None:
            decode_residual = decode_hidden_states
            decode_hidden_states = self.input_layernorm(decode_hidden_states)
        else:
            decode_hidden_states, decode_residual = self.input_layernorm(
                decode_hidden_states, decode_residual)

        recompute_hidden_states, decode_hidden_states = (
            self.self_attn.forward_with_partial_recompute(
                recompute_positions=recompute_positions,
                recompute_hidden_states=recompute_hidden_states,
                recompute_attn_metadata=recompute_attn_metadata,
                decode_positions=decode_positions,
                decode_hidden_states=decode_hidden_states,
                decode_attn_metadata=decode_attn_metadata,
                kv_cache=kv_cache,
            ))

        recompute_hidden_states, recompute_residual = (
            self.post_attention_layernorm(recompute_hidden_states,
                                          recompute_residual))
        decode_hidden_states, decode_residual = (
            self.post_attention_layernorm(decode_hidden_states,
                                          decode_residual))

        if isinstance(self.mlp, DeepseekV2MoE):
            recompute_hidden_states, decode_hidden_states = (
                self.mlp.forward_with_partial_recompute(
                    recompute_hidden_states, decode_hidden_states))
        else:
            num_recompute_tokens = recompute_hidden_states.shape[0]
            combined_hidden_states = torch.cat(
                (recompute_hidden_states, decode_hidden_states), dim=0)
            combined_hidden_states = self.mlp(combined_hidden_states)
            recompute_hidden_states = combined_hidden_states[
                :num_recompute_tokens]
            decode_hidden_states = combined_hidden_states[
                num_recompute_tokens:]

        return (recompute_hidden_states, recompute_residual,
                decode_hidden_states, decode_residual)


class DeepseekV2Model(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config,
                                   layer_idx,
                                   cache_config=cache_config,
                                   quant_config=quant_config)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: Union[AttentionMetadata, List[AttentionMetadata]],
        seq_data_list: Optional[List] = None,
        layer_group_events: Optional[Dict[int, torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        if (isinstance(attn_metadata, list)
                and len(attn_metadata) != len(self.layers)):
            raise ValueError("attention metadata must match model layers")
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            if (layer_group_events is not None
                    and i % DEFAULT_LAYER_GROUP_SIZE == 0):
                event = layer_group_events.get(i // DEFAULT_LAYER_GROUP_SIZE)
                if event is not None:
                    torch.cuda.current_stream().wait_event(event)
            layer = self.layers[i]
            layer_attn_metadata = (attn_metadata[i]
                                   if isinstance(attn_metadata, list) else
                                   attn_metadata)
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], layer_attn_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_with_partial_recompute(
        self,
        recompute_input_ids: torch.Tensor,
        recompute_positions: torch.Tensor,
        recompute_attn_metadata: Optional[Union[AttentionMetadata,
                                                List[AttentionMetadata]]],
        decode_input_ids: torch.Tensor,
        decode_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        decode_attn_metadata: Optional[Union[AttentionMetadata,
                                             List[AttentionMetadata]]],
        metadata_planner: Optional[object] = None,
        layer_group_events: Optional[Dict[int, torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        """Decode using recomputed prefix K/V and a cached suffix.

        The caller owns the token/layer map and supplies temporary cache slots
        for the evicted prefix through ``recompute_attn_metadata``. Decode
        block tables must address those temporary slots followed by retained
        suffix slots and the new token slot.
        """
        if recompute_input_ids.numel() == 0:
            raise ValueError("partial recompute requires at least one token")
        if len(kv_caches) != len(self.layers):
            raise ValueError("partial recompute requires one cache per layer")

        if metadata_planner is not None:
            recompute_metadata_by_layer = None
            decode_metadata_by_layer = None
        elif recompute_attn_metadata is None:
            raise ValueError("recompute metadata is required")
        elif isinstance(recompute_attn_metadata, list):
            if len(recompute_attn_metadata) != len(self.layers):
                raise ValueError("recompute metadata must match model layers")
            recompute_metadata_by_layer = recompute_attn_metadata
        else:
            recompute_metadata_by_layer = [recompute_attn_metadata
                                           ] * len(self.layers)

        if metadata_planner is not None:
            pass
        elif isinstance(decode_attn_metadata, list):
            if len(decode_attn_metadata) != len(self.layers):
                raise ValueError("decode metadata must match model layers")
            decode_metadata_by_layer = decode_attn_metadata
        elif decode_attn_metadata is not None:
            decode_metadata_by_layer = [decode_attn_metadata] * len(self.layers)
        else:
            raise ValueError("decode metadata is required")

        recompute_hidden_states = self.embed_tokens(recompute_input_ids)
        decode_hidden_states = self.embed_tokens(decode_input_ids)
        recompute_residual = None
        decode_residual = None
        metadata_context = None
        active_layer_group = None
        try:
            for layer_idx, (layer, kv_cache) in enumerate(
                    zip(self.layers, kv_caches)):
                if (layer_group_events is not None
                        and layer_idx % DEFAULT_LAYER_GROUP_SIZE == 0):
                    event = layer_group_events.get(
                        layer_idx // DEFAULT_LAYER_GROUP_SIZE)
                    if event is not None:
                        torch.cuda.current_stream().wait_event(event)
                if metadata_planner is not None:
                    layer_group = metadata_planner.cache_map.get_layer_group(
                        layer_idx)
                    if layer_group != active_layer_group:
                        if metadata_context is not None:
                            metadata_context.__exit__(None, None, None)
                            metadata_context = None
                        new_metadata_context = (
                            metadata_planner.metadata_for_layer_group(
                                layer_group))
                        layer_recompute_metadata, layer_decode_metadata = (
                            new_metadata_context.__enter__())
                        metadata_context = new_metadata_context
                        active_layer_group = layer_group
                else:
                    layer_recompute_metadata = (
                        recompute_metadata_by_layer[layer_idx])
                    layer_decode_metadata = decode_metadata_by_layer[layer_idx]

                (recompute_hidden_states, recompute_residual,
                 decode_hidden_states,
                 decode_residual) = layer.forward_with_partial_recompute(
                     recompute_positions,
                     recompute_hidden_states,
                     layer_recompute_metadata,
                     recompute_residual,
                     decode_positions,
                     decode_hidden_states,
                     kv_cache,
                     layer_decode_metadata,
                     decode_residual,
                 )
        finally:
            if metadata_context is not None:
                metadata_context.__exit__(None, None, None)

        decode_hidden_states, _ = self.norm(decode_hidden_states,
                                            decode_residual)
        return decode_hidden_states


class DeepseekV2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = DeepseekV2Model(config, cache_config, quant_config)
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: Union[AttentionMetadata, List[AttentionMetadata]],
        seq_data_list: Optional[List] = None,
        layer_group_events: Optional[Dict[int, torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, seq_data_list,
                                   layer_group_events)
        return hidden_states

    def forward_with_partial_recompute(
        self,
        recompute_input_ids: torch.Tensor,
        recompute_positions: torch.Tensor,
        recompute_attn_metadata: Optional[Union[AttentionMetadata,
                                                List[AttentionMetadata]]],
        decode_input_ids: torch.Tensor,
        decode_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        decode_attn_metadata: Optional[Union[AttentionMetadata,
                                             List[AttentionMetadata]]],
        metadata_planner: Optional[object] = None,
        layer_group_events: Optional[Dict[int, torch.cuda.Event]] = None,
    ) -> torch.Tensor:
        return self.model.forward_with_partial_recompute(
            recompute_input_ids,
            recompute_positions,
            recompute_attn_metadata,
            decode_input_ids,
            decode_positions,
            kv_caches,
            decode_attn_metadata,
            metadata_planner,
            layer_group_events,
        )

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (("mlp.experts." in name or "mlp.shared_experts." in name)
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if (("mlp.experts." in name or "mlp.shared_experts." in name)
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
