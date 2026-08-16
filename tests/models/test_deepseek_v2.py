from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from vllm.attention.backends.xformers import XFormersMetadata
from vllm.config import CacheConfig, ModelConfig
from vllm.core.ellm_cache import TokenLayerKVMap
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.deepseek_v2 import (DeepseekV2ForCausalLM,
                                                    DeepseekV2MoE)
from vllm.utils import get_open_port
from vllm.worker.ellm_metadata import (
    LayerGroupMetadataPlanner, build_layerwise_partial_recompute_metadata)


def _make_config() -> PretrainedConfig:
    config = PretrainedConfig()
    values = {
        "pad_token_id": 0,
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "q_lora_rank": 16,
        "kv_lora_rank": 8,
        "rope_theta": 10000,
        "max_position_embeddings": 128,
        "rope_scaling": None,
        "rms_norm_eps": 1e-6,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "scoring_func": "softmax",
        "routed_scaling_factor": 1.0,
        "norm_topk_prob": True,
        "n_shared_experts": 1,
        "moe_intermediate_size": 16,
        "intermediate_size": 32,
        "hidden_act": "silu",
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
    }
    for name, value in values.items():
        setattr(config, name, value)
    return config


def _make_attention_metadata(num_tokens: int) -> XFormersMetadata:
    device = torch.device("cuda")
    return XFormersMetadata(
        num_prefills=1,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        total_seq_len=num_tokens,
        slot_mapping=torch.arange(num_tokens,
                                  dtype=torch.long,
                                  device=device),
        seq_lens=[num_tokens],
        seq_lens_tensor=torch.tensor([num_tokens],
                                     dtype=torch.int32,
                                     device=device),
        max_query_len=num_tokens,
        max_prefill_seq_len=num_tokens,
        max_decode_seq_len=0,
        query_start_loc=torch.tensor([0, num_tokens],
                                     dtype=torch.int32,
                                     device=device),
        seq_start_loc=torch.tensor([0, num_tokens],
                                   dtype=torch.int32,
                                   device=device),
        context_lens_tensor=torch.zeros(1,
                                        dtype=torch.int32,
                                        device=device),
        block_tables=torch.empty((1, 0),
                                 dtype=torch.int32,
                                 device=device),
        use_cuda_graph=False,
    )


def _make_decode_attention_metadata(context_len: int) -> XFormersMetadata:
    device = torch.device("cuda")
    seq_len = context_len + 1
    num_blocks = (seq_len + 15) // 16
    return XFormersMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        total_seq_len=seq_len,
        slot_mapping=torch.tensor([context_len],
                                  dtype=torch.long,
                                  device=device),
        seq_lens=None,
        seq_lens_tensor=torch.tensor([seq_len],
                                     dtype=torch.int32,
                                     device=device),
        max_query_len=None,
        max_prefill_seq_len=0,
        max_decode_seq_len=seq_len,
        query_start_loc=None,
        seq_start_loc=None,
        context_lens_tensor=None,
        block_tables=torch.arange(num_blocks,
                                  dtype=torch.int32,
                                  device=device).unsqueeze(0),
        use_cuda_graph=False,
    )


def _make_batched_decode_attention_metadata(
        seq_lens, block_tables) -> XFormersMetadata:
    device = torch.device("cuda")
    slots = []
    for seq_len, table in zip(seq_lens, block_tables):
        position = seq_len - 1
        slots.append(table[position // 16] * 16 + position % 16)
    return XFormersMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=len(seq_lens),
        total_seq_len=sum(seq_lens),
        slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        seq_lens=None,
        seq_lens_tensor=torch.tensor(seq_lens,
                                     dtype=torch.int32,
                                     device=device),
        max_query_len=None,
        max_prefill_seq_len=0,
        max_decode_seq_len=max(seq_lens),
        query_start_loc=None,
        seq_start_loc=None,
        context_lens_tensor=None,
        block_tables=torch.tensor(block_tables,
                                  dtype=torch.int32,
                                  device=device),
        use_cuda_graph=False,
    )


def test_deepseek_v2_uses_dedicated_model_implementation():
    model_cls = ModelRegistry.load_model_cls("DeepseekV2ForCausalLM")

    assert model_cls is DeepseekV2ForCausalLM
    assert model_cls.__module__.endswith(".deepseek_v2")


def test_deepseek_v2_cache_head_size_matches_padded_attention():
    config = _make_config()
    config.model_type = "deepseek_v2"
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    model_config = object.__new__(ModelConfig)
    model_config.hf_text_config = config

    assert model_config.get_head_size() == 256


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v2_forward_and_weight_loading():
    init_method = f"tcp://localhost:{get_open_port()}"
    init_distributed_environment(world_size=1,
                                 rank=0,
                                 distributed_init_method=init_method,
                                 local_rank=0)
    ensure_model_parallel_initialized(1, 1)

    config = _make_config()
    cache_config = CacheConfig(block_size=16,
                               gpu_memory_utilization=0.9,
                               swap_space=1,
                               cache_dtype="auto",
                               store_cache_layers=1.0,
                               flatten_layers=config.num_hidden_layers)
    with set_default_torch_dtype(torch.float16), torch.device("cuda"):
        model = DeepseekV2ForCausalLM(config, cache_config=cache_config)

    assert isinstance(model.model.layers[1].mlp, DeepseekV2MoE)
    attention = model.model.layers[0].self_attn
    assert hasattr(attention, "kv_a_proj_with_mqa")
    assert hasattr(attention, "kv_b_proj")

    gate_weight = torch.full((config.moe_intermediate_size,
                              config.hidden_size),
                             2.0,
                             dtype=torch.float16,
                             device="cuda")
    up_weight = torch.full_like(gate_weight, 3.0)
    q_a_weight = torch.full((config.q_lora_rank, config.hidden_size),
                            4.0,
                            dtype=torch.float16,
                            device="cuda")
    model.load_weights([
        ("model.layers.1.mlp.experts.0.gate_proj.weight", gate_weight),
        ("model.layers.1.mlp.experts.0.up_proj.weight", up_weight),
        ("model.layers.0.self_attn.q_a_proj.weight", q_a_weight),
    ])

    expert_weight = model.model.layers[1].mlp.experts[0].gate_up_proj.weight
    torch.testing.assert_close(expert_weight[:config.moe_intermediate_size],
                               gate_weight)
    torch.testing.assert_close(expert_weight[config.moe_intermediate_size:],
                               up_weight)
    torch.testing.assert_close(attention.q_a_proj.weight, q_a_weight)

    for parameter in model.parameters():
        parameter.data.normal_(mean=0.0, std=0.02)
    input_ids = torch.arange(1, 21, device="cuda")
    positions = torch.arange(input_ids.numel(), device="cuda")
    head_size = attention.attn.impl.head_size
    kv_caches = [
        torch.zeros((2, 4, cache_config.block_size *
                     config.num_attention_heads * head_size),
                    dtype=torch.float16,
                    device="cuda") for _ in range(config.num_hidden_layers)
    ]
    cache_ready = torch.cuda.Event()
    cache_ready.record(torch.cuda.current_stream())
    output = model(input_ids,
                   positions,
                   kv_caches,
                   _make_attention_metadata(input_ids.numel()),
                   layer_group_events={0: cache_ready})

    assert output.shape == (input_ids.numel(), config.hidden_size)
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()

    partial_kv_caches = [kv_cache.clone() for kv_cache in kv_caches]
    decode_output = model(torch.tensor([5], device="cuda"),
                          torch.tensor([input_ids.numel()], device="cuda"),
                          kv_caches,
                          _make_decode_attention_metadata(input_ids.numel()))
    assert decode_output.shape == (1, config.hidden_size)
    assert torch.isfinite(decode_output).all()

    num_evicted_tokens = cache_config.block_size
    x = 16 // partial_kv_caches[0].element_size()
    for partial_kv_cache in partial_kv_caches:
        key_cache = partial_kv_cache[0].view(
            4, config.num_attention_heads, head_size // x,
            cache_config.block_size, x)
        value_cache = partial_kv_cache[1].view(
            4, config.num_attention_heads, head_size,
            cache_config.block_size)
        key_cache[0].zero_()
        value_cache[0].zero_()

    cache_map = TokenLayerKVMap(
        num_layers=config.num_hidden_layers,
        block_size=cache_config.block_size,
        num_physical_blocks=4,
        layer_group_size=1,
    )
    cache_map.allocate_sequence(seq_id=1, num_tokens=input_ids.numel())
    cache_map.evict_prefix(seq_id=1, num_tokens=num_evicted_tokens)
    cache_map.append_token(seq_id=1)
    metadata_planner = LayerGroupMetadataPlanner(
        cache_map,
        seq_ids=[1],
        recompute_seq_lens=[num_evicted_tokens],
        decode_seq_lens=[input_ids.numel() + 1],
        device=torch.device("cuda"),
    )
    free_blocks_before_partial = cache_map.get_num_free_blocks()

    moe = model.model.layers[1].mlp
    with mock.patch.object(
            moe,
            "forward_with_partial_recompute",
            wraps=moe.forward_with_partial_recompute) as fused_moe_forward:
        with mock.patch.object(
                attention.attn,
                "forward",
                wraps=attention.attn.forward) as combined_attention_forward:
            partial_decode_output = model.forward_with_partial_recompute(
                recompute_input_ids=input_ids[:num_evicted_tokens],
                recompute_positions=positions[:num_evicted_tokens],
                recompute_attn_metadata=None,
                decode_input_ids=torch.tensor([5], device="cuda"),
                decode_positions=torch.tensor([input_ids.numel()],
                                              device="cuda"),
                kv_caches=partial_kv_caches,
                decode_attn_metadata=None,
                metadata_planner=metadata_planner,
            )
    fused_moe_forward.assert_called_once()
    combined_attention_forward.assert_called_once()
    assert cache_map.get_num_free_blocks() == free_blocks_before_partial
    torch.testing.assert_close(partial_decode_output,
                               decode_output,
                               atol=2e-3,
                               rtol=2e-3)

    mixed_full_caches = [kv_cache.clone() for kv_cache in kv_caches]
    for mixed_cache in mixed_full_caches:
        mixed_cache[:, 2].copy_(mixed_cache[:, 0])
        mixed_cache[:, 3].copy_(mixed_cache[:, 1])
    mixed_partial_caches = [
        mixed_cache.clone() for mixed_cache in mixed_full_caches
    ]
    for mixed_cache in mixed_partial_caches:
        mixed_cache[:, 2].zero_()

    mixed_decode_ids = torch.tensor([6, 6], device="cuda")
    mixed_decode_positions = torch.tensor([21, 21], device="cuda")
    mixed_expected = model(
        mixed_decode_ids,
        mixed_decode_positions,
        mixed_full_caches,
        _make_batched_decode_attention_metadata([22, 22], [[0, 1],
                                                            [2, 3]]),
    )

    mixed_cache_map = TokenLayerKVMap(
        num_layers=config.num_hidden_layers,
        block_size=cache_config.block_size,
        num_physical_blocks=4,
        layer_group_size=1,
    )
    mixed_cache_map.allocate_sequence(seq_id=1, num_tokens=21)
    mixed_cache_map.allocate_sequence(seq_id=2, num_tokens=21)
    mixed_cache_map.evict_prefix(seq_id=2, num_tokens=16)
    mixed_cache_map.append_token(seq_id=1)
    mixed_cache_map.append_token(seq_id=2)
    mixed_plans = mixed_cache_map.build_layer_group_plans([1, 2])
    mixed_recompute_metadata, mixed_decode_metadata = (
        build_layerwise_partial_recompute_metadata(
            mixed_plans,
            recompute_seq_lens=[0, 16],
            decode_seq_lens=[22, 22],
            num_layers=config.num_hidden_layers,
            layer_group_size=1,
            device=torch.device("cuda"),
        ))

    with mock.patch.object(
            moe,
            "forward_with_partial_recompute",
            wraps=moe.forward_with_partial_recompute) as mixed_moe_forward:
        mixed_actual = model.forward_with_partial_recompute(
            recompute_input_ids=input_ids[:16],
            recompute_positions=positions[:16],
            recompute_attn_metadata=mixed_recompute_metadata,
            decode_input_ids=mixed_decode_ids,
            decode_positions=mixed_decode_positions,
            kv_caches=mixed_partial_caches,
            decode_attn_metadata=mixed_decode_metadata,
        )
    mixed_moe_forward.assert_called_once()
    torch.testing.assert_close(mixed_actual,
                               mixed_expected,
                               atol=2e-3,
                               rtol=2e-3)

    lite_config = _make_config()
    lite_config.q_lora_rank = None
    with set_default_torch_dtype(torch.float16), torch.device("cuda"):
        lite_model = DeepseekV2ForCausalLM(lite_config,
                                           cache_config=cache_config)
    lite_attention = lite_model.model.layers[0].self_attn
    assert hasattr(lite_attention, "q_proj")
    assert not hasattr(lite_attention, "q_a_proj")
