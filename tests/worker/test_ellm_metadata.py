import torch

from vllm.attention.backends.xformers import XFormersMetadata
from vllm.core.ellm_cache import TokenLayerKVMap
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata
from vllm.worker.ellm_metadata import (
    LayerGroupMetadataPlanner, build_layerwise_attention_metadata,
    build_layerwise_partial_recompute_metadata,
    combine_partial_recompute_metadata)


def test_build_layerwise_partial_recompute_metadata():
    cache_map = TokenLayerKVMap(num_layers=6,
                                block_size=4,
                                num_physical_blocks=8,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=1, num_tokens=8)
    cache_map.allocate_sequence(seq_id=2, num_tokens=12)
    cache_map.evict_prefix(seq_id=1, num_tokens=4)
    cache_map.evict_prefix(seq_id=2, num_tokens=8)
    plans = cache_map.build_layer_group_plans([1, 2])

    recompute_metadata, decode_metadata = (
        build_layerwise_partial_recompute_metadata(
            plans,
            recompute_seq_lens=[4, 8],
            decode_seq_lens=[8, 12],
            num_layers=6,
            layer_group_size=4,
            device=torch.device("cpu"),
        ))

    assert len(recompute_metadata) == 6
    assert len(decode_metadata) == 6
    assert recompute_metadata[0].slot_mapping.numel() == 12
    assert recompute_metadata[0].query_start_loc.tolist() == [0, 4, 12]
    assert decode_metadata[0].slot_mapping.numel() == 2
    assert decode_metadata[0].block_tables.shape == (2, 3)
    assert torch.equal(decode_metadata[0].block_tables,
                       decode_metadata[3].block_tables)
    assert decode_metadata[3] is not decode_metadata[4]


def test_partial_metadata_filters_full_cache_sequences_from_recompute():
    cache_map = TokenLayerKVMap(num_layers=4,
                                block_size=4,
                                num_physical_blocks=8,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=1, num_tokens=4)
    cache_map.allocate_sequence(seq_id=2, num_tokens=8)
    cache_map.evict_prefix(seq_id=2, num_tokens=4)
    plans = cache_map.build_layer_group_plans([1, 2])

    recompute, decode = build_layerwise_partial_recompute_metadata(
        plans,
        recompute_seq_lens=[0, 4],
        decode_seq_lens=[4, 8],
        num_layers=4,
        layer_group_size=4,
        device=torch.device("cpu"),
    )

    assert recompute[0].num_prefills == 1
    assert recompute[0].seq_lens == [4]
    assert recompute[0].query_start_loc.tolist() == [0, 4]
    assert recompute[0].slot_mapping.numel() == 4
    assert decode[0].num_decode_tokens == 2
    assert decode[0].block_tables.shape == (2, 2)

    combined = combine_partial_recompute_metadata(recompute[0], decode[0])
    assert combined.num_prefill_tokens == 4
    assert combined.num_decode_tokens == 2
    assert combined.slot_mapping.numel() == 6
    assert combined.prefill_metadata is recompute[0]
    assert combined.decode_metadata is decode[0]


def test_metadata_planner_holds_and_releases_temporary_blocks():
    cache_map = TokenLayerKVMap(num_layers=8,
                                block_size=4,
                                num_physical_blocks=8,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=1, num_tokens=12)
    cache_map.evict_prefix(seq_id=1, num_tokens=8)
    free_before = cache_map.get_num_free_blocks()
    planner = LayerGroupMetadataPlanner(
        cache_map,
        seq_ids=[1],
        recompute_seq_lens=[8],
        decode_seq_lens=[12],
        device=torch.device("cpu"),
    )

    with planner.metadata_for_layer_group(0) as (recompute, decode):
        assert cache_map.get_num_free_blocks() == free_before - 2
        assert recompute.slot_mapping.numel() == 8
        assert decode.block_tables.shape == (1, 3)

    assert cache_map.get_num_free_blocks() == free_before


def test_build_layerwise_attention_metadata_uses_group_tables():
    seq_data = SequenceData([1, 2, 3, 4])
    metadata = SequenceGroupMetadata(
        request_id="request",
        is_prompt=True,
        seq_data={1: seq_data},
        sampling_params=SamplingParams(),
        block_tables={1: [2]},
        layer_group_block_tables={1: [[2], [5]]},
    )
    base = XFormersMetadata(
        num_prefills=1,
        num_prefill_tokens=4,
        num_decode_tokens=0,
        total_seq_len=4,
        slot_mapping=torch.tensor([8, 9, 10, 11]),
        seq_lens=[4],
        seq_lens_tensor=torch.tensor([4], dtype=torch.int32),
        max_query_len=4,
        max_prefill_seq_len=4,
        max_decode_seq_len=0,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        context_lens_tensor=torch.tensor([0], dtype=torch.int32),
        block_tables=torch.empty((1, 0), dtype=torch.int32),
        use_cuda_graph=False,
    )

    layerwise = build_layerwise_attention_metadata(
        base,
        [metadata],
        num_layers=4,
        layer_group_size=2,
        block_size=4,
        device=torch.device("cpu"),
    )

    assert [item.slot_mapping.tolist() for item in layerwise
            ] == [[8, 9, 10, 11], [8, 9, 10, 11], [20, 21, 22, 23],
                  [20, 21, 22, 23]]


def test_layerwise_metadata_offsets_compact_retained_tables():
    seq_data = SequenceData(list(range(9)))
    metadata = SequenceGroupMetadata(
        request_id="request",
        is_prompt=False,
        seq_data={1: seq_data},
        sampling_params=SamplingParams(),
        block_tables={1: [5]},
        layer_group_block_tables={1: [[5], [7]]},
    )
    base = XFormersMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        total_seq_len=9,
        slot_mapping=torch.tensor([20]),
        seq_lens=None,
        seq_lens_tensor=torch.tensor([9], dtype=torch.int32),
        max_query_len=None,
        max_prefill_seq_len=0,
        max_decode_seq_len=9,
        query_start_loc=None,
        seq_start_loc=None,
        context_lens_tensor=None,
        block_tables=torch.tensor([[5]], dtype=torch.int32),
        use_cuda_graph=False,
    )

    layerwise = build_layerwise_attention_metadata(
        base,
        [metadata],
        num_layers=4,
        layer_group_size=2,
        block_size=4,
        device=torch.device("cpu"),
    )

    assert [item.slot_mapping.tolist()
            for item in layerwise] == [[20], [20], [28], [28]]
