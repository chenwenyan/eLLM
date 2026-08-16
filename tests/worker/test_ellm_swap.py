from unittest.mock import MagicMock

import pytest
import torch

from vllm.worker.cache_engine import CacheEngine
from vllm.worker.worker import Worker


def test_cache_engine_applies_swap_only_to_mapped_layer_group():
    engine = object.__new__(CacheEngine)
    engine.num_layers = 6
    engine.attn_backend = MagicMock()
    source = [object() for _ in range(6)]
    destination = [object() for _ in range(6)]
    mappings = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.int64)

    engine._swap_layer_groups(source, destination, mappings)

    assert engine.attn_backend.swap_blocks.call_count == 6
    for layer_idx, call in enumerate(
            engine.attn_backend.swap_blocks.call_args_list):
        assert call.args[0] is source[layer_idx]
        assert call.args[1] is destination[layer_idx]
        expected = [[1, 2]] if layer_idx < 4 else [[3, 4]]
        assert call.args[2].tolist() == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cache_engine_records_layer_group_swap_events():

    class CopyBackend:

        @staticmethod
        def swap_blocks(source, destination, mappings):
            for source_id, destination_id in mappings.tolist():
                destination[destination_id].copy_(source[source_id],
                                                  non_blocking=True)

        @staticmethod
        def copy_blocks(caches, mappings):
            for cache in caches:
                for source_id, destination_id in mappings.tolist():
                    cache[destination_id].copy_(cache[source_id])

    engine = object.__new__(CacheEngine)
    engine.num_layers = 8
    engine.attn_backend = CopyBackend()
    engine.swap_in_stream = torch.cuda.Stream()
    engine.swap_out_stream = torch.cuda.Stream()
    engine.cpu_cache = [
        torch.full((4, 8), layer + 1, pin_memory=True)
        for layer in range(8)
    ]
    engine.gpu_cache = [torch.zeros((4, 8), device="cuda") for _ in range(8)]
    mappings = torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.int64)

    events = engine.swap_in_async(mappings)

    assert set(events) == {0, 1}
    torch.cuda.current_stream().wait_event(events[0])
    torch.cuda.current_stream().synchronize()
    for layer in range(4):
        assert torch.all(engine.gpu_cache[layer][1] == layer + 1)
    events[1].synchronize()
    for layer in range(4, 8):
        assert torch.all(engine.gpu_cache[layer][3] == layer + 1)

    swap_out_mappings = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    swap_out_event = engine.swap_out_async(swap_out_mappings)
    assert swap_out_event is not None
    swap_out_event.synchronize()
    for layer in range(4):
        assert torch.all(engine.cpu_cache[layer][2] == layer + 1)

    engine.copy(torch.tensor([[0, 1, 0], [1, 3, 0]],
                             dtype=torch.int64,
                             device="cuda"))
    torch.cuda.current_stream().synchronize()
    for layer in range(8):
        assert torch.all(engine.gpu_cache[layer][0] == layer + 1)


def test_worker_adds_dependencies_only_for_reused_swap_out_blocks():
    worker = object.__new__(Worker)
    worker.cache_engine = MagicMock()
    swap_out_event = MagicMock()
    worker.cache_engine.swap_out_async.return_value = swap_out_event
    worker.cache_engine.swap_in_async.return_value = {}
    blocks_to_copy = torch.empty((0, 2), dtype=torch.int64)

    _, returned_event, wait_for_swap_out = worker.cache_swap_async(
        blocks_to_swap_in=torch.tensor([[0, 3, 1]], dtype=torch.int64),
        blocks_to_swap_out=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        blocks_to_copy=blocks_to_copy,
        current_block_ids=torch.tensor([1], dtype=torch.int64),
    )

    assert returned_event is swap_out_event
    assert wait_for_swap_out
    worker.cache_engine.swap_in_stream.wait_event.assert_called_once_with(
        swap_out_event)

    worker.cache_engine.swap_in_stream.reset_mock()
    _, _, wait_for_swap_out = worker.cache_swap_async(
        blocks_to_swap_in=torch.tensor([[0, 3, 4]], dtype=torch.int64),
        blocks_to_swap_out=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        blocks_to_copy=blocks_to_copy,
        current_block_ids=torch.tensor([5], dtype=torch.int64),
    )

    assert not wait_for_swap_out
    worker.cache_engine.swap_in_stream.wait_event.assert_not_called()


def test_cache_engine_copies_only_cow_layer_groups():
    engine = object.__new__(CacheEngine)
    engine.num_layers = 6
    engine.attn_backend = MagicMock()
    engine.gpu_cache = [object() for _ in range(6)]
    mappings = torch.tensor([[0, 1, 2], [1, 3, 4]], dtype=torch.int64)

    engine.copy(mappings)

    assert engine.attn_backend.copy_blocks.call_count == 2
    first, second = engine.attn_backend.copy_blocks.call_args_list
    assert first.args[0] == engine.gpu_cache[:4]
    assert first.args[1].tolist() == [[1, 2]]
    assert second.args[0] == engine.gpu_cache[4:6]
    assert second.args[1].tolist() == [[3, 4]]
