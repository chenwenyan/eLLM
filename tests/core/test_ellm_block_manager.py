from unittest.mock import MagicMock

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.ellm_block_manager import ELLMBlockSpaceManager
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import Scheduler
from vllm.sequence import Logprob, SequenceStatus

from .utils import create_dummy_prompt


def test_ellm_block_manager_allocates_each_layer_group():
    manager = ELLMBlockSpaceManager(
        block_size=4,
        num_gpu_blocks=6,
        num_cpu_blocks=0,
        num_layers=8,
        layer_group_size=4,
    )
    seq, seq_group = create_dummy_prompt("1",
                                         prompt_length=8,
                                         block_size=4)

    assert manager.can_allocate(seq_group) == AllocStatus.OK
    manager.allocate(seq_group)
    seq.status = SequenceStatus.RUNNING

    tables = manager.get_layer_group_block_tables(seq)
    assert len(tables) == 2
    assert all(len(table) == 2 for table in tables)
    assert manager.get_num_free_gpu_blocks() == 4

    seq.append_token_id(1, {1: Logprob(0.0)})
    assert manager.can_append_slots(seq_group)
    manager.append_slots(seq)
    assert all(len(table) == 3
               for table in manager.get_layer_group_block_tables(seq))
    assert manager.get_num_free_gpu_blocks() == 3

    manager.free(seq)
    assert manager.get_num_free_gpu_blocks() == 6


def test_ellm_block_manager_reports_per_group_capacity():
    manager = ELLMBlockSpaceManager(
        block_size=4,
        num_gpu_blocks=2,
        num_cpu_blocks=0,
        num_layers=8,
        layer_group_size=4,
    )
    _, seq_group = create_dummy_prompt("2",
                                       prompt_length=12,
                                       block_size=4)

    assert manager.can_allocate(seq_group) == AllocStatus.NEVER


def test_scheduler_builds_partial_recompute_plan_for_decode():
    cache_config = CacheConfig(block_size=4,
                               gpu_memory_utilization=1.0,
                               swap_space=0,
                               cache_dtype="auto",
                               store_cache_layers=0.5,
                               flatten_layers=4)
    cache_config.num_gpu_blocks = 8
    cache_config.num_cpu_blocks = 0
    cache_config.num_layers = 8
    scheduler_config = SchedulerConfig(max_num_batched_tokens=32,
                                       max_num_seqs=4,
                                       max_model_len=32)
    scheduler = Scheduler(scheduler_config,
                          cache_config,
                          model_config=MagicMock(),
                          lora_config=None)
    seq, seq_group = create_dummy_prompt("3",
                                         prompt_length=8,
                                         block_size=4)
    scheduler.add_seq_group(seq_group)

    prefill_metadata, prefill_output = scheduler.schedule()
    assert prefill_metadata[0].layer_group_block_tables is not None
    seq_group.update_num_computed_tokens(
        prefill_output.scheduled_seq_groups[0].token_chunk_size)
    seq.append_token_id(1, {1: Logprob(0.0)})

    decode_metadata, _ = scheduler.schedule()
    metadata = decode_metadata[0]
    assert metadata.ellm_recompute_seq_lens == [4]
    assert len(metadata.ellm_layer_group_plans) == 2
    assert len(metadata.ellm_layer_group_plans[0].recompute_slot_mapping) == 4


def test_scheduler_allows_mixed_full_and_partial_decode_batch():
    cache_config = CacheConfig(block_size=4,
                               gpu_memory_utilization=1.0,
                               swap_space=0,
                               cache_dtype="auto",
                               store_cache_layers=0.5,
                               flatten_layers=4)
    cache_config.num_gpu_blocks = 8
    cache_config.num_cpu_blocks = 0
    cache_config.num_layers = 8
    scheduler_config = SchedulerConfig(max_num_batched_tokens=32,
                                       max_num_seqs=4,
                                       max_model_len=32)
    scheduler = Scheduler(scheduler_config,
                          cache_config,
                          model_config=MagicMock(),
                          lora_config=None)
    scheduler.get_opt_bs_and_layers = lambda: ([4], [4])
    short_seq, short_group = create_dummy_prompt("4",
                                                  prompt_length=3,
                                                  block_size=4)
    long_seq, long_group = create_dummy_prompt("5",
                                                prompt_length=8,
                                                block_size=4)
    scheduler.add_seq_group(short_group)
    scheduler.add_seq_group(long_group)

    _, prefill_output = scheduler.schedule()
    for scheduled in prefill_output.scheduled_seq_groups:
        scheduled.seq_group.update_num_computed_tokens(
            scheduled.token_chunk_size)
    short_seq.append_token_id(1, {1: Logprob(0.0)})
    long_seq.append_token_id(1, {1: Logprob(0.0)})

    decode_metadata, _ = scheduler.schedule()
    assert decode_metadata[0].ellm_recompute_seq_lens == [0, 4]
    assert decode_metadata[1].ellm_recompute_seq_lens == [0, 4]


def test_ellm_block_manager_swaps_retained_layer_group_blocks():
    manager = ELLMBlockSpaceManager(
        block_size=4,
        num_gpu_blocks=6,
        num_cpu_blocks=6,
        num_layers=8,
        layer_group_size=4,
    )
    seq, seq_group = create_dummy_prompt("6",
                                         prompt_length=12,
                                         block_size=4)
    manager.allocate(seq_group)
    seq.status = SequenceStatus.RUNNING
    manager.evict_to_cached_layer_ratio(seq, cached_layers=4)
    free_gpu_before = manager.get_num_free_gpu_blocks()

    swap_out = manager.swap_out(seq_group)
    assert all(len(mapping) == 3 for mapping in swap_out)
    assert {mapping[0] for mapping in swap_out} == {0, 1}
    assert manager.get_num_free_gpu_blocks() > free_gpu_before
    seq.status = SequenceStatus.SWAPPED
    assert manager.can_swap_in(seq_group) == AllocStatus.OK

    swap_in = manager.swap_in(seq_group)
    assert all(len(mapping) == 3 for mapping in swap_in)
    assert manager.get_num_evicted_tokens(seq.seq_id) == 4
    assert all(len(table) == 2
               for table in manager.get_layer_group_block_tables(seq))
    assert manager.get_num_free_cpu_blocks() == 6


def test_ellm_block_manager_forks_and_returns_group_cow_mappings():
    manager = ELLMBlockSpaceManager(
        block_size=4,
        num_gpu_blocks=6,
        num_cpu_blocks=0,
        num_layers=8,
        layer_group_size=4,
    )
    parent, seq_group = create_dummy_prompt("7",
                                            prompt_length=3,
                                            block_size=4)
    manager.allocate(seq_group)
    parent.status = SequenceStatus.RUNNING
    child = parent.fork(8)
    child.status = SequenceStatus.RUNNING
    manager.fork(parent, child)
    free_before_append = manager.get_num_free_gpu_blocks()

    parent.append_token_id(11, {11: Logprob(0.0)})
    mappings = manager.append_slots(parent)

    assert len(mappings) == 2
    assert all(len(mapping) == 3 for mapping in mappings)
    assert manager.get_num_free_gpu_blocks() == free_before_append - 1
    assert manager.get_layer_group_block_tables(parent) != (
        manager.get_layer_group_block_tables(child))


def test_ellm_block_manager_reserves_capacity_for_cow():
    manager = ELLMBlockSpaceManager(
        block_size=4,
        num_gpu_blocks=1,
        num_cpu_blocks=0,
        num_layers=4,
        layer_group_size=4,
    )
    parent, seq_group = create_dummy_prompt("9",
                                            prompt_length=3,
                                            block_size=4)
    manager.allocate(seq_group)
    parent.status = SequenceStatus.RUNNING
    child = parent.fork(10)
    child.status = SequenceStatus.RUNNING
    manager.fork(parent, child)
    parent.append_token_id(11, {11: Logprob(0.0)})

    assert not manager.can_append_slots(seq_group)
