import pytest

from vllm.core.ellm_cache import TokenLayerKVMap


def test_token_layer_map_evicts_prefix_and_reuses_one_layer_group():
    cache_map = TokenLayerKVMap(num_layers=8,
                                block_size=4,
                                num_physical_blocks=8,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=11, num_tokens=12)

    assert cache_map.get_num_free_blocks() == 10
    assert len(cache_map.get_map_entries(11)) == 6

    assert cache_map.evict_prefix(11, num_tokens=9) == 8
    assert cache_map.get_num_free_blocks() == 14
    assert len(cache_map.get_map_entries(11)) == 2

    decode_table = cache_map.begin_recompute(11, layer_group=0)
    assert len(decode_table) == 3
    expected_recompute_slots = []
    for physical_block_id in decode_table[:2]:
        expected_recompute_slots.extend(
            range(physical_block_id * 4, physical_block_id * 4 + 4))
    assert cache_map.get_recompute_slot_mapping(
        11, 0) == expected_recompute_slots
    assert cache_map.get_decode_slot(11, 0) == decode_table[-1] * 4 + 3
    assert cache_map.get_layer_group(0) == 0
    assert cache_map.get_layer_group(4) == 1
    assert cache_map.get_num_evicted_tokens(11) == 8
    assert cache_map.get_num_free_blocks() == 12
    assert len(cache_map.get_map_entries(11)) == 4
    assert sum(entry.temporary
               for entry in cache_map.get_map_entries(11)) == 2

    cache_map.end_recompute(11, layer_group=0)
    assert cache_map.get_num_free_blocks() == 14

    second_group_table = cache_map.begin_recompute(11, layer_group=1)
    assert len(second_group_table) == 3
    cache_map.end_recompute(11, layer_group=1)

    plans = cache_map.build_layer_group_plans([11])
    assert len(plans) == 2
    assert len(plans[0].recompute_slot_mapping) == 8
    assert len(plans[0].decode_block_tables[0]) == 3
    assert cache_map.get_num_free_blocks() == 14


def test_token_layer_map_tracks_filled_tokens_and_append():
    cache_map = TokenLayerKVMap(num_layers=5,
                                block_size=4,
                                num_physical_blocks=4,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=7, num_tokens=3)

    assert {entry.num_filled for entry in cache_map.get_map_entries(7)} == {3}
    cache_map.append_token(7)
    assert {entry.num_filled for entry in cache_map.get_map_entries(7)} == {4}
    cache_map.append_token(7)
    assert sorted(entry.num_filled
                  for entry in cache_map.get_map_entries(7)) == [1, 1, 4, 4]

    cache_map.free_sequence(7)
    assert cache_map.get_num_free_blocks() == 8
    assert cache_map.get_map_entries(7) == []


def test_token_layer_map_rolls_back_failed_allocation():
    cache_map = TokenLayerKVMap(num_layers=8,
                                block_size=4,
                                num_physical_blocks=2,
                                layer_group_size=4)

    with pytest.raises(RuntimeError, match="No free KV blocks"):
        cache_map.allocate_sequence(seq_id=1, num_tokens=12)

    assert cache_map.get_num_free_blocks() == 4
    assert cache_map.get_map_entries() == []


def test_layer_group_plan_lease_keeps_blocks_reserved():
    cache_map = TokenLayerKVMap(num_layers=8,
                                block_size=4,
                                num_physical_blocks=8,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=1, num_tokens=12)
    cache_map.evict_prefix(seq_id=1, num_tokens=8)
    free_before = cache_map.get_num_free_blocks()

    with cache_map.lease_layer_group_plan([1], 0) as plan:
        assert len(plan.recompute_slot_mapping) == 8
        assert cache_map.get_num_free_blocks() == free_before - 2
        with pytest.raises(RuntimeError,
                           match="already has active recompute"):
            cache_map.acquire_layer_group_plan([1], 0)

    assert cache_map.get_num_free_blocks() == free_before
    assert not any(entry.temporary
                   for entry in cache_map.get_map_entries(seq_id=1))


def test_fork_shares_blocks_and_append_uses_layer_group_cow():
    cache_map = TokenLayerKVMap(num_layers=8,
                                block_size=4,
                                num_physical_blocks=4,
                                layer_group_size=4)
    cache_map.allocate_sequence(seq_id=1, num_tokens=3)
    parent_tables = [
        cache_map.get_retained_block_table(1, group) for group in range(2)
    ]
    free_before_fork = cache_map.get_num_free_blocks()

    cache_map.fork_sequence(parent_seq_id=1, child_seq_id=2)

    assert cache_map.get_num_free_blocks() == free_before_fork
    assert [cache_map.get_retained_block_table(2, group)
            for group in range(2)] == parent_tables
    assert all(
        cache_map.get_block_ref_count(group, parent_tables[group][0]) == 2
        for group in range(2))

    copy_mappings = cache_map.append_token(seq_id=1)

    assert len(copy_mappings) == 2
    assert {mapping[0] for mapping in copy_mappings} == {0, 1}
    assert all(mapping[1] == parent_tables[mapping[0]][0]
               for mapping in copy_mappings)
    assert [cache_map.get_retained_block_table(1, group)
            for group in range(2)] != parent_tables
    assert [cache_map.get_retained_block_table(2, group)
            for group in range(2)] == parent_tables

    assert cache_map.append_token(seq_id=2) == []
    cache_map.free_sequence(1)
    cache_map.free_sequence(2)
    assert cache_map.get_num_free_blocks() == 8
