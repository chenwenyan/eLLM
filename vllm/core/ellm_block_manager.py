"""Layer-group-aware GPU block manager for eLLM."""

import math
from collections import deque
from typing import Deque, Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.core.ellm_cache import DEFAULT_LAYER_GROUP_SIZE, TokenLayerKVMap
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus


class ELLMBlockSpaceManager(BlockSpaceManager):
    """Allocate an independent physical block table per layer group.

    The cache engine already owns one physical block pool per model layer.
    Unlike the conventional manager, this manager allows the same physical
    block number to represent different token ranges in different layer
    groups. Swap and beam-search copy-on-write mappings therefore include a
    layer-group dimension.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        num_layers: int,
        layer_group_size: int = DEFAULT_LAYER_GROUP_SIZE,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        flatten_layers: int = 0,
    ) -> None:
        if sliding_window is not None:
            raise NotImplementedError("eLLM does not support sliding window")
        if enable_caching:
            raise NotImplementedError("eLLM prefix sharing is not implemented")
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.num_layers = num_layers
        self.layer_group_size = layer_group_size
        self.flatten_layers = flatten_layers
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.cache_map = TokenLayerKVMap(
            num_layers=num_layers,
            block_size=block_size,
            num_physical_blocks=num_gpu_blocks,
            layer_group_size=layer_group_size,
        )
        self._free_cpu_blocks: List[Deque[int]] = [
            deque(range(num_cpu_blocks))
            for _ in range(self.cache_map.num_layer_groups)
        ]
        self._swapped_sequences: Dict[
            int, Tuple[int, int, List[List[int]]]] = {}

    def _required_blocks(self, seq: Sequence) -> int:
        return math.ceil(seq.get_len() / self.block_size)

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        waiting = seq_group.get_seqs(status=SequenceStatus.WAITING)
        required = sum(self._required_blocks(seq) for seq in waiting)
        if required + self.watermark_blocks > self.num_total_gpu_blocks:
            return AllocStatus.NEVER
        min_free = min(
            self.cache_map.get_num_free_blocks(group)
            for group in range(self.cache_map.num_layer_groups))
        if required + self.watermark_blocks <= min_free:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        allocated: List[int] = []
        try:
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                self.cache_map.allocate_sequence(seq.seq_id, seq.get_len())
                allocated.append(seq.seq_id)
        except Exception:
            for seq_id in allocated:
                self.cache_map.free_sequence(seq_id)
            raise

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        if num_lookahead_slots:
            return False
        needs_by_group = [0] * self.cache_map.num_layer_groups
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            old_length = self.cache_map.get_sequence_length(seq.seq_id)
            if seq.get_len() <= old_length:
                continue
            if seq.get_len() % self.block_size == 1:
                needs_by_group = [count + 1 for count in needs_by_group]
                continue
            for group in range(self.cache_map.num_layer_groups):
                if self.cache_map.sequence_needs_cow(seq.seq_id, group):
                    needs_by_group[group] += 1
        return all(
            self.cache_map.get_num_free_blocks(group) >= needs_by_group[group]
            for group in range(self.cache_map.num_layer_groups))

    def append_slots(self,
                     seq: Sequence,
                     num_lookahead_slots: int = 0) -> List[Tuple[int, int]]:
        if num_lookahead_slots:
            raise NotImplementedError("eLLM lookahead is not implemented")
        copy_mappings = []
        while self.cache_map.get_sequence_length(seq.seq_id) < seq.get_len():
            copy_mappings.extend(self.cache_map.append_token(seq.seq_id))
        return copy_mappings

    def get_layer_group_block_tables(self, seq: Sequence) -> List[List[int]]:
        return [
            self.cache_map.get_retained_block_table(seq.seq_id, group)
            for group in range(self.cache_map.num_layer_groups)
        ]

    def evict_to_cached_layer_ratio(self, seq: Sequence,
                                    cached_layers: int) -> int:
        """Translate the selected cache ratio into an old-prefix eviction."""
        cached_layers = min(max(cached_layers, 0), self.num_layers)
        num_blocks = math.ceil(seq.get_len() / self.block_size)
        retained_blocks = max(
            1, math.ceil(num_blocks * cached_layers / self.num_layers))
        evicted_blocks = max(num_blocks - retained_blocks, 0)
        return self.cache_map.evict_prefix(
            seq.seq_id, evicted_blocks * self.block_size)

    def build_partial_recompute_plans(self, seq_ids: List[int]):
        return self.cache_map.build_layer_group_plans(seq_ids)

    def get_num_evicted_tokens(self, seq_id: int) -> int:
        return self.cache_map.get_num_evicted_tokens(seq_id)

    def get_block_table(self, seq: Sequence) -> List[int]:
        return self.get_layer_group_block_tables(seq)[0]

    def get_seq_used_block_id(self, seq_group: SequenceGroup) -> List[int]:
        block_ids = []
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            for table in self.get_layer_group_block_tables(seq):
                block_ids.extend(table)
        return block_ids

    def free(self, seq: Sequence) -> None:
        self.cache_map.free_sequence(seq.seq_id)

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.cache_map.fork_sequence(parent_seq.seq_id, child_seq.seq_id)

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int = 0) -> AllocStatus:
        if num_lookahead_slots:
            return AllocStatus.NEVER
        required_by_group = [0] * self.cache_map.num_layer_groups
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if seq.seq_id not in self._swapped_sequences:
                return AllocStatus.NEVER
            _, _, cpu_tables = self._swapped_sequences[seq.seq_id]
            for group, table in enumerate(cpu_tables):
                required_by_group[group] += len(table)
        feasible = all(
            self.cache_map.get_num_free_blocks(group) >= required
            for group, required in enumerate(required_by_group))
        return AllocStatus.OK if feasible else AllocStatus.LATER

    def swap_in(self, seq_group: SequenceGroup,
                num_lookahead_slots: int = 0) -> List[Tuple[int, int]]:
        if num_lookahead_slots:
            raise NotImplementedError("eLLM lookahead is not implemented")
        mappings = []
        restored = []
        try:
            for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
                length, evicted_tokens, cpu_tables = (
                    self._swapped_sequences[seq.seq_id])
                self.cache_map.allocate_sequence_with_evicted_prefix(
                    seq.seq_id, length, evicted_tokens)
                restored.append(seq.seq_id)
                gpu_tables = self.get_layer_group_block_tables(seq)
                for group, (cpu_table, gpu_table) in enumerate(
                        zip(cpu_tables, gpu_tables)):
                    mappings.extend((group, cpu_id, gpu_id)
                                    for cpu_id, gpu_id in zip(
                                        cpu_table, gpu_table))
        except Exception:
            for seq_id in restored:
                self.cache_map.free_sequence(seq_id)
            raise

        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            _, _, cpu_tables = self._swapped_sequences.pop(seq.seq_id)
            for group, table in enumerate(cpu_tables):
                self._free_cpu_blocks[group].extend(table)
        return mappings

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        required_by_group = [0] * self.cache_map.num_layer_groups
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            for group, table in enumerate(
                    self.get_layer_group_block_tables(seq)):
                required_by_group[group] += len(table)
        return all(len(self._free_cpu_blocks[group]) >= required
                   for group, required in enumerate(required_by_group))

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        if not self.can_swap_out(seq_group):
            raise RuntimeError("No free CPU blocks for eLLM swap-out")
        mappings = []
        allocated: List[Tuple[int, int]] = []
        states = {}
        try:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                gpu_tables = self.get_layer_group_block_tables(seq)
                cpu_tables = []
                for group, gpu_table in enumerate(gpu_tables):
                    cpu_table = []
                    for gpu_id in gpu_table:
                        cpu_id = self._free_cpu_blocks[group].popleft()
                        allocated.append((group, cpu_id))
                        cpu_table.append(cpu_id)
                        mappings.append((group, gpu_id, cpu_id))
                    cpu_tables.append(cpu_table)
                states[seq.seq_id] = (
                    self.cache_map.get_sequence_length(seq.seq_id),
                    self.cache_map.get_num_evicted_tokens(seq.seq_id),
                    cpu_tables,
                )
        except Exception:
            for group, cpu_id in allocated:
                self._free_cpu_blocks[group].append(cpu_id)
            raise

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            self._swapped_sequences[seq.seq_id] = states[seq.seq_id]
            self.cache_map.free_sequence(seq.seq_id)
        return mappings

    def get_num_free_gpu_blocks(self) -> int:
        return min(
            self.cache_map.get_num_free_blocks(group)
            for group in range(self.cache_map.num_layer_groups))

    def get_num_free_cpu_blocks(self) -> int:
        return min(len(blocks) for blocks in self._free_cpu_blocks)

    def access_all_blocks_in_seq(self, seq: Sequence,
                                 access_time: float) -> None:
        pass

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        return []

    def mark_blocks_as_computed(self, seq_group: SequenceGroup) -> None:
        pass
