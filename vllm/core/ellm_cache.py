"""Token- and layer-aware KV block mapping for eLLM partial recompute."""

import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Deque, Dict, Iterator, List, Optional, Tuple

DEFAULT_LAYER_GROUP_SIZE = 4


@dataclass(frozen=True)
class KVMapEntry:
    """Address of one token block in one consecutive layer group."""

    seq_id: int
    token_start: int
    layer_start: int
    logical_block_id: int
    physical_block_id: int
    num_filled: int
    temporary: bool = False


@dataclass(frozen=True)
class LayerGroupBlockPlan:
    """PagedAttention addresses for all requests in one layer group."""

    recompute_slot_mapping: List[int]
    decode_slot_mapping: List[int]
    decode_block_tables: List[List[int]]


class TokenLayerKVMap:
    """Manage eLLM KV blocks independently at token and layer granularity.

    A physical block ID belongs to one layer group, so the same numeric ID can
    be used concurrently by different groups. This matches a cache engine with
    one physical pool per layer and lets temporary prefix K/V occupy only the
    layer group currently being recomputed.
    """

    def __init__(
        self,
        num_layers: int,
        block_size: int,
        num_physical_blocks: int,
        layer_group_size: int = DEFAULT_LAYER_GROUP_SIZE,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_physical_blocks <= 0:
            raise ValueError("num_physical_blocks must be positive")
        if layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive")

        self.num_layers = num_layers
        self.block_size = block_size
        self.num_physical_blocks = num_physical_blocks
        self.layer_group_size = layer_group_size
        self.num_layer_groups = math.ceil(num_layers / layer_group_size)
        self._free_blocks: List[Deque[int]] = [
            deque(range(num_physical_blocks))
            for _ in range(self.num_layer_groups)
        ]
        self._block_ref_counts: List[List[int]] = [
            [0] * num_physical_blocks for _ in range(self.num_layer_groups)
        ]
        self._entries: Dict[Tuple[int, int, int], KVMapEntry] = {}
        self._temporary_entries: Dict[Tuple[int, int, int], KVMapEntry] = {}
        self._sequence_lengths: Dict[int, int] = {}
        self._evicted_prefix_blocks: Dict[int, int] = {}

    def _allocate_block(self, layer_group: int) -> int:
        try:
            physical_block_id = self._free_blocks[layer_group].popleft()
        except IndexError as exc:
            raise RuntimeError(
                f"No free KV blocks in layer group {layer_group}") from exc
        if self._block_ref_counts[layer_group][physical_block_id] != 0:
            raise RuntimeError("Free KV block has a non-zero reference count")
        self._block_ref_counts[layer_group][physical_block_id] = 1
        return physical_block_id

    def _retain_block(self, layer_group: int, physical_block_id: int) -> None:
        ref_counts = self._block_ref_counts[layer_group]
        if ref_counts[physical_block_id] == 0:
            raise ValueError("Cannot retain a free KV block")
        ref_counts[physical_block_id] += 1

    def _free_block(self, layer_group: int, physical_block_id: int) -> None:
        ref_counts = self._block_ref_counts[layer_group]
        if ref_counts[physical_block_id] == 0:
            raise ValueError("KV physical block was freed twice")
        ref_counts[physical_block_id] -= 1
        if ref_counts[physical_block_id] > 0:
            return
        free_blocks = self._free_blocks[layer_group]
        free_blocks.append(physical_block_id)

    def _num_logical_blocks(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def _num_filled(self, num_tokens: int, logical_block_id: int) -> int:
        remaining = num_tokens - logical_block_id * self.block_size
        return min(self.block_size, max(remaining, 0))

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> None:
        self.allocate_sequence_with_evicted_prefix(seq_id, num_tokens, 0)

    def allocate_sequence_with_evicted_prefix(
        self,
        seq_id: int,
        num_tokens: int,
        num_evicted_tokens: int,
    ) -> None:
        """Allocate only the retained suffix when restoring a sequence."""
        if seq_id in self._sequence_lengths:
            raise ValueError(f"Sequence {seq_id} is already allocated")
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if num_evicted_tokens < 0 or num_evicted_tokens % self.block_size:
            raise ValueError("evicted tokens must be non-negative and aligned")
        prefix_blocks = num_evicted_tokens // self.block_size
        if prefix_blocks > self._num_logical_blocks(num_tokens):
            raise ValueError("evicted prefix exceeds the sequence")

        allocated_keys: List[Tuple[int, int, int]] = []
        try:
            for layer_group in range(self.num_layer_groups):
                layer_start = layer_group * self.layer_group_size
                for logical_block_id in range(prefix_blocks,
                        self._num_logical_blocks(num_tokens)):
                    physical_block_id = self._allocate_block(layer_group)
                    key = (seq_id, logical_block_id, layer_group)
                    self._entries[key] = KVMapEntry(
                        seq_id=seq_id,
                        token_start=logical_block_id * self.block_size,
                        layer_start=layer_start,
                        logical_block_id=logical_block_id,
                        physical_block_id=physical_block_id,
                        num_filled=self._num_filled(num_tokens,
                                                    logical_block_id),
                    )
                    allocated_keys.append(key)
        except Exception:
            for key in allocated_keys:
                entry = self._entries.pop(key)
                self._free_block(key[2], entry.physical_block_id)
            raise

        self._sequence_lengths[seq_id] = num_tokens
        self._evicted_prefix_blocks[seq_id] = prefix_blocks

    def append_token(self, seq_id: int) -> List[Tuple[int, int, int]]:
        old_num_tokens = self._sequence_lengths[seq_id]
        new_num_tokens = old_num_tokens + 1
        old_num_blocks = self._num_logical_blocks(old_num_tokens)
        new_num_blocks = self._num_logical_blocks(new_num_tokens)
        if new_num_blocks > old_num_blocks:
            logical_block_id = new_num_blocks - 1
            allocated: List[Tuple[int, int, int]] = []
            try:
                for layer_group in range(self.num_layer_groups):
                    physical_block_id = self._allocate_block(layer_group)
                    key = (seq_id, logical_block_id, layer_group)
                    self._entries[key] = KVMapEntry(
                        seq_id=seq_id,
                        token_start=logical_block_id * self.block_size,
                        layer_start=layer_group * self.layer_group_size,
                        logical_block_id=logical_block_id,
                        physical_block_id=physical_block_id,
                        num_filled=1,
                    )
                    allocated.append(key)
            except Exception:
                for key in allocated:
                    entry = self._entries.pop(key)
                    self._free_block(key[2], entry.physical_block_id)
                raise
            copy_mappings: List[Tuple[int, int, int]] = []
        else:
            logical_block_id = new_num_blocks - 1
            cow_blocks: Dict[int, int] = {}
            try:
                for layer_group in range(self.num_layer_groups):
                    key = (seq_id, logical_block_id, layer_group)
                    entry = self._entries[key]
                    ref_count = self._block_ref_counts[layer_group][
                        entry.physical_block_id]
                    if ref_count > 1:
                        cow_blocks[layer_group] = self._allocate_block(
                            layer_group)
            except Exception:
                for layer_group, physical_block_id in cow_blocks.items():
                    self._free_block(layer_group, physical_block_id)
                raise

            copy_mappings = []
            for layer_group in range(self.num_layer_groups):
                key = (seq_id, logical_block_id, layer_group)
                entry = self._entries[key]
                if layer_group in cow_blocks:
                    new_block_id = cow_blocks[layer_group]
                    self._free_block(layer_group, entry.physical_block_id)
                    copy_mappings.append((layer_group,
                                          entry.physical_block_id,
                                          new_block_id))
                    self._entries[key] = replace(
                        entry,
                        physical_block_id=new_block_id,
                        num_filled=entry.num_filled + 1,
                    )
                else:
                    self._entries[key] = replace(
                        entry, num_filled=entry.num_filled + 1)
        self._sequence_lengths[seq_id] = new_num_tokens
        return copy_mappings

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> None:
        """Share all retained blocks with a new beam-search child."""
        if child_seq_id in self._sequence_lengths:
            raise ValueError(f"Sequence {child_seq_id} is already allocated")
        if any(key[0] == parent_seq_id for key in self._temporary_entries):
            raise RuntimeError("Cannot fork a sequence during recompute")
        parent_entries = [
            (key, entry) for key, entry in self._entries.items()
            if key[0] == parent_seq_id
        ]
        for key, entry in parent_entries:
            layer_group = key[2]
            self._retain_block(layer_group, entry.physical_block_id)
            child_key = (child_seq_id, key[1], layer_group)
            self._entries[child_key] = replace(entry, seq_id=child_seq_id)
        self._sequence_lengths[child_seq_id] = self._sequence_lengths[
            parent_seq_id]
        self._evicted_prefix_blocks[child_seq_id] = (
            self._evicted_prefix_blocks[parent_seq_id])

    def get_block_ref_count(self, layer_group: int,
                            physical_block_id: int) -> int:
        return self._block_ref_counts[layer_group][physical_block_id]

    def sequence_needs_cow(self, seq_id: int, layer_group: int) -> bool:
        """Return whether appending within the tail block needs a copy."""
        num_tokens = self._sequence_lengths[seq_id]
        if num_tokens % self.block_size == 0:
            return False
        logical_block_id = (num_tokens - 1) // self.block_size
        entry = self._entries[(seq_id, logical_block_id, layer_group)]
        return self.get_block_ref_count(
            layer_group, entry.physical_block_id) > 1

    def evict_prefix(self, seq_id: int, num_tokens: int) -> int:
        """Evict complete old token blocks and return the effective count."""
        if self._temporary_entries:
            raise RuntimeError("Cannot evict while recompute blocks are active")
        sequence_length = self._sequence_lengths[seq_id]
        num_tokens = min(max(num_tokens, 0), sequence_length)
        num_prefix_blocks = num_tokens // self.block_size
        old_prefix_blocks = self._evicted_prefix_blocks[seq_id]
        if num_prefix_blocks < old_prefix_blocks:
            raise ValueError("Growing the cached prefix is not supported")

        for logical_block_id in range(old_prefix_blocks, num_prefix_blocks):
            for layer_group in range(self.num_layer_groups):
                key = (seq_id, logical_block_id, layer_group)
                entry = self._entries.pop(key)
                self._free_block(layer_group, entry.physical_block_id)
        self._evicted_prefix_blocks[seq_id] = num_prefix_blocks
        return num_prefix_blocks * self.block_size

    def begin_recompute(self, seq_id: int, layer_group: int) -> List[int]:
        """Allocate temporary prefix blocks for one layer group."""
        if not 0 <= layer_group < self.num_layer_groups:
            raise ValueError("layer_group is out of range")
        prefix_blocks = self._evicted_prefix_blocks[seq_id]
        if prefix_blocks == 0:
            return self.get_retained_block_table(seq_id, layer_group)
        if any(key[0] == seq_id for key in self._temporary_entries):
            raise RuntimeError("Sequence already has active recompute blocks")

        allocated_keys: List[Tuple[int, int, int]] = []
        try:
            for logical_block_id in range(prefix_blocks):
                physical_block_id = self._allocate_block(layer_group)
                key = (seq_id, logical_block_id, layer_group)
                self._temporary_entries[key] = KVMapEntry(
                    seq_id=seq_id,
                    token_start=logical_block_id * self.block_size,
                    layer_start=layer_group * self.layer_group_size,
                    logical_block_id=logical_block_id,
                    physical_block_id=physical_block_id,
                    num_filled=self.block_size,
                    temporary=True,
                )
                allocated_keys.append(key)
        except Exception:
            for key in allocated_keys:
                entry = self._temporary_entries.pop(key)
                self._free_block(layer_group, entry.physical_block_id)
            raise
        return self.get_decode_block_table(seq_id, layer_group)

    def get_recompute_slot_mapping(self, seq_id: int,
                                   layer_group: int) -> List[int]:
        """Return paged-cache slots for the active temporary prefix."""
        prefix_blocks = self._evicted_prefix_blocks[seq_id]
        slots: List[int] = []
        for logical_block_id in range(prefix_blocks):
            key = (seq_id, logical_block_id, layer_group)
            if key not in self._temporary_entries:
                raise RuntimeError("Recompute blocks are not active")
            physical_block_id = self._temporary_entries[key].physical_block_id
            block_start = physical_block_id * self.block_size
            slots.extend(range(block_start, block_start + self.block_size))
        return slots

    def get_decode_slot(self, seq_id: int, layer_group: int) -> int:
        """Return the cache slot occupied by the sequence's newest token."""
        num_tokens = self._sequence_lengths[seq_id]
        logical_block_id = (num_tokens - 1) // self.block_size
        key = (seq_id, logical_block_id, layer_group)
        entry = self._entries[key]
        return (entry.physical_block_id * self.block_size +
                (num_tokens - 1) % self.block_size)

    def get_layer_group(self, layer_idx: int) -> int:
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is out of range")
        return layer_idx // self.layer_group_size

    def get_num_evicted_tokens(self, seq_id: int) -> int:
        return self._evicted_prefix_blocks[seq_id] * self.block_size

    def get_sequence_length(self, seq_id: int) -> int:
        return self._sequence_lengths[seq_id]

    def build_layer_group_plans(
            self, seq_ids: List[int]) -> List[LayerGroupBlockPlan]:
        """Snapshot addresses for a synchronous scheduler execution.

        The returned plans do not retain temporary blocks. They are safe when
        scheduling and model execution are serialized, so no subsequent cache
        allocation can occur before the kernels consume the plan. Direct or
        asynchronous model execution must use :meth:`lease_layer_group_plan`.
        """
        plans: List[LayerGroupBlockPlan] = []
        for layer_group in range(self.num_layer_groups):
            recompute_slots: List[int] = []
            decode_slots: List[int] = []
            decode_tables: List[List[int]] = []
            active_seq_ids: List[int] = []
            try:
                for seq_id in seq_ids:
                    decode_tables.append(
                        self.begin_recompute(seq_id, layer_group))
                    active_seq_ids.append(seq_id)
                    recompute_slots.extend(
                        self.get_recompute_slot_mapping(seq_id, layer_group))
                    decode_slots.append(
                        self.get_decode_slot(seq_id, layer_group))
                plans.append(
                    LayerGroupBlockPlan(
                        recompute_slot_mapping=recompute_slots,
                        decode_slot_mapping=decode_slots,
                        decode_block_tables=decode_tables,
                    ))
            finally:
                for seq_id in active_seq_ids:
                    self.end_recompute(seq_id, layer_group)
        return plans

    def acquire_layer_group_plan(
        self,
        seq_ids: List[int],
        layer_group: int,
    ) -> LayerGroupBlockPlan:
        """Reserve one layer group's temporary blocks and return its plan."""
        recompute_slots: List[int] = []
        decode_slots: List[int] = []
        decode_tables: List[List[int]] = []
        active_seq_ids: List[int] = []
        try:
            for seq_id in seq_ids:
                decode_tables.append(
                    self.begin_recompute(seq_id, layer_group))
                active_seq_ids.append(seq_id)
                recompute_slots.extend(
                    self.get_recompute_slot_mapping(seq_id, layer_group))
                decode_slots.append(self.get_decode_slot(seq_id, layer_group))
        except Exception:
            self.release_layer_group_plan(active_seq_ids, layer_group)
            raise
        return LayerGroupBlockPlan(
            recompute_slot_mapping=recompute_slots,
            decode_slot_mapping=decode_slots,
            decode_block_tables=decode_tables,
        )

    def release_layer_group_plan(self, seq_ids: List[int],
                                 layer_group: int) -> None:
        """Release temporary blocks reserved for a layer-group batch."""
        for seq_id in seq_ids:
            self.end_recompute(seq_id, layer_group)

    @contextmanager
    def lease_layer_group_plan(
        self,
        seq_ids: List[int],
        layer_group: int,
    ) -> Iterator[LayerGroupBlockPlan]:
        """Keep temporary blocks alive for one layer group's execution."""
        plan = self.acquire_layer_group_plan(seq_ids, layer_group)
        try:
            yield plan
        finally:
            self.release_layer_group_plan(seq_ids, layer_group)

    def end_recompute(self, seq_id: int, layer_group: int) -> None:
        keys = [
            key for key in self._temporary_entries
            if key[0] == seq_id and key[2] == layer_group
        ]
        for key in keys:
            entry = self._temporary_entries.pop(key)
            self._free_block(layer_group, entry.physical_block_id)

    def get_retained_block_table(self, seq_id: int,
                                 layer_group: int) -> List[int]:
        entries = [
            entry for key, entry in self._entries.items()
            if key[0] == seq_id and key[2] == layer_group
        ]
        entries.sort(key=lambda entry: entry.logical_block_id)
        return [entry.physical_block_id for entry in entries]

    def get_decode_block_table(self, seq_id: int,
                               layer_group: int) -> List[int]:
        num_blocks = self._num_logical_blocks(self._sequence_lengths[seq_id])
        table: List[Optional[int]] = [None] * num_blocks
        for entries in (self._entries, self._temporary_entries):
            for key, entry in entries.items():
                if key[0] == seq_id and key[2] == layer_group:
                    table[entry.logical_block_id] = entry.physical_block_id
        if any(block_id is None for block_id in table):
            raise RuntimeError("Decode block table has an uncached gap")
        return [int(block_id) for block_id in table]

    def get_map_entries(self, seq_id: Optional[int] = None) -> List[KVMapEntry]:
        entries = list(self._entries.values())
        entries.extend(self._temporary_entries.values())
        if seq_id is not None:
            entries = [entry for entry in entries if entry.seq_id == seq_id]
        return sorted(entries,
                      key=lambda entry: (entry.seq_id, entry.layer_start,
                                         entry.logical_block_id,
                                         entry.temporary))

    def get_num_free_blocks(self, layer_group: Optional[int] = None) -> int:
        if layer_group is not None:
            return len(self._free_blocks[layer_group])
        return sum(len(blocks) for blocks in self._free_blocks)

    def free_sequence(self, seq_id: int) -> None:
        for entries in (self._entries, self._temporary_entries):
            keys = [key for key in entries if key[0] == seq_id]
            for key in keys:
                entry = entries.pop(key)
                self._free_block(key[2], entry.physical_block_id)
        self._sequence_lengths.pop(seq_id, None)
        self._evicted_prefix_blocks.pop(seq_id, None)
