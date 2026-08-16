"""Build and lease attention metadata for eLLM partial recompute."""

from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch

from vllm.attention.backends.xformers import XFormersMetadata
from vllm.core.ellm_cache import LayerGroupBlockPlan, TokenLayerKVMap
from vllm.sequence import SequenceGroupMetadata


def _cumulative_lengths(lengths: List[int],
                        device: torch.device) -> torch.Tensor:
    result = torch.zeros(len(lengths) + 1,
                         dtype=torch.int32,
                         device=device)
    if lengths:
        lengths_tensor = torch.tensor(lengths,
                                      dtype=torch.int32,
                                      device=device)
        torch.cumsum(lengths_tensor, dim=0, out=result[1:])
    return result


def _make_block_tables(block_tables: List[List[int]],
                       device: torch.device) -> torch.Tensor:
    max_num_blocks = max((len(table) for table in block_tables), default=0)
    result = torch.zeros((len(block_tables), max_num_blocks),
                         dtype=torch.int32,
                         device=device)
    for row, table in enumerate(block_tables):
        if table:
            result[row, :len(table)] = torch.tensor(table,
                                                    dtype=torch.int32,
                                                    device=device)
    return result


def get_block_table_logical_start(
    num_tokens: int,
    block_table: List[int],
    block_size: int,
) -> int:
    """Return the first logical block represented by a compact table."""
    num_logical_blocks = (num_tokens + block_size - 1) // block_size
    logical_start = num_logical_blocks - len(block_table)
    if logical_start < 0:
        raise ValueError("block table is longer than the sequence")
    return logical_start


def combine_partial_recompute_metadata(
    recompute_metadata: XFormersMetadata,
    decode_metadata: XFormersMetadata,
) -> XFormersMetadata:
    """Combine eLLM recompute and decode metadata for one attention call.

    The flattened token order is recompute first and decode second.  Keep the
    original metadata objects as the cached views because prompt block tables
    have width zero while decode block tables contain the restored prefix.
    A single rectangular tensor cannot represent both without accidentally
    selecting xFormers' prefix-attention path for recompute tokens.
    """
    if recompute_metadata.num_decode_tokens != 0:
        raise ValueError("recompute metadata must be prefill-only")
    if decode_metadata.num_prefills != 0:
        raise ValueError("decode metadata must be decode-only")
    if recompute_metadata.slot_mapping.device != (
            decode_metadata.slot_mapping.device):
        raise ValueError("recompute and decode metadata must share a device")
    if recompute_metadata.seq_lens_tensor is None:
        raise ValueError("recompute sequence lengths are required")
    if decode_metadata.seq_lens_tensor is None:
        raise ValueError("decode sequence lengths are required")
    if decode_metadata.block_tables is None:
        raise ValueError("decode block tables are required")

    num_prefills = recompute_metadata.num_prefills
    num_decode_tokens = decode_metadata.num_decode_tokens
    max_blocks = decode_metadata.block_tables.shape[1]
    block_tables = torch.zeros(
        (num_prefills + num_decode_tokens, max_blocks),
        dtype=decode_metadata.block_tables.dtype,
        device=decode_metadata.block_tables.device,
    )
    block_tables[num_prefills:] = decode_metadata.block_tables
    seq_lens_tensor = torch.cat(
        (recompute_metadata.seq_lens_tensor,
         decode_metadata.seq_lens_tensor),
        dim=0,
    )
    combined = XFormersMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=recompute_metadata.num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        total_seq_len=(recompute_metadata.total_seq_len +
                       decode_metadata.total_seq_len),
        slot_mapping=torch.cat((recompute_metadata.slot_mapping,
                                decode_metadata.slot_mapping)),
        seq_lens=recompute_metadata.seq_lens,
        seq_lens_tensor=seq_lens_tensor,
        max_query_len=recompute_metadata.max_query_len,
        max_prefill_seq_len=recompute_metadata.max_prefill_seq_len,
        max_decode_seq_len=decode_metadata.max_decode_seq_len,
        query_start_loc=recompute_metadata.query_start_loc,
        seq_start_loc=recompute_metadata.seq_start_loc,
        context_lens_tensor=recompute_metadata.context_lens_tensor,
        block_tables=block_tables,
        use_cuda_graph=False,
    )
    combined._cached_prefill_metadata = recompute_metadata
    combined._cached_decode_metadata = decode_metadata
    return combined


def build_layerwise_partial_recompute_metadata(
    layer_group_plans: List[LayerGroupBlockPlan],
    recompute_seq_lens: List[int],
    decode_seq_lens: List[int],
    num_layers: int,
    layer_group_size: int,
    device: torch.device,
) -> Tuple[List[XFormersMetadata], List[XFormersMetadata]]:
    """Expand layer-group block plans into per-layer attention metadata."""
    if len(recompute_seq_lens) != len(decode_seq_lens):
        raise ValueError("recompute and decode batches must have equal size")
    expected_groups = (num_layers + layer_group_size - 1) // layer_group_size
    if len(layer_group_plans) != expected_groups:
        raise ValueError("layer group plans do not cover the model")

    num_seqs = len(decode_seq_lens)
    active_recompute_lens = [length for length in recompute_seq_lens if length]
    num_recompute_seqs = len(active_recompute_lens)
    num_recompute_tokens = sum(recompute_seq_lens)
    recompute_start_loc = _cumulative_lengths(active_recompute_lens, device)
    decode_seq_lens_tensor = torch.tensor(decode_seq_lens,
                                           dtype=torch.int32,
                                           device=device)
    recompute_metadata: List[XFormersMetadata] = []
    decode_metadata: List[XFormersMetadata] = []

    for layer_idx in range(num_layers):
        plan = layer_group_plans[layer_idx // layer_group_size]
        if len(plan.recompute_slot_mapping) != num_recompute_tokens:
            raise ValueError("recompute slot mapping has the wrong size")
        if len(plan.decode_slot_mapping) != num_seqs:
            raise ValueError("decode slot mapping has the wrong size")
        if len(plan.decode_block_tables) != num_seqs:
            raise ValueError("decode block tables have the wrong size")

        recompute_metadata.append(
            XFormersMetadata(
                num_prefills=num_recompute_seqs,
                num_prefill_tokens=num_recompute_tokens,
                num_decode_tokens=0,
                total_seq_len=num_recompute_tokens,
                slot_mapping=torch.tensor(plan.recompute_slot_mapping,
                                          dtype=torch.long,
                                          device=device),
                seq_lens=active_recompute_lens,
                seq_lens_tensor=torch.tensor(active_recompute_lens,
                                             dtype=torch.int32,
                                             device=device),
                max_query_len=max(active_recompute_lens, default=0),
                max_prefill_seq_len=max(active_recompute_lens, default=0),
                max_decode_seq_len=0,
                query_start_loc=recompute_start_loc,
                seq_start_loc=recompute_start_loc,
                context_lens_tensor=torch.zeros(num_recompute_seqs,
                                                dtype=torch.int32,
                                                device=device),
                block_tables=torch.empty((num_recompute_seqs, 0),
                                         dtype=torch.int32,
                                         device=device),
                use_cuda_graph=False,
            ))
        decode_metadata.append(
            XFormersMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decode_tokens=num_seqs,
                total_seq_len=sum(decode_seq_lens),
                slot_mapping=torch.tensor(plan.decode_slot_mapping,
                                          dtype=torch.long,
                                          device=device),
                seq_lens=None,
                seq_lens_tensor=decode_seq_lens_tensor,
                max_query_len=None,
                max_prefill_seq_len=0,
                max_decode_seq_len=max(decode_seq_lens, default=0),
                query_start_loc=None,
                seq_start_loc=None,
                context_lens_tensor=None,
                block_tables=_make_block_tables(plan.decode_block_tables,
                                                device),
                use_cuda_graph=False,
            ))
    return recompute_metadata, decode_metadata


class LayerGroupMetadataPlanner:
    """Lease temporary KV blocks while a model executes one layer group."""

    def __init__(
        self,
        cache_map: TokenLayerKVMap,
        seq_ids: List[int],
        recompute_seq_lens: List[int],
        decode_seq_lens: List[int],
        device: torch.device,
    ) -> None:
        if len(seq_ids) != len(recompute_seq_lens):
            raise ValueError("sequence IDs and recompute lengths must match")
        if len(seq_ids) != len(decode_seq_lens):
            raise ValueError("sequence IDs and decode lengths must match")
        self.cache_map = cache_map
        self.seq_ids = seq_ids
        self.recompute_seq_lens = recompute_seq_lens
        self.decode_seq_lens = decode_seq_lens
        self.device = device

    @contextmanager
    def metadata_for_layer_group(
        self,
        layer_group: int,
    ) -> Iterator[Tuple[XFormersMetadata, XFormersMetadata]]:
        """Yield metadata while the corresponding temporary blocks are live."""
        with self.cache_map.lease_layer_group_plan(
                self.seq_ids, layer_group) as plan:
            recompute, decode = build_layerwise_partial_recompute_metadata(
                [plan],
                self.recompute_seq_lens,
                self.decode_seq_lens,
                num_layers=1,
                layer_group_size=1,
                device=self.device,
            )
            yield recompute[0], decode[0]


def build_layerwise_attention_metadata(
    base_metadata: XFormersMetadata,
    seq_group_metadata_list: List[SequenceGroupMetadata],
    num_layers: int,
    layer_group_size: int,
    block_size: int,
    device: torch.device,
) -> List[XFormersMetadata]:
    """Replace a conventional batch's addresses with layer-group tables."""
    if layer_group_size <= 0:
        raise ValueError("layer_group_size must be positive")
    rows = []
    for seq_group_metadata in seq_group_metadata_list:
        layer_tables = seq_group_metadata.layer_group_block_tables
        if layer_tables is None:
            raise ValueError("layer-group block tables are required")
        for seq_id, seq_data in seq_group_metadata.seq_data.items():
            if seq_group_metadata.is_prompt:
                context_len = seq_data.get_num_computed_tokens()
            else:
                context_len = seq_data.get_len() - 1
            seq_len = min(
                seq_data.get_len(),
                context_len + seq_group_metadata.token_chunk_size,
            )
            rows.append((layer_tables[seq_id], context_len, seq_len,
                         seq_data.get_len(), seq_group_metadata.is_prompt))

    expected_groups = (num_layers + layer_group_size - 1) // layer_group_size
    if any(len(tables) != expected_groups
           for tables, _, _, _, _ in rows):
        raise ValueError("layer-group block tables do not cover the model")

    result = []
    for layer_idx in range(num_layers):
        layer_group = layer_idx // layer_group_size
        slots = []
        block_tables = []
        for tables, context_len, seq_len, total_len, is_prompt in rows:
            table = tables[layer_group]
            logical_start = get_block_table_logical_start(
                total_len, table, block_size)
            for position in range(context_len, seq_len):
                logical_block = position // block_size
                block_id = table[logical_block - logical_start]
                slots.append(block_id * block_size + position % block_size)
            # Non-chunked prompt attention does not read through PagedAttention.
            block_tables.append([] if is_prompt else table)

        result.append(
            XFormersMetadata(
                num_prefills=base_metadata.num_prefills,
                num_prefill_tokens=base_metadata.num_prefill_tokens,
                num_decode_tokens=base_metadata.num_decode_tokens,
                total_seq_len=base_metadata.total_seq_len,
                slot_mapping=torch.tensor(slots,
                                          dtype=torch.long,
                                          device=device),
                seq_lens=base_metadata.seq_lens,
                seq_lens_tensor=base_metadata.seq_lens_tensor,
                max_query_len=base_metadata.max_query_len,
                max_prefill_seq_len=base_metadata.max_prefill_seq_len,
                max_decode_seq_len=base_metadata.max_decode_seq_len,
                query_start_loc=base_metadata.query_start_loc,
                seq_start_loc=base_metadata.seq_start_loc,
                context_lens_tensor=base_metadata.context_lens_tensor,
                block_tables=_make_block_tables(block_tables, device),
                use_cuda_graph=False,
            ))
    return result
