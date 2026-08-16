"""Discrete cache-ratio and batch-size policy for eLLM."""

import math
from dataclasses import dataclass
from typing import List

from vllm.core.ellm_cache import DEFAULT_LAYER_GROUP_SIZE


@dataclass(frozen=True)
class ELLMPolicyDecision:
    """A feasible scheduling point selected by the eLLM policy."""

    batch_size: int
    cached_layers: int
    recompute_layers: int
    resident_token_layers: int
    temporary_token_layers: int
    estimated_work: float


def _round_tokens(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size) * block_size


def select_ellm_policy(
    request_lengths: List[int],
    max_batch_size: int,
    num_layers: int,
    num_gpu_blocks: int,
    block_size: int,
    layer_group_size: int = DEFAULT_LAYER_GROUP_SIZE,
) -> ELLMPolicyDecision:
    """Select a feasible batch and cache ratio at layer-group granularity.

    The cache capacity is expressed in token-layer slots. One conventional
    vLLM physical block reserves the same token range in every model layer.
    eLLM can redistribute those slots independently across token/layer pairs.
    During partial recompute, the current layer group additionally needs a
    temporary copy of the evicted prefix.
    """
    if not request_lengths:
        raise ValueError("request_lengths must not be empty")
    if any(length <= 0 for length in request_lengths):
        raise ValueError("request lengths must be positive")
    if max_batch_size <= 0 or num_layers <= 0:
        raise ValueError("batch size and number of layers must be positive")
    if num_gpu_blocks <= 0 or block_size <= 0:
        raise ValueError("cache dimensions must be positive")
    if layer_group_size <= 0:
        raise ValueError("layer_group_size must be positive")

    group_size = min(layer_group_size, num_layers)
    layer_choices = list(range(group_size, num_layers + 1, group_size))
    if layer_choices[-1] != num_layers:
        layer_choices.append(num_layers)

    capacity = num_gpu_blocks * block_size * num_layers
    best = None
    max_batch = min(max_batch_size, len(request_lengths))
    for batch_size in range(1, max_batch + 1):
        lengths = request_lengths[:batch_size]
        padded_tokens = sum(_round_tokens(length, block_size)
                            for length in lengths)
        for cached_layers in layer_choices:
            recompute_layers = num_layers - cached_layers
            resident = padded_tokens * cached_layers
            temporary = padded_tokens * min(group_size, recompute_layers)
            if resident + temporary > capacity:
                continue

            # Attention recompute grows quadratically with prefix length while
            # a cached decode layer reads the prefix once. The value is a
            # hardware-independent ranking proxy, not a latency prediction.
            work = sum(recompute_layers * length * length +
                       cached_layers * length for length in lengths)
            throughput = batch_size / max(float(work), 1.0)
            rank = (throughput, batch_size, cached_layers)
            if best is None or rank > best[0]:
                best = (rank,
                        ELLMPolicyDecision(
                            batch_size=batch_size,
                            cached_layers=cached_layers,
                            recompute_layers=recompute_layers,
                            resident_token_layers=resident,
                            temporary_token_layers=temporary,
                            estimated_work=float(work),
                        ))

    if best is None:
        raise RuntimeError("No feasible eLLM cache policy")
    return best[1]
