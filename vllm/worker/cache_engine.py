"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.core.ellm_cache import DEFAULT_LAYER_GROUP_SIZE
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    layer_group_size = DEFAULT_LAYER_GROUP_SIZE

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.flatten_layers = cache_config.flatten_layers
        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        self.swap_in_stream = torch.cuda.Stream()
        self.swap_out_stream = torch.cuda.Stream()

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        if src_to_dst.ndim == 2 and src_to_dst.shape[1] == 3:
            self._swap_layer_groups(self.cpu_cache, self.gpu_cache,
                                    src_to_dst)
            return
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        if src_to_dst.ndim == 2 and src_to_dst.shape[1] == 3:
            self._swap_layer_groups(self.gpu_cache, self.cpu_cache,
                                    src_to_dst)
            return
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def swap_in_async(
            self, src_to_dst: torch.Tensor) -> Dict[int, torch.cuda.Event]:
        """Launch swap-in and return readiness events by layer group."""
        if src_to_dst.numel() == 0:
            return {}
        events = {}
        if src_to_dst.shape[1] == 3:
            for layer_group in src_to_dst[:, 0].unique().tolist():
                group_rows = src_to_dst[src_to_dst[:, 0] == layer_group]
                with torch.cuda.stream(self.swap_in_stream):
                    self._swap_layer_groups(self.cpu_cache, self.gpu_cache,
                                            group_rows)
                    event = torch.cuda.Event()
                    event.record(self.swap_in_stream)
                events[int(layer_group)] = event
        else:
            with torch.cuda.stream(self.swap_in_stream):
                self.swap_in(src_to_dst)
                event = torch.cuda.Event()
                event.record(self.swap_in_stream)
            events[-1] = event
        return events

    def swap_out_async(
            self, src_to_dst: torch.Tensor) -> Optional[torch.cuda.Event]:
        """Launch swap-out independently from the model compute stream."""
        if src_to_dst.numel() == 0:
            return None
        with torch.cuda.stream(self.swap_out_stream):
            self.swap_out(src_to_dst)
            event = torch.cuda.Event()
            event.record(self.swap_out_stream)
        return event

    def _swap_layer_groups(self, src_cache: List[torch.Tensor],
                           dst_cache: List[torch.Tensor],
                           mappings: torch.Tensor) -> None:
        """Apply ``(layer_group, source, destination)`` block mappings."""
        for layer_group in mappings[:, 0].unique().tolist():
            group_mappings = mappings[mappings[:, 0] == layer_group, 1:]
            layer_start = layer_group * self.layer_group_size
            layer_end = min(layer_start + self.layer_group_size,
                            self.num_layers)
            for layer_idx in range(layer_start, layer_end):
                self.attn_backend.swap_blocks(src_cache[layer_idx],
                                              dst_cache[layer_idx],
                                              group_mappings)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        if src_to_dsts.ndim == 2 and src_to_dsts.shape[1] == 3:
            for layer_group in src_to_dsts[:, 0].unique().tolist():
                group_rows = src_to_dsts[src_to_dsts[:, 0] == layer_group]
                layer_start = layer_group * self.layer_group_size
                layer_end = min(layer_start + self.layer_group_size,
                                self.num_layers)
                self.attn_backend.copy_blocks(
                    self.gpu_cache[layer_start:layer_end], group_rows[:, 1:])
            return
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
