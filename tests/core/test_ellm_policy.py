import pytest

from vllm.core.ellm_policy import select_ellm_policy


def test_policy_obeys_layer_group_and_peak_memory_constraints():
    decision = select_ellm_policy(
        request_lengths=[16, 16, 32],
        max_batch_size=3,
        num_layers=12,
        num_gpu_blocks=8,
        block_size=4,
        layer_group_size=4,
    )

    capacity = 8 * 4 * 12
    assert decision.batch_size <= 3
    assert decision.cached_layers % 4 == 0
    assert decision.cached_layers + decision.recompute_layers == 12
    assert (decision.resident_token_layers +
            decision.temporary_token_layers <= capacity)


def test_policy_reduces_cache_ratio_under_memory_pressure():
    roomy = select_ellm_policy([16], 1, 12, 16, 4, 4)
    constrained = select_ellm_policy([16], 1, 12, 3, 4, 4)

    assert roomy.cached_layers == 12
    assert constrained.cached_layers < roomy.cached_layers
    assert constrained.recompute_layers > 0


def test_policy_rejects_invalid_or_infeasible_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        select_ellm_policy([], 1, 8, 8, 4)
    with pytest.raises(RuntimeError, match="No feasible"):
        select_ellm_policy([128], 1, 8, 1, 4)
