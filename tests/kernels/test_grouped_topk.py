import pytest
import torch

from vllm.model_executor.layers.fused_moe import grouped_topk


def test_grouped_topk_limits_experts_to_selected_groups():
    hidden_states = torch.zeros(2, 4)
    router_logits = torch.tensor([
        [9.0, 8.0, 1.0, 0.0, 7.0, 6.0, 2.0, 3.0],
        [0.0, 1.0, 9.0, 8.0, 2.0, 3.0, 7.0, 6.0],
    ])

    weights, expert_ids = grouped_topk(hidden_states,
                                       router_logits,
                                       topk=2,
                                       renormalize=True,
                                       num_expert_group=4,
                                       topk_group=1)

    assert set(expert_ids[0].tolist()) == {0, 1}
    assert set(expert_ids[1].tolist()) == {2, 3}
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert expert_ids.dtype == torch.int32
    assert weights.dtype == torch.float32


def test_grouped_topk_applies_routed_scaling_after_normalization():
    hidden_states = torch.zeros(1, 2)
    router_logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    weights, _ = grouped_topk(hidden_states,
                              router_logits,
                              topk=2,
                              renormalize=True,
                              num_expert_group=2,
                              topk_group=2,
                              routed_scaling_factor=2.5)

    torch.testing.assert_close(weights.sum(), torch.tensor(2.5))


@pytest.mark.parametrize("scoring_func", ["unknown", "relu"])
def test_grouped_topk_rejects_unknown_scoring_function(scoring_func):
    with pytest.raises(ValueError, match="Unsupported scoring function"):
        grouped_topk(torch.zeros(1, 2),
                     torch.zeros(1, 4),
                     topk=1,
                     renormalize=False,
                     num_expert_group=2,
                     topk_group=1,
                     scoring_func=scoring_func)
