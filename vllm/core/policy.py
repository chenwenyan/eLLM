from collections import deque
from typing import Deque

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
        wt_weight: float = 0.5,
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group, wt_weight=wt_weight),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
        wt_weight: float = 0.5,
    ) -> float:
        return now - seq_group.metrics.arrival_time


class DLLM(Policy):

    def get_priority(
            self, 
            now: float, 
            seq_group: SequenceGroup,
            wt_weight: float = 0.5,
    ) -> float:
        waiting_time = now - seq_group.metrics.arrival_time
        prompt_length = seq_group.get_seqs()[0].get_prompt_len()
        return (wt_weight * waiting_time) * (1-wt_weight) * prompt_length

class PolicyFactory:

    _POLICY_REGISTRY = {'fcfs': FCFS, 'dllm': DLLM}

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
