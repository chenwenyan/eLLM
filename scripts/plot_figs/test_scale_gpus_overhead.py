from dataclasses import dataclass
import time
import statistics
import numpy as np
import logging
import pytest

logger = logging.getLogger(__name__)


import numpy as np
import logging

logger = logging.getLogger(__name__)


class ScheduleClass:
    def _init_model_consts(self, tp_size: int):
        self.hidden_size = self.model_config.get_hidden_size()
        self.num_layers = self.cache_config.num_layers
        self.vocab_size = self.model_config.get_vocab_size()

        self.FLOPS = 312 * 10**12 * tp_size
        util = float(self.cache_config.gpu_memory_utilization)
        self.M_G = 80 * max(util - 0.1, 0.0) * tp_size

        if self.num_layers <= 40:
            self.M_W = 13 * 2
            self.SLO = 0.130
            self.epsilon = 20
        elif self.num_layers == 80:
            self.M_W = 70 * 2
            self.SLO = 0.250
            self.epsilon = 40
        else:
            self.M_W = max(1, self.num_layers) * 2
            self.SLO = 0.250
            self.epsilon = 40

    # ---------------- workspace ----------------
    def _ensure_ws(self, R: int):
        ws = getattr(self, "_ws", None)
        if ws is None or ws["R"] != R:
            ws = {
                "R": R,
                # numeric buffers (float64 keeps your current semantics safest)
                "s": np.empty(R, dtype=np.float64),
                "s2": np.empty(R, dtype=np.float64),
                "tmp": np.empty(R, dtype=np.float64),
                "tmp2": np.empty(R, dtype=np.float64),
                "psA": np.empty(R, dtype=np.float64),
                "psB": np.empty(R, dtype=np.float64),
                "psC": np.empty(R, dtype=np.float64),
                "psS": np.empty(R, dtype=np.float64),
                "l_low": np.empty(R, dtype=np.float64),
                "l_high": np.empty(R, dtype=np.float64),
                "l_int": np.empty(R, dtype=np.float64),
                "obj": np.empty(R, dtype=np.float64),
                "b_arr": np.arange(1, R + 1, dtype=np.float64),

                # bool buffers (avoid allocating masks every call)
                "feasible": np.empty(R, dtype=bool),
                "m1": np.empty(R, dtype=bool),
                "m2": np.empty(R, dtype=bool),
                "m3": np.empty(R, dtype=bool),
                "not_feasible": np.empty(R, dtype=bool),
            }
            self._ws = ws
        return ws

    def get_opt_bs_and_layers(self, tp_size: int | None = None):
        """
        Further-optimized vectorized version:
        - reuse workspace to avoid per-call allocations
        - replace np.where(...) with in-place ufuncs (out/where)
        - avoid allocating masking arrays on every call
        Semantics matches your current vectorized implementation.
        """
        if tp_size is None:
            tp_size = int(self.parallel_config.tensor_parallel_size)

        self._init_model_consts(tp_size)

        h = float(self.hidden_size)
        L = int(self.num_layers)
        V = float(self.vocab_size)

        # ---- input s ----
        R = len(self.request_lengths)
        if R <= 0:
            raise ValueError("request_lengths is empty")

        ws = self._ensure_ws(R)
        s = ws["s"]
        # copy into float64 buffer once per call (no new allocation)
        s[:] = np.asarray(self.request_lengths, dtype=np.float64)
        np.multiply(s, s, out=ws["s2"])

        FLOPS = float(self.FLOPS)
        SLO = float(self.SLO)
        M_W = float(self.M_W)
        M_G = float(self.M_G)

        if M_G < M_W:
            return [1], [min(L, 4)]

        # ---- build psA/psB/psC/psS without allocating A/B/C ----
        h2 = h * h
        s2 = ws["s2"]
        tmp = ws["tmp"]
        tmp2 = ws["tmp2"]

        # psS = cumsum(s)
        np.cumsum(s, out=ws["psS"])

        # A = 24*s*h^2 + 4*s^2*h
        np.multiply(s, (24.0 * h2), out=tmp)
        np.multiply(s2, (4.0 * h), out=tmp2)
        np.add(tmp, tmp2, out=tmp)
        np.cumsum(tmp, out=ws["psA"])

        # B = (24*h^2 + 4*h) - 20*s*h^2 - 4*s^2*h
        tmp.fill(24.0 * h2 + 4.0 * h)
        np.multiply(s, (20.0 * h2), out=tmp2)
        np.subtract(tmp, tmp2, out=tmp)
        np.multiply(s2, (4.0 * h), out=tmp2)
        np.subtract(tmp, tmp2, out=tmp)
        np.cumsum(tmp, out=ws["psB"])

        # C = 2*s*h*V + 2*h*V + 2*epsilon
        np.multiply(s, (2.0 * h * V), out=tmp)
        tmp += (2.0 * h * V + 2.0 * float(self.epsilon))
        np.cumsum(tmp, out=ws["psC"])

        psA = ws["psA"]
        psB = ws["psB"]
        psC = ws["psC"]
        psS = ws["psS"]
        b_arr = ws["b_arr"]

        # ---- constraints ----
        l_low = ws["l_low"]
        l_high = ws["l_high"]
        feasible = ws["feasible"]
        m1, m2, m3 = ws["m1"], ws["m2"], ws["m3"]

        # init
        l_low.fill(0.0)
        feasible.fill(True)

        # memory bound: l_high = min(L, (M_G-M_W)/(2*h*psS))
        denom_mem = 2.0 * h * psS
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide((M_G - M_W), denom_mem, out=l_high)
        np.minimum(l_high, float(L), out=l_high)

        # rhs = SLO*FLOPS - (psA*L + psC)
        # reuse tmp as rhs (no allocation)
        tmp[:] = psA
        tmp *= float(L)
        tmp += psC
        tmp *= -1.0
        tmp += (SLO * FLOPS)
        rhs = tmp  # view

        epsB = 1e-12

        # masks
        # m1: B_small
        np.abs(psB, out=tmp2)
        np.less(tmp2, epsB, out=m1)

        # m2: B_pos
        np.greater(psB, epsB, out=m2)

        # m3: B_neg
        np.less(psB, -epsB, out=m3)

        # feasible &= ~(B_small & (rhs < 0))
        np.less(rhs, 0.0, out=tmp2.astype(bool, copy=False))  # tmp2 is float; avoid this path
        # safer: compute (rhs < 0) into m2 temporarily, then restore B_pos later
        # We'll use ws["m2"] temporarily and recompute B_pos after.
        np.less(rhs, 0.0, out=m2)          # m2 = (rhs < 0)
        np.logical_and(m1, m2, out=m2)     # m2 = B_small & (rhs < 0)
        np.logical_not(m2, out=m2)         # m2 = ~(...) 
        np.logical_and(feasible, m2, out=feasible)
        # recompute B_pos into m2 (restore)
        np.greater(psB, epsB, out=m2)

        # if B > 0: l_high = min(l_high, rhs/B)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.divide(rhs, psB, out=tmp2)  # tmp2 = rhs/psB
        # only apply where B_pos
        np.minimum(l_high, tmp2, out=l_high, where=m2)

        # if B < 0: l_low = max(l_low, rhs/B)
        # (rhs/psB is ok; when psB<0 it gives the correct lower bound)
        # reuse tmp2 already has rhs/psB; but it was computed for all b, so fine
        np.maximum(l_low, tmp2, out=l_low, where=m3)

        # clamp bounds
        np.maximum(l_low, 0.0, out=l_low)
        np.minimum(l_high, float(L), out=l_high)

        # feasible &= (l_low <= l_high)
        np.less_equal(l_low, l_high, out=m1)  # reuse m1 as bound_ok
        np.logical_and(feasible, m1, out=feasible)

        # ---- rounding to multiple=4 ----
        multiple = 4.0
        l_int = ws["l_int"]

        # l_int = floor(l_high/4)*4 when B_neg else ceil(l_low/4)*4
        # compute both into tmp/tmp2 then select in-place to avoid np.where alloc
        np.floor(l_high / multiple, out=tmp2)   # tmp2 = floor(l_high/4)
        tmp2 *= multiple                        # tmp2 = floor*4

        np.ceil(l_low / multiple, out=tmp)      # tmp = ceil(l_low/4)
        tmp *= multiple                         # tmp = ceil*4

        # select: start with ceil(l_low), then overwrite where B_neg
        l_int[:] = tmp
        np.copyto(l_int, tmp2, where=m3)

        np.maximum(l_int, 4.0, out=l_int)

        # extra fix for B_neg: if chosen l_int < l_low, try ceil(l_low)
        # violate = B_neg & (l_int < l_low - 1e-9)
        np.less(l_int, (l_low - 1e-9), out=m1)
        np.logical_and(m3, m1, out=m1)  # m1 = violate

        if np.any(m1):
            # l_fix = ceil(l_low/4)*4 (already in tmp)
            # ok_fix = violate & (l_fix <= l_high + 1e-9)
            np.less_equal(tmp, (l_high + 1e-9), out=m2)
            np.logical_and(m1, m2, out=m2)  # m2 = ok_fix

            # apply fix where ok_fix
            np.copyto(l_int, tmp, where=m2)

            # infeasible &= ~(violate & ~ok_fix)
            np.logical_not(m2, out=m2)          # m2 = ~ok_fix
            np.logical_and(m1, m2, out=m2)      # m2 = violate & ~ok_fix
            np.logical_not(m2, out=m2)          # m2 = ~(violate & ~ok_fix)
            np.logical_and(feasible, m2, out=feasible)

        np.clip(l_int, 0.0, float(L), out=l_int)

        # ---- objective ----
        obj = ws["obj"]
        # obj = (psA*L + psB*l_int + psC) / (b_arr*FLOPS)
        obj[:] = psA
        obj *= float(L)
        obj += psB * l_int
        obj += psC
        obj /= (b_arr * FLOPS)

        # mask infeasible in-place without allocating (~feasible)
        np.logical_not(feasible, out=ws["not_feasible"])
        obj[ws["not_feasible"]] = np.inf

        best_idx = int(np.argmin(obj))
        if not np.isfinite(obj[best_idx]):
            return [1], [min(L, 4)]

        return [best_idx + 1], [int(l_int[best_idx])]


# ---------------- Dummy configs ----------------

@dataclass
class DummyModelConfig:
    hidden_size: int = 8192
    vocab_size: int = 32000

    def get_hidden_size(self) -> int:
        return self.hidden_size

    def get_vocab_size(self) -> int:
        return self.vocab_size


@dataclass
class DummyCacheConfig:
    num_layers: int = 80
    gpu_memory_utilization: float = 0.90


@dataclass
class DummyParallelConfig:
    tensor_parallel_size: int = 1


def build_request_lengths(base_pattern, R: int):
    """固定 pattern，重复填满到长度 R。"""
    base_pattern = list(base_pattern)
    if len(base_pattern) == 0:
        raise ValueError("base_pattern is empty")
    return (base_pattern * ((R + len(base_pattern) - 1) // len(base_pattern)))[:R]


def setup_llama2_70b_case(obj, tp_size: int, R: int, base_pattern, gpu_mem_util: float = 0.90):
    obj.model_config = DummyModelConfig(hidden_size=8192, vocab_size=32000)
    obj.cache_config = DummyCacheConfig(num_layers=80, gpu_memory_utilization=gpu_mem_util)
    obj.parallel_config = DummyParallelConfig(tensor_parallel_size=tp_size)
    obj.request_lengths = build_request_lengths(base_pattern, R)
    return obj


def mean_runtime_ms(fn, repeat=25, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.mean(ts), min(ts), max(ts)


def benchmark_tp_and_R(
    obj_factory,
    tp_list=(4, 8, 16, 32),
    R_list=(512, 1024, 2048, 4096),
    base_pattern=(64, 128, 256, 512, 1024, 2048),
    gpu_mem_util=0.90,
    repeat=25,
    warmup=5,
):
    if len(tp_list) != len(R_list):
        raise ValueError(f"tp_list length {len(tp_list)} != R_list length {len(R_list)}")

    results = {}
    t_all0 = time.perf_counter()

    for tp, R in zip(tp_list, R_list):
        obj = obj_factory()
        setup_llama2_70b_case(obj, tp_size=tp, R=int(R), base_pattern=base_pattern, gpu_mem_util=gpu_mem_util)

        mean, tmin, tmax = mean_runtime_ms(
            lambda: obj.get_opt_bs_and_layers(tp_size=tp),
            repeat=repeat,
            warmup=warmup,
        )
        results[tp] = {"R": int(R), "mean_ms": mean, "min_ms": tmin, "max_ms": tmax}
        print(f"TP={tp:<2d} R={R:<4d} | mean={mean:.3f} ms, min={tmin:.3f} ms, max={tmax:.3f} ms")

    total_ms = (time.perf_counter() - t_all0) * 1000.0
    print(f"\nTOTAL benchmark time: {total_ms:.3f} ms")
    return results, total_ms


# ---------------- pytest test ----------------

@pytest.mark.parametrize(
    "tp_size, R",
    [(4, 512), (8, 1024), (16, 2048), (32, 4096)],
)
def test_llama2_70b_runtime(tp_size, R):
    obj = ScheduleClass()
    base_pattern = [64, 128, 256, 512, 1024, 2048]
    setup_llama2_70b_case(obj, tp_size=tp_size, R=R, base_pattern=base_pattern, gpu_mem_util=0.90)

    mean, tmin, tmax = mean_runtime_ms(lambda: obj.get_opt_bs_and_layers(tp_size=tp_size))
    print(f"TP={tp_size} R={R}: mean={mean:.3f} ms, min={tmin:.3f} ms, max={tmax:.3f} ms")

    assert mean < 500.0  # R 变大后，给更宽松阈值，避免机器差异导致 flaky


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


if __name__ == "__main__":
    _setup_logging()

    tp_list = (4, 8, 16, 32)
    R_list = (512, 1024, 2048, 4096)
    base_pattern = (64, 128, 256, 512, 1024, 2048)

    benchmark_tp_and_R(
        obj_factory=ScheduleClass,
        tp_list=tp_list,
        R_list=R_list,
        base_pattern=base_pattern,
        repeat=25,
        warmup=5,
    )
