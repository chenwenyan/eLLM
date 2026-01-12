from dataclasses import dataclass
import time
import statistics
import numpy as np
import logging
import pytest

logger = logging.getLogger(__name__)


class ScheduleClass:
    def _init_model_consts(self, tp_size: int):
        self.hidden_size = self.model_config.get_hidden_size()
        self.num_layers = self.cache_config.num_layers
        self.vocab_size = self.model_config.get_vocab_size()

        # FLOPS / Mem scale with TP size (your original assumption)
        self.FLOPS = 312 * 10**12 * tp_size
        util = float(self.cache_config.gpu_memory_utilization)
        self.M_G = 80 * max(util - 0.1, 0.0) * tp_size  # avoid negative

        if self.num_layers <= 40:  # llama2-13B
            self.M_W = 13 * 2
            self.SLO = 0.130
            self.epsilon = 20
        elif self.num_layers == 80:  # llama2-70B
            self.M_W = 70 * 2
            self.SLO = 0.250
            self.epsilon = 40
        else:
            self.M_W = max(1, self.num_layers) * 2
            self.SLO = 0.250
            self.epsilon = 40

    @staticmethod
    def _ceil_to_multiple(x: float, base: int) -> int:
        return int(np.ceil(x / base) * base)

    @staticmethod
    def _floor_to_multiple(x: float, base: int) -> int:
        return int(np.floor(x / base) * base)

    def get_opt_bs_and_layers(self, tp_size: int | None = None):
        """
        Return: ([final_b], [final_l])  (same format as your original function)
        """
        if tp_size is None:
            tp_size = int(self.parallel_config.tensor_parallel_size)

        self._init_model_consts(tp_size)

        h = float(self.hidden_size)
        L = int(self.num_layers)
        V = float(self.vocab_size)

        s = np.asarray(self.request_lengths, dtype=np.float64)
        R = int(s.shape[0])
        if R <= 0:
            raise ValueError("request_lengths is empty")

        st = time.perf_counter()

        h2 = h * h
        A = 24.0 * s * h2 + 4.0 * (s * s) * h
        B = (24.0 * h2 + 4.0 * h) - 20.0 * s * h2 - 4.0 * (s * s) * h
        C = 2.0 * s * h * V + 2.0 * h * V + 2.0 * float(self.epsilon)

        psA = np.cumsum(A)
        psB = np.cumsum(B)
        psC = np.cumsum(C)
        psS = np.cumsum(s)

        FLOPS = float(self.FLOPS)
        SLO = float(self.SLO)
        M_W = float(self.M_W)
        M_G = float(self.M_G)

        best_obj = np.inf
        best_b = None
        best_l = None

        if M_G < M_W:
            et = time.perf_counter()
            logger.info(
                f"Infeasible: M_G({M_G}) < M_W({M_W}). Solve time {(et-st)*1000:.3f} ms"
            )
            return [1], [min(L, 4)]

        for b in range(1, R + 1):
            sumA = float(psA[b - 1])
            sumB = float(psB[b - 1])
            sumC = float(psC[b - 1])
            sumS = float(psS[b - 1])

            # Memory constraint: 2*l*sumS*h + M_W <= M_G
            if sumS <= 0:
                l_max_mem = float("inf")
            else:
                l_max_mem = (M_G - M_W) / (2.0 * h * sumS)
                if l_max_mem < 0:
                    continue

            # SLO constraint: (sumA*L + sumB*l + sumC)/FLOPS <= SLO
            rhs = SLO * FLOPS - (sumA * L + sumC)

            l_low = 0.0
            l_high = float(L)
            l_high = min(l_high, l_max_mem)

            if abs(sumB) < 1e-12:
                if rhs < 0:
                    continue
            elif sumB > 0:
                l_max_slo = rhs / sumB
                if l_max_slo < 0:
                    continue
                l_high = min(l_high, l_max_slo)
            else:
                l_min_slo = rhs / sumB
                l_low = max(l_low, l_min_slo)

            if l_low > l_high:
                continue

            # objective linear in l
            if sumB < 0:
                target_l = l_high
                l_int = self._floor_to_multiple(target_l, 4)
                l_int = max(l_int, 4)
                if l_int < l_low - 1e-9:
                    l_int2 = self._ceil_to_multiple(l_low, 4)
                    l_int2 = max(l_int2, 4)
                    if l_int2 > l_high + 1e-9:
                        continue
                    l_int = l_int2
            else:
                target_l = l_low
                l_int = self._ceil_to_multiple(target_l, 4)
                l_int = max(l_int, 4)
                if l_int > l_high + 1e-9:
                    continue

            l_int = int(min(max(l_int, 0), L))
            if l_int < 0 or l_int > L:
                continue

            obj = (sumA * L + sumB * l_int + sumC) / (b * FLOPS)
            if obj < best_obj:
                best_obj = obj
                best_b = b
                best_l = l_int

        et = time.perf_counter()
        # logger.info(f"Solve time: {(et - st)*1000:.3f} ms (R={R}, tp={tp_size})")

        if best_b is None:
            return [1], [min(L, 4)]

        return [int(best_b)], [int(best_l)]


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


def setup_llama2_70b_case(obj, tp_size: int, gpu_mem_util: float = 0.90, R: int = 4096):
    obj.model_config = DummyModelConfig(hidden_size=8192, vocab_size=32000)
    obj.cache_config = DummyCacheConfig(num_layers=80, gpu_memory_utilization=gpu_mem_util)
    obj.parallel_config = DummyParallelConfig(tensor_parallel_size=tp_size)

    base = [64, 128, 256, 512, 1024, 1536, 2048]
    obj.request_lengths = (base * ((R + len(base) - 1) // len(base)))[:R]
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


def benchmark_tp_sizes(obj_factory, tp_list=(1, 2, 4, 8), gpu_mem_util=0.90, R=4096):
    results = {}
    for tp in tp_list:
        obj = obj_factory()
        setup_llama2_70b_case(obj, tp_size=tp, gpu_mem_util=gpu_mem_util, R=R)

        mean, tmin, tmax = mean_runtime_ms(lambda: obj.get_opt_bs_and_layers(tp_size=tp))
        results[tp] = (mean, tmin, tmax)
        print(f"TP={tp}: mean={mean:.3f} ms, min={tmin:.3f} ms, max={tmax:.3f} ms")
    return results


# ---------------- pytest test ----------------

@pytest.mark.parametrize("tp_size", [8, 16, 32, 64])
def test_llama2_70b_runtime(tp_size):
    obj = ScheduleClass()  # FIX: was SolverClass
    setup_llama2_70b_case(obj, tp_size=tp_size, gpu_mem_util=0.90, R=4096)

    mean, tmin, tmax = mean_runtime_ms(lambda: obj.get_opt_bs_and_layers(tp_size=tp_size))
    print(f"TP={tp_size}: mean={mean:.3f} ms, min={tmin:.3f} ms, max={tmax:.3f} ms")

    # keep a loose upper bound to avoid flaky tests
    assert mean < 200.0


# ---------------- main entry ----------------

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

if __name__ == "__main__":
    _setup_logging()
    # Run a quick benchmark when executing as a script
    benchmark_tp_sizes(obj_factory=ScheduleClass, tp_list=(4, 8, 16, 32, 64), R=4096)
