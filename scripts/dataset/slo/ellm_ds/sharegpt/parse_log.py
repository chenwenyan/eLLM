#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from collections import defaultdict

# Matches:
#   DeepseekV2 layer 6 attention time: 2.869 ms (comm: 0.393 ms)
#   Llama     layer 6 attention time: 0.577 ms (comm: 0.163 ms)
PAT_ATTN_FFN = re.compile(
    r"(?P<model>DeepseekV2|Llama)\s+layer\s+(?P<layer>\d+)\s+"
    r"(?P<kind>attention|ffn)\s+time\s*:\s*"
    r"(?P<total>[0-9.]+)\s*ms\s*"
    r"\(comm\s*:\s*(?P<comm>[0-9.]+)\s*ms\)",
    re.IGNORECASE,
)

# Matches:
#   DeepseekV2 layer 6 total time: 4.672 ms
#   Llama     layer 6 total time: 1.325 ms
PAT_LAYER_TOTAL = re.compile(
    r"(?P<model>DeepseekV2|Llama)\s+layer\s+(?P<layer>\d+)\s+"
    r"total\s+time\s*:\s*(?P<total>[0-9.]+)\s*ms\b",
    re.IGNORECASE,
)

def new_iter_state():
    return {
        "attention": {"total": 0.0, "comm": 0.0, "exec": 0.0},
        "ffn": {"total": 0.0, "comm": 0.0, "exec": 0.0},
        "layer_total": {"total": 0.0},  # sum of "layer k total time"
        "seen": set(),  # (kind, layer) and ("layer_total", layer)
        "started": False,
    }

def flush(model_iters, model_state, model_name):
    st = model_state[model_name]
    if st["started"]:
        model_iters[model_name].append({
            "attention": st["attention"].copy(),
            "ffn": st["ffn"].copy(),
            "layer_total": st["layer_total"].copy(),
            "seen": set(st["seen"]),
        })
    model_state[model_name] = new_iter_state()

def parse_by_iteration(path: str):
    """
    Maintain independent iteration streams per model (DeepseekV2 vs Llama).
    Iteration boundary rule: encountering 'layer 0 attention' for that model.
    """
    model_iters = defaultdict(list)            # model -> list[iter_record]
    model_state = defaultdict(new_iter_state)  # model -> current iter state

    matched = {"attn_ffn": 0, "layer_total": 0}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m1 = PAT_ATTN_FFN.search(line)
            if m1:
                matched["attn_ffn"] += 1
                model = m1.group("model").capitalize()  # Deepseekv2->Deepseekv2; fix below
                # Normalize model name exactly
                model = "DeepseekV2" if model.lower() == "deepseekv2" else "Llama"

                layer = int(m1.group("layer"))
                kind = m1.group("kind").lower()
                total = float(m1.group("total"))
                comm = float(m1.group("comm"))
                exec_t = total - comm

                # boundary: layer 0 attention
                st = model_state[model]
                if kind == "attention" and layer == 0 and st["started"]:
                    flush(model_iters, model_state, model)

                st = model_state[model]
                st["started"] = True
                st[kind]["total"] += total
                st[kind]["comm"] += comm
                st[kind]["exec"] += exec_t
                st["seen"].add((kind, layer))
                continue

            m2 = PAT_LAYER_TOTAL.search(line)
            if m2:
                matched["layer_total"] += 1
                model = m2.group("model").capitalize()
                model = "DeepseekV2" if model.lower() == "deepseekv2" else "Llama"

                layer = int(m2.group("layer"))
                total = float(m2.group("total"))

                st = model_state[model]
                st["started"] = True
                st["layer_total"]["total"] += total
                st["seen"].add(("layer_total", layer))
                continue

    # flush all models
    for model in list(model_state.keys()):
        flush(model_iters, model_state, model)

    return model_iters, matched

def summarize_model(model: str, iters: list, num_layers: int | None = None, show_per_iter: bool = False):
    n = len(iters)
    if n == 0:
        print(f"\n=== {model}: no iterations found ===")
        return

    # Optionally infer layer count from seen layers (max layer + 1)
    if num_layers is None:
        max_layer = -1
        for it in iters:
            for (k, layer) in it["seen"]:
                if isinstance(layer, int):
                    max_layer = max(max_layer, layer)
        num_layers = max_layer + 1 if max_layer >= 0 else 0

    if show_per_iter:
        print(f"\n=== {model}: Per-iteration totals (sum over layers) ===")
        print("iter,attn_total,attn_comm,attn_exec,ffn_total,ffn_comm,ffn_exec,layer_total_sum,missing")
        for i, it in enumerate(iters):
            missing = []
            for kind in ("attention", "ffn"):
                for layer in range(num_layers):
                    if (kind, layer) not in it["seen"]:
                        missing.append(f"{kind}:{layer}")
            for layer in range(num_layers):
                if ("layer_total", layer) not in it["seen"]:
                    missing.append(f"layer_total:{layer}")

            print(
                f"{i},"
                f"{it['attention']['total']:.3f},{it['attention']['comm']:.3f},{it['attention']['exec']:.3f},"
                f"{it['ffn']['total']:.3f},{it['ffn']['comm']:.3f},{it['ffn']['exec']:.3f},"
                f"{it['layer_total']['total']:.3f},"
                f"{'|'.join(missing) if missing else ''}"
            )

    mean = {
        "attention": {"total": 0.0, "comm": 0.0, "exec": 0.0},
        "ffn": {"total": 0.0, "comm": 0.0, "exec": 0.0},
        "layer_total": {"total": 0.0},
    }
    for it in iters:
        for kind in ("attention", "ffn"):
            for k in ("total", "comm", "exec"):
                mean[kind][k] += it[kind][k]
        mean["layer_total"]["total"] += it["layer_total"]["total"]

    for kind in ("attention", "ffn"):
        for k in ("total", "comm", "exec"):
            mean[kind][k] /= n
    mean["layer_total"]["total"] /= n

    print(f"\n=== {model}: Model-level mean across iterations ===")
    print(f"iterations={n}  (inferred_layers={num_layers})")
    print(
        f"attention: mean_total={mean['attention']['total']:.3f} ms, "
        f"mean_comm={mean['attention']['comm']:.3f} ms, "
        f"mean_exec={mean['attention']['exec']:.3f} ms"
    )
    print(
        f"ffn:       mean_total={mean['ffn']['total']:.3f} ms, "
        f"mean_comm={mean['ffn']['comm']:.3f} ms, "
        f"mean_exec={mean['ffn']['exec']:.3f} ms"
    )
    print(
        f"layer_total(sum over layers): mean_total={mean['layer_total']['total']:.3f} ms"
    )

def main():
    ap = argparse.ArgumentParser(description="Parse DeepseekV2 + Llama layer times and compute model-level means.")
    ap.add_argument("logfile")
    ap.add_argument("--show-per-iter", action="store_true", help="Print per-iteration CSV lines")
    args = ap.parse_args()

    model_iters, matched = parse_by_iteration(args.logfile)
    print(f"[parse] matched attn/ffn lines : {matched['attn_ffn']}")
    print(f"[parse] matched total-time lines: {matched['layer_total']}")

    # Summarize each model separately
    model='Llama'
    summarize_model(model, model_iters.get(model, []), num_layers=None, show_per_iter=args.show_per_iter)

if __name__ == "__main__":
    main()
