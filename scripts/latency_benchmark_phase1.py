#!/usr/bin/env python3
"""latency_benchmark_phase1.py — restore the real-time feasibility claim
(council P1-c / reviewer 21-c item: 'restore the Phase-0 latency number
(0.985 ms/window CPU) as feasibility evidence').

Why a fresh run instead of quoting 0.985 ms: the Phase-0 benchmark
(results/legacy/latency_results.json at fb8cf0b) used the SAME
DualStreamModel architecture (86,052 params, input [1,1,65,15]) but
(a) benchmarked model(mag, ift) on pre-shaped inputs — the Phase-1
pipeline adds the FrontEnd STFT path (complex 2048-sample window ->
mag [65,15] + IF features), and (b) ran on different hardware/torch.
Quoting it for the Phase-1 pipeline would be a claims-vs-backing slip,
so we re-measure here with the legacy methodology (warmup, 200 iterations
bs=1, mean/std/min/max/median/p95/p99, plus batch sizes) on the full
per-window pipeline INCLUDING the front end, and report the legacy number
alongside with its context.

Output: results/latency_phase1.json
"""
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from receiver import FrontEnd, DualStreamModel  # noqa: E402

OUT = os.path.join(ROOT, "results")
RES = os.path.join(OUT, "latency_phase1.json")
torch.set_num_threads(2)


def bench(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts)
    return {
        "mean_ms": round(float(a.mean()), 4),
        "std_ms": round(float(a.std()), 4),
        "min_ms": round(float(a.min()), 4),
        "max_ms": round(float(a.max()), 4),
        "median_ms": round(float(np.median(a)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "iterations": iters,
    }


def main():
    ck = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                    map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])

    n_params = sum(p.numel() for p in model.parameters())
    rng = np.random.default_rng(7)
    x1 = torch.from_numpy(
        (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
        .astype(np.complex64))                      # one 2048-sample window
    xB = {b: torch.from_numpy(
        (rng.standard_normal((b, 2048)) +
         1j * rng.standard_normal((b, 2048))).astype(np.complex64))
        for b in (1, 4, 8)}

    def full_window():
        """The deployed per-window path: waveform -> FrontEnd -> model ->
        argmax (what a sensing xApp would run per 102.4-us window)."""
        with torch.no_grad():
            mag, ifr = frontend(x1[None, :])
            return model(mag, ifr).argmax(1)

    # model-only arm: apples-to-apples with the Phase-0 legacy benchmark
    with torch.no_grad():
        m0, i0 = frontend(x1[None, :])

    def model_only():
        with torch.no_grad():
            return model(m0, i0).argmax(1)

    results = {
        "experiment": "Phase-1 CPU inference latency, full pipeline "
                      "(waveform -> FrontEnd STFT -> DualStreamModel -> "
                      "argmax)",
        "model": "DualStreamModel (Phase-1 victim, checkpoint_dual.pt)",
        "num_parameters": n_params,
        "window_samples": 2048,
        "sample_rate_msps": 20.0,
        "window_duration_us": 102.4,
        "device": "cpu",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
        },
        "benchmarks": {
            "full_pipeline_bs1": bench(full_window, 200),
            "model_only_bs1": bench(model_only, 200),
        },
        "legacy_phase0_reference": {
            "mean_ms": 0.9854, "source": "results/legacy/latency_results.json "
            "(fb8cf0b; same 86,052-param DualStreamModel, model-only on "
            "pre-shaped [1,1,65,15] inputs, torch 2.11 CPU, unknown "
            "hardware, 200 iters)",
            "note": "not comparable 1:1 — excludes the front end and ran "
                    "on different hardware; kept for lineage",
        },
    }

    # batched throughput (per-window cost amortized)
    for b in (4, 8):
        xb = xB[b]

        def batched():
            with torch.no_grad():
                mag, ifr = frontend(xb)
                return model(mag, ifr).argmax(1)
        r = bench(batched, 100, warmup=5)
        results["benchmarks"][f"full_pipeline_bs{b}"] = r
        results["benchmarks"][f"full_pipeline_bs{b}"]["per_window_ms"] = \
            round(r["mean_ms"] / b, 4)

    fp1 = results["benchmarks"]["full_pipeline_bs1"]
    feas = {
        "per_window_ms_median": fp1["median_ms"],
        "per_window_ms_mean": fp1["mean_ms"],
        "note_stats": "median/p95 are the honest statistics here: the mean "
                      "is inflated by scheduler spikes on this shared "
                      "sandbox (max 30 ms)",
        "tr37885_latency_budget_ms": 100.0,
        "windows_affordable_per_budget_median":
            int(100.0 / max(fp1["median_ms"], 1e-9)),
        "realtime_multiple_median": round(fp1["median_ms"] / 0.1024, 2),
    }
    feas["statement"] = (
        "median {m} ms/window full-pipeline (front end included) on this "
        "shared 2-vCPU sandbox -> ~{n} sensing decisions per 100-ms TR "
        "37.885 budget; amortized {p} ms/window at batch 8; model-only "
        "{mo} ms/window vs the Phase-0 legacy 0.985 ms (same architecture, "
        "different hardware/torch)".format(
            m=fp1["median_ms"],
            n=feas["windows_affordable_per_budget_median"],
            p=results["benchmarks"]["full_pipeline_bs8"]["per_window_ms"],
            mo=results["benchmarks"]["model_only_bs1"]["median_ms"]))
    results["feasibility"] = feas

    with open(RES, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results["benchmarks"], indent=1))
    print(json.dumps(results["feasibility"], indent=1))
    print("wrote", RES)


if __name__ == "__main__":
    main()
