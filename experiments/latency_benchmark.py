#!/usr/bin/env python3
"""
CPU inference latency benchmark for V2X DualStreamModel (86,052 params).
Measures single-stream and batched inference latency on CPU and optionally GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────
# Model Architecture (exact copy from autoattack_eval.py)
# ──────────────────────────────────────────────────────────────────────
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4 = out_ch // 4
        c12 = out_ch // 2
        self.branch1 = nn.Conv2d(in_ch, c4, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)))


class SingleStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.inc1 = InceptionBlock(16, 32)
        self.inc2 = InceptionBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.inc1(x)
        x = F.max_pool2d(x, 2)
        x = self.inc2(x)
        x = self.pool(x)
        return x.flatten(1)


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.mag_stream = SingleStream()
        self.if_stream = SingleStream()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        f_mag = self.mag_stream(mag)
        f_if = self.if_stream(ift)
        x = torch.cat([f_mag, f_if], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────────────────────────────
# Benchmarking utilities
# ──────────────────────────────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_latency(model, mag_input, if_input, warmup=50, iters=500, device="cpu"):
    """Run timed inference and return latency stats in milliseconds.
    
    Benchmarking methodology:
      1. Warmup: run warmup iterations to eliminate first-run overhead (JIT compilation,
         cache warming, GPU kernel compilation)
      2. CUDA sync: for GPU benchmarks, torch.cuda.synchronize() ensures all
         GPU operations complete before timing (avoids async execution artifacts)
      3. Timed runs: use time.perf_counter() (highest resolution timer available)
      4. Report percentiles: P95/P99 are critical for real-time V2X systems that
         must meet latency SLAs (e.g., <10ms for collision avoidance)
    """
    model = model.to(device).eval()
    mag_input = mag_input.to(device)
    if_input = if_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(mag_input, if_input)

    # Synchronize if CUDA
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(mag_input, if_input)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # convert to ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "iterations": iters,
    }


def main():
    print("=" * 70)
    print("V2X DualStreamModel — CPU Inference Latency Benchmark")
    print("=" * 70)

    # Create model and verify param count
    model = DualStreamModel(num_classes=4)
    num_params = count_parameters(model)
    print(f"\nModel: DualStreamModel")
    print(f"Parameters: {num_params:,}")
    assert num_params == 86052, f"Expected 86,052 params, got {num_params:,}"

    # Input shapes
    # Spectrogram dimensions: 65 freq bins × 15 time frames
    # These come from STFT parameters: n_freq = NFFT//2+1 = 65, n_frames = (1024-128)//64+1 = 15
    H, W = 65, 15

    results = {
        "model": "DualStreamModel",
        "num_parameters": num_params,
        "input_shape": [1, 1, H, W],
        "benchmarks": {},
    }

    # ── CPU benchmarks (batch sizes 1, 4, 8, 16) ──
    print("\n" + "-" * 50)
    print("CPU BENCHMARKS")
    print("-" * 50)

    # (batch_size, warmup, iters) — tuned for slow CI/server CPUs
    # (batch_size, warmup, iters) — warmup/iters tuned per batch size for practical
    # wall-clock times: smaller batches get more iterations for statistical stability
    batch_configs = [(1, 50, 500), (4, 20, 200), (8, 10, 100), (16, 10, 100)]

    for bs, warm, iters in batch_configs:
        mag_input = torch.randn(bs, 1, H, W)
        if_input = torch.randn(bs, 1, H, W)

        print(f"\n  Batch size = {bs} ({iters} iters, {warm} warmup)")
        t0 = time.time()
        stats = benchmark_latency(model, mag_input, if_input,
                                  warmup=warm, iters=iters, device="cpu")
        elapsed = time.time() - t0
        results["benchmarks"][f"cpu_bs{bs}"] = stats

        print(f"    (wall: {elapsed:.1f}s)")
        print(f"    Mean:   {stats['mean_ms']:.4f} ms")
        print(f"    Std:    {stats['std_ms']:.4f} ms")
        print(f"    Min:    {stats['min_ms']:.4f} ms")
        print(f"    Max:    {stats['max_ms']:.4f} ms")
        print(f"    Median: {stats['median_ms']:.4f} ms")
        print(f"    P95:    {stats['p95_ms']:.4f} ms")
        print(f"    P99:    {stats['p99_ms']:.4f} ms")

    # ── GPU benchmarks (if available) ──
    if torch.cuda.is_available():
        print("\n" + "-" * 50)
        print("GPU BENCHMARKS")
        print("-" * 50)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        results["gpu_name"] = gpu_name

        for bs, warm, iters in batch_configs:
            mag_input = torch.randn(bs, 1, H, W)
            if_input = torch.randn(bs, 1, H, W)

            print(f"\n  Batch size = {bs}")
            stats = benchmark_latency(model, mag_input, if_input,
                                      warmup=warm, iters=iters, device="cuda")
            results["benchmarks"][f"cuda_bs{bs}"] = stats

            print(f"    Mean:   {stats['mean_ms']:.4f} ms")
            print(f"    Std:    {stats['std_ms']:.4f} ms")
            print(f"    Min:    {stats['min_ms']:.4f} ms")
            print(f"    Max:    {stats['max_ms']:.4f} ms")
            print(f"    Median: {stats['median_ms']:.4f} ms")
            print(f"    P95:    {stats['p95_ms']:.4f} ms")
            print(f"    P99:    {stats['p99_ms']:.4f} ms")
    else:
        print("\n  CUDA not available — skipping GPU benchmarks.")
        results["gpu_name"] = None

    # ── Save results ──
    output_path = Path("/home/z/my-project/download/latency_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
