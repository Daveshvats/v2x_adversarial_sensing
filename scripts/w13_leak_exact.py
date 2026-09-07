#!/usr/bin/env python3
"""Exact realized-waveform overlap diagnostic for the C4 leak (one-off).

Replays the OLD sweep's build_set (seed 42, interleaved channel/noise draws)
hashing every realized baseband waveform, and hashes the training corpus's
baseband (gen_dataset(1000, seed 42), pure consecutive draws). Reports the
exact collision count of realized waveforms, plus the same for the clean
seed 777. Updates results/snr_sweep_clean.json:leakage_diagnostic with
'old_eval_realized_overlap_exact' and 'new_eval_realized_overlap_exact'.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from waveforms import gen_signal, gen_dataset  # noqa: E402
import channels as CH  # noqa: E402

OUT = os.path.join(ROOT, "results")
N_PER_CLASS = 250


def hashify(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def old_sweep_hashes(seed):
    """Exact replay of run_snr_sweep.build_set baseband draws (seed 42 =
    the old leaky arm): gen_signal interleaved with channel/noise draws."""
    rng = np.random.default_rng(seed)
    hs = []
    for cls in range(4):
        for _ in range(N_PER_CLASS):
            s = gen_signal(cls, rng)
            hs.append(hashify(s))
            h = CH.draw_channel("urban", rng)          # consumes rng
            snr = rng.uniform(-10.0, 0.0)              # consumes rng
            rng.standard_normal(2048)
            rng.standard_normal(2048)                  # noise draws
    return set(hs)


def main():
    train = set()
    Xtr, _ = gen_dataset(n_per_class=1000, seed=42)
    for i in range(Xtr.shape[0]):
        train.add(hashify(Xtr[i]))

    old = old_sweep_hashes(42)
    new = old_sweep_hashes(777)
    exact = {
        "train_corpus_waveforms": len(train),
        "old_eval_seed": 42,
        "old_eval_realized_overlap_exact": len(old & train),
        "old_eval_realized_overlap_fraction":
            round(len(old & train) / len(old), 4),
        "new_eval_seed": 777,
        "new_eval_realized_overlap_exact": len(new & train),
        "new_eval_realized_overlap_fraction":
            round(len(new & train) / len(new), 4),
        "note": "realized waveforms: old sweep interleaves channel draws "
                "between gen_signal calls, so realized overlap (below) is "
                "smaller than the pure-stream bound; both measured exactly "
                "by replay + sha256 of raw baseband bytes.",
    }
    print(json.dumps(exact, indent=1))

    res_path = os.path.join(OUT, "snr_sweep_clean.json")
    doc = json.load(open(res_path))
    doc["leakage_diagnostic"].update(exact)
    with open(res_path, "w") as f:
        json.dump(doc, f, indent=2)
    print("updated", res_path)


if __name__ == "__main__":
    main()
