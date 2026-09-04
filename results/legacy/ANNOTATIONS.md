# Legacy results — provenance annotations

These JSON files are copied **verbatim** from the original repository state (2026-04 zip).
They are kept for the record. Read `../../AUDIT.md` before citing any of them.

| File | Status | Notes |
|---|---|---|
| `mobility_scenario_results.json` | **VALID** — reconciles with paper Table IX | Produced by `experiments/mobility_scenario_eval.py` (code preserved & consistent). |
| `latency_results.json` | **VALID numbers, wrong device label in paper** | CPU batch-1 mean 0.985 ms. Paper text wrongly said "NVIDIA GPU". |
| `intermediate.json` | **STALE (v1 universe)** | 35–48 % accuracies from the removed v1 `code/` pipeline. Not comparable to the paper. |
| `defense_results.json` | **STALE (v1 universe)** | Same v1-era source. Not comparable to the paper. |
| `autoattack_official_results.json` | **INVALID — DO NOT CITE** | 58.2 % clean acc; produced by an unpreserved code state (JSON says `device: cuda`; shipped script hardcodes CPU). Contradicts the reproducible v3 pipeline. See AUDIT.md §2. |

Rule going forward: a result JSON is only admitted to `results/reproduction/` if it embeds
its config, seeds, library versions, and device, and was produced by code committed *before*
the run.
