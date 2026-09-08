# tools/precheck — Pre-Submission Originality Pipeline

Local, free, private pre-submission checks for `paper/main.tex`. Everything
runs **repo-relative** (no absolute paths): run the scripts from this
directory and they resolve the manuscript, the legacy ICE2CT-2026 manuscript
(`paper/legacy/`), and the public repo docs themselves.

Full narrative: `download/PLAGIARISM_CHECK_REPORT.md` and the formal report
`download/Pre_Submission_Originality_Report.docx` in the release workspace.

## What it checks

| Script | Check | Latest result (2026-09-09, post record-correction) |
|---|---|---|
| `self_overlap_scan.py` | Manuscript vs the **unpublished** ICE2CT-2026 manuscript (`paper/legacy/v2x_paper_v8.tex`) + 5 public repo docs: 5-gram containment, rapidfuzz near-verbatim sentences, winnowing fingerprints, longest shared run | ICE2CT manuscript: 0.10% containment, 0 sentences >= 75, longest run 10 words; repo docs 0.15–1.82% |
| `citation_verify.py` | All `thebibliography` entries vs Crossref API (title fuzzy-match >= 85, year within 1) | 12/12 scholarly VERIFIED at 100%; 1 own-unpublished-manuscript (SELF_UNPUBLISHED, correctly absent from Crossref); 3 grey-lit (resolved by prior live-web pass); 6 standards out-of-scope; 0 hallucinated |
| `related_work_search.py` | Crossref 2024+ topical watchlist (6 queries, cited, deduped) | 24 records; re-surfaces the paper's own citations |
| `style_patch_w15.py` | Idempotent style fixes in main.tex | 7 edits (4x centre->center, 2x correctly-classified, 1x adversarially-trained) |
| `make_precheck_chart.py` | Chart of the self-overlap containment by source (data-driven, reads `results/self_overlap_results.json`) | `precheck_chart.png` |
| `extract_tex_generic.py` | Parameterized LaTeX -> plain text (used for both manuscripts) | used automatically by the scan |

Dependencies: `pip install rapidfuzz winnowing habanero requests matplotlib`

## How to re-run (from this directory)

```bash
python3 self_overlap_scan.py     # auto-extracts plain text, runs the battery
python3 citation_verify.py       # Crossref + OpenAlex reference verification
python3 related_work_search.py   # literature watchlist
python3 make_precheck_chart.py   # chart from the latest results JSON
```

Runtime extracts land in `work/` (git-ignored); evidence lands in `results/`.

## Results evidence

`results/` holds the JSON outputs backing every number above:
- `self_overlap_results.json` — per-source containment, runs, sentence matches
- `citation_verify_results.json` — per-reference verdicts with DOIs and scores
- `related_work_watchlist.json` — 24 Crossref records

## Interpretation guardrails

- 5-gram containment is a lower-bound proxy for iThenticate-style similarity,
  not the Turnitin number itself.
- Winnowing Jaccard uses character-level fingerprints and is dominated by
  common English letter sequences; use containment + sentence metrics for
  decisions.
- The prior ICE2CT-2026 manuscript was **never published**, so no indexing
  service holds it; the scan keeps the overlap honest for the day any version
  of it appears online (e.g., a preprint server).
- For the publisher-grade check as an individual (no institutional Turnitin):
  iThenticate sells single credits (~$125) — see the report, section 6.
