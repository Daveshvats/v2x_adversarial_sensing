# tools/precheck — Pre-Submission Originality Pipeline

Local, free, private pre-submission checks for `paper/main.tex`, built and
executed 2026-09-09. Full narrative: `download/PLAGIARISM_CHECK_REPORT.md`
and the formal report `download/Pre_Submission_Originality_Report.docx`
in the release workspace.

## What it checks

| Script | Check | Latest result (2026-09-09) |
|---|---|---|
| `self_overlap_scan.py` | Manuscript vs prior ICE2CT-2026 paper + 5 public repo docs: 5-gram containment, rapidfuzz near-verbatim sentences, winnowing fingerprints, longest shared run | ICE2CT paper: 0.05% containment, 0 sentences >= 75, longest run 6 words; repo docs 0.15-1.82% |
| `citation_verify.py` | All `thebibliography` entries vs Crossref API (title fuzzy-match >= 85, year within 1) | 12/12 scholarly VERIFIED at 100%; 4 grey-lit/own-paper resolved live; 6 standards out-of-scope; 0 hallucinated |
| `related_work_search.py` | Crossref 2024+ topical watchlist (6 queries, cited, deduped) | 24 records; re-surfaces the paper's own citations |
| `style_patch_w15.py` | Idempotent style fixes in main.tex | 7 edits (4x centre->center, 2x correctly-classified, 1x adversarially-trained) |
| `make_precheck_chart.py` | Figure 1 source (matplotlib, 200 dpi) | precheck_chart.png |
| `extract_tex_generic.py` | Parameterized LaTeX -> plain text (used for the old paper) | 5,846 words from v2x_paper_v8.tex |

Dependencies: `pip install rapidfuzz winnowing habanero proselint textstat`

## How to re-run (in the original workspace)

```bash
python3 /home/z/my-project/scripts/extract_tex_text.py     # refresh paper_plain.txt
python3 /home/z/my-project/scripts/self_overlap_scan.py    # similarity battery
python3 /home/z/my-project/scripts/citation_verify.py      # reference verification
python3 /home/z/my-project/scripts/related_work_search.py  # literature watchlist
```

Note: scripts reference absolute workspace paths (`/home/z/my-project/...`);
adjust `PAPER` / `SOURCES` / `TEX` constants at the top when running
elsewhere. The old conference paper text is extracted from the archived
`v2x_zip_extracted/.../v2x_paper_v8.tex` (not part of this repo).

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
- For the publisher-grade check as an individual (no institutional Turnitin):
  iThenticate sells single credits (~$125) — see the report, section 6.
