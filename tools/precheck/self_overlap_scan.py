#!/usr/bin/env python3
"""Pre-submission self-overlap scan (iThenate-style proxy, local & free).

Compares the journal manuscript (paper_plain.txt, from paper/main.tex @ 8db4a5c)
against:
  A. the PRIOR PUBLISHED conference paper (ICE2CT-2026, v2x_paper_v8.tex)
     -> the author self-overlap source iThenticate actually indexes
  B. public repo docs (README, ONE_PAGER, RESEARCH_COUNCIL,
     REGULATORY_COMMENT_DRAFT, paper/COVER_LETTER)

Three independent measures per comparison:
  1. word 5-gram containment (both directions) - token-level overlap
  2. rapidfuzz sentence-level near-duplicates (>=75 near-verbatim, 60-74 suspect)
  3. winnowing document-fingerprint Jaccard (MOSS-family algorithm)
plus longest shared contiguous word run (binary-search on k-gram sets).

Output: console report + JSON evidence (scripts/toolcheck/self_overlap_results.json)
"""
import json
import re
from rapidfuzz import fuzz, process

from winnowing import winnow

PAPER = "/home/z/my-project/scripts/paper_plain.txt"
OLD = "/home/z/my-project/scripts/old_paper_plain.txt"
OUT = "/home/z/my-project/scripts/toolcheck/self_overlap_results.json"

SOURCES = {
    "ICE2CT2026_conference_paper": OLD,
    "repo_README": "/home/z/my-project/v2x_repo/README.md",
    "repo_ONE_PAGER": "/home/z/my-project/v2x_repo/ONE_PAGER.md",
    "repo_RESEARCH_COUNCIL": "/home/z/my-project/v2x_repo/RESEARCH_COUNCIL.md",
    "repo_REGULATORY_COMMENT_DRAFT": "/home/z/my-project/v2x_repo/REGULATORY_COMMENT_DRAFT.md",
    "repo_paper_COVER_LETTER": "/home/z/my-project/v2x_repo/paper/COVER_LETTER.md",
}

PLACEHOLDER = re.compile(r"\[(cite|ref|math|url|link|float)\]", re.I)


def clean(text: str) -> str:
    """Strip extractor placeholders so '[cite]' tokens don't create fake matches."""
    return PLACEHOLDER.sub(" ", text)


def words(text: str):
    return re.findall(r"[a-z0-9']+", clean(text).lower())


def ngrams(ws, n):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean(text))
            if len(s.split()) >= 8]


def norm_sent(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()


def longest_shared_run(aw, bw):
    """Largest k such that some contiguous k-word run of A appears in B.
    Monotone in k -> binary search."""
    def has(k):
        bkg = set(ngrams(bw, k))
        return any(kg in bkg for kg in ngrams(aw, k))
    lo, hi = 1, min(len(aw), len(bw))
    if not has(1):
        return 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if has(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def fingerprints(text: str):
    fp = winnow(clean(text).lower(), 5)
    out = set()
    for item in fp:
        if isinstance(item, tuple):
            out.add(item[-1])  # (pos, hash) or (hash, pos)
        else:
            out.add(item)
    return out


def compare(label: str, a_text: str, b_text: str):
    aw, bw = words(a_text), words(b_text)
    a5, b5 = set(ngrams(aw, 5)), set(ngrams(bw, 5))
    inter = a5 & b5
    res = {
        "source": label,
        "paper_words": len(aw),
        "source_words": len(bw),
        "shared_5grams": len(inter),
        "pct_of_paper_5grams": round(100 * len(inter) / max(len(a5), 1), 2),
        "pct_of_source_5grams": round(100 * len(inter) / max(len(b5), 1), 2),
        "longest_shared_word_run": longest_shared_run(aw, bw),
    }
    # winnowing jaccard
    try:
        fa, fb = fingerprints(a_text), fingerprints(b_text)
        res["winnowing_jaccard"] = round(len(fa & fb) / max(len(fa | fb), 1), 4)
    except Exception as e:  # noqa: BLE001
        res["winnowing_jaccard"] = f"error: {e}"

    # sentence-level near-duplicates (paper sentence vs source sentences)
    asents = [norm_sent(s) for s in sentences(a_text)]
    bsents = [norm_sent(s) for s in sentences(b_text)]
    near, mid = [], []
    for sa in asents:
        if not sa:
            continue
        m = process.extractOne(sa, bsents, scorer=fuzz.ratio, score_cutoff=58)
        if m is None:
            continue
        score, bj = m[1], m[2]
        rec = {"score": score, "paper_sentence": sa[:220],
               "source_sentence": bsents[bj][:220]}
        if score >= 75:
            near.append(rec)
        elif score >= 58:
            mid.append(rec)
    near.sort(key=lambda r: -r["score"])
    mid.sort(key=lambda r: -r["score"])
    res["near_verbatim_sentences_75plus"] = len(near)
    res["suspect_sentences_58_74"] = len(mid)
    res["near_verbatim_examples"] = near[:12]
    res["suspect_examples"] = mid[:8]
    return res


def main():
    with open(PAPER) as f:
        paper_text = f.read()
    results = []
    for label, path in SOURCES.items():
        try:
            with open(path, errors="replace") as f:
                src = f.read()
        except FileNotFoundError:
            print(f"  (missing, skipped: {path})")
            continue
        print(f"scanning vs {label} ...")
        results.append(compare(label, paper_text, src))

    print("\n" + "=" * 78)
    print("SELF-OVERLAP SCAN — journal manuscript vs prior/public text")
    print("=" * 78)
    hdr = (f"{'source':34s} {'%paper':>7s} {'%src':>7s} {'run':>5s} "
           f"{'wnj':>6s} {'>=75':>5s} {'58-74':>6s}")
    print(hdr)
    print("-" * 78)
    for r in results:
        wnj = r["winnowing_jaccard"]
        wnj_s = f"{wnj:.4f}" if isinstance(wnj, float) else "n/a"
        print(f"{r['source']:34s} {r['pct_of_paper_5grams']:6.2f}% "
              f"{r['pct_of_source_5grams']:6.2f}% {r['longest_shared_word_run']:5d} "
              f"{wnj_s:>6s} {r['near_verbatim_sentences_75plus']:5d} "
              f"{r['suspect_sentences_58_74']:6d}")

    # detail for the conference paper (the critical one)
    conf = next((r for r in results if "ICE2CT" in r["source"]), None)
    if conf:
        print("\n--- vs PRIOR CONFERENCE PAPER: near-verbatim (>=75) passages ---")
        for ex in conf["near_verbatim_examples"]:
            print(f"\n  [{ex['score']:.0f}] PAPER: {ex['paper_sentence'][:180]}")
            print(f"        OLD  : {ex['source_sentence'][:180]}")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
