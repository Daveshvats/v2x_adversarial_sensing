#!/usr/bin/env python3
"""Related-work watchlist via Crossref (relevance-ranked) + arXiv API.

Crossref citation-sort proved noisy (mega-cited adjacent-field papers), so
this version uses Crossref's default relevance ranking restricted to 2024+,
plus the arXiv API for recent topical preprints. Dedupes by DOI/arXiv id.

Application: living related-work list for the cover letter, reviewer
responses, and 'what's new since submission' checks.
"""
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "related_work_watchlist.json")

QUERIES = [
    "adversarial machine learning spectrum sensing wireless",
    "adversarial attacks radio signal classification modulation",
    "C-V2X security jamming sidelink",
    "regulatory emission mask compliant adversarial",
    "primary user emulation attack detection",
    "spectrum sensing deep learning V2X safety",
]

seen = {}


def add(key, rec):
    if key not in seen:
        seen[key] = rec


print("=== Crossref (relevance, 2024+) ===")
for q in QUERIES:
    try:
        r = requests.get("https://api.crossref.org/works",
                         params={"query": q,
                                 "filter": "from-pub-date:2024-01-01,type:journal-article",
                                 "rows": 12, "mailto": "precheck@example.org"},
                         timeout=40)
        r.raise_for_status()
        items = r.json()["message"]["items"]
    except Exception as e:  # noqa: BLE001
        print("  failed:", q, e)
        continue
    n = 0
    for it in items:
        doi = it.get("DOI")
        title = (it.get("title") or [""])[0]
        cites = it.get("is-referenced-by-count", 0)
        year = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
        venue = (it.get("container-title") or [""])[0]
        if not doi or not title or cites < 3:
            continue
        # topical guard: at least one domain keyword in title
        if not re.search(r"adversar|spectrum|jam|V2X|vehic|radio|cognitive|emission|sensing", title, re.I):
            continue
        add(doi, {"source": "crossref", "doi": doi, "title": title,
                  "venue": venue, "year": year, "cited_by": cites, "query": q})
        n += 1
    print(f"  {q[:50]:50s} -> {n} kept")
    time.sleep(1)

print("\n=== arXiv (recent, topical) ===")
ATOM = "{http://www.w3.org/2005/Atom}"
for q in QUERIES:
    try:
        r = requests.get("http://export.arxiv.org/api/query",
                         params={"search_query": f'all:"{q}"',
                                 "sortBy": "submittedDate", "sortOrder": "descending",
                                 "max_results": 8},
                         timeout=40)
        root = ET.fromstring(r.content)
    except Exception as e:  # noqa: BLE001
        print("  failed:", q, e)
        continue
    n = 0
    for e in root.findall(f"{ATOM}entry"):
        title = re.sub(r"\s+", " ", e.findtext(f"{ATOM}title", "").strip())
        aid = e.findtext(f"{ATOM}id", "")
        pub = e.findtext(f"{ATOM}published", "")[:10]
        if not title or (pub and pub < "2024-01-01"):
            continue
        if not re.search(r"adversar|spectrum|jam|V2X|vehic|radio|cognitive|emission|sensing", title, re.I):
            continue
        add(aid, {"source": "arxiv", "arxiv_id": aid, "title": title,
                  "published": pub, "query": q})
        n += 1
    print(f"  {q[:50]:50s} -> {n} kept")
    time.sleep(3)

all_recs = list(seen.values())
cross = [r for r in all_recs if r["source"] == "crossref"]
cross.sort(key=lambda r: -r["cited_by"])
arx = sorted([r for r in all_recs if r["source"] == "arxiv"],
             key=lambda r: -r.get("published", ""))

print(f"\n{len(cross)} journal works (2024+, cited, topical) — top 20:")
for r in cross[:20]:
    print(f"  [{r['cited_by']:3d} cites] {r['year']} {r['title'][:80]} ({r['venue'][:35]})")
print(f"\n{len(arx)} arXiv preprints (2024+) — 10 most recent:")
for r in arx[:10]:
    print(f"  {r['published']} {r['title'][:90]}")

with open(OUT, "w") as f:
    json.dump({"crossref": cross, "arxiv": arx}, f, indent=1)
print(f"\nsaved {len(all_recs)} records -> {OUT}")
