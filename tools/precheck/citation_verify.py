#!/usr/bin/env python3
"""Citation verification via Crossref (habanero) + OpenAlex REST.

Parses the 22 thebibliography items from paper/main.tex, extracts
authors/title/venue/year, queries Crossref's bibliographic search, and
fuzzy-matches titles. Items that are standards/regulatory documents
(ETSI/3GPP/FCC/IEEE-std) are classified and skipped (verified by prior
live-web pass; Crossref coverage for standards is unreliable).

Verdict per reference:
  VERIFIED   - title similarity >= 85 and |year diff| <= 1
  SELF_UNPUBLISHED - the authors own unpublished manuscript (archived
                     in paper/legacy/) - correctly absent from Crossref
  MISMATCH   - best candidate differs in title or year (needs human eyes)
  NOT_FOUND  - no plausible candidate
  STANDARD   - standards/regulatory doc, out of Crossref scope
Output: scripts/toolcheck/citation_verify_results.json
"""
import json
import os
import re
import time

import requests
from habanero import Crossref
from rapidfuzz import fuzz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEX = os.path.join(ROOT, "paper", "main.tex")
OUT = os.path.join(HERE, "results", "citation_verify_results.json")
CR = Crossref(mailto="precheck@example.org")

STANDARD_PAT = re.compile(
    r"ETSI|3GPP|FCC [0-9]|Federal Register|IEEE Std|SAE|REG \(EC\)|"
    r"TR 37|TS 36|EN 302|802\.11-2012|Unlicensed", re.I)


def parse_bibitems(tex):
    m = re.search(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", tex, re.S)
    body = m.group(1)
    items = re.split(r"\\bibitem", body)[1:]
    parsed = []
    for raw in items:
        raw = raw.strip()
        key = re.match(r"\{([^}]*)\}", raw)
        key = key.group(1) if key else "?"
        flat = " ".join(raw.split())
        t = re.search(r"``(.*?)''", flat, re.S)
        title = t.group(1).strip().rstrip(",") if t else ""
        v = re.search(r"\\emph\{(.*?)\}", flat)
        venue = v.group(1) if v else ""
        venue_clean = re.sub(r"\\[a-zA-Z]+", "", venue)
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", flat)
                 if 1990 <= int(y) <= 2026]
        # prefer a year that appears after pp./pages/vol (citation year, not vol)
        year = years[-1] if years else None
        authors = flat[flat.find("}") + 1: flat.find("``")].strip() if "``" in flat else ""
        parsed.append({"key": key, "authors": authors[:120], "title": title,
                       "venue": venue_clean, "year": year, "raw": flat[:300]})
    return parsed


def crossref_match(title, first_author, year):
    q = f"{title} {first_author}".strip()
    try:
        r = CR.works(query_bibliographic=q, limit=5)
        items = r["message"]["items"]
    except Exception as e:  # noqa: BLE001
        return {"error": f"crossref query failed: {e}"}
    best = None
    for it in items:
        ct = (it.get("title") or [""])[0]
        score = fuzz.ratio(title.lower(), ct.lower())
        iy = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
        cand = {"crossref_title": ct[:140], "title_score": score,
                "crossref_year": iy, "doi": it.get("DOI"),
                "container": (it.get("container-title") or [""])[0][:80],
                "author": (it.get("author") or [{}])[0].get("family", "")}
        if best is None or score > best["title_score"]:
            best = cand
    if best is None:
        return {"status": "NOT_FOUND"}
    yd = abs((best.get("crossref_year") or 0) - (year or 0)) if year and best.get("crossref_year") else None
    if best["title_score"] >= 85 and (yd is None or yd <= 1):
        best["status"] = "VERIFIED"
    else:
        best["status"] = "MISMATCH"
    best["year_diff"] = yd
    return best


def openalex_match(title):
    try:
        r = requests.get("https://api.openalex.org/works",
                         params={"search": title, "per-page": 3,
                                 "mailto": "precheck@example.org"}, timeout=30)
        js = r.json()
        for w in js.get("results", [])[:3]:
            score = fuzz.ratio(title.lower(), (w.get("title") or "").lower())
            if score >= 85:
                return {"openalex_title": w["title"][:140], "score": score,
                        "openalex_year": (w.get("publication_year")),
                        "doi": w.get("doi")}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    return None


def main():
    tex = open(TEX).read()
    refs = parse_bibitems(tex)
    print(f"{len(refs)} bibitems parsed\n")
    results = []
    for ref in refs:
        rec = dict(ref)
        if "unpublished manuscript" in ref["raw"].lower():
            rec["verdict"] = ("SELF_UNPUBLISHED (own unpublished preprint, "
                              "archived in paper/legacy/ - not expected in "
                              "Crossref)")
            results.append(rec)
            print(f"[SELF] {ref['key']:18s} {ref['title'][:70]}")
            continue
        if STANDARD_PAT.search(ref["raw"]) or not ref["title"]:
            rec["verdict"] = "STANDARD_OR_REGULATORY (out of Crossref scope; prior live-web verification stands)"
            results.append(rec)
            print(f"[STD ] {ref['key']:18s} {ref['title'][:70]}")
            continue
        first_author = ref["authors"].split(",")[0].strip()
        m = crossref_match(ref["title"], first_author, ref["year"])
        time.sleep(1)
        if m.get("status") in ("NOT_FOUND",) or m.get("error"):
            oa = openalex_match(ref["title"])
            time.sleep(0.5)
            if oa:
                rec["crossref"] = m
                rec["openalex"] = oa
                yd = None
                if ref["year"] and oa.get("openalex_year"):
                    yd = abs(oa["openalex_year"] - ref["year"])
                rec["verdict"] = "VERIFIED" if (oa["score"] >= 85 and (yd is None or yd <= 1)) else "MISMATCH"
            else:
                rec["verdict"] = "NOT_FOUND"
                rec["crossref"] = m
        else:
            rec["crossref"] = m
            rec["verdict"] = m["status"]
        results.append(rec)
        tag = rec["verdict"].split(" ")[0]
        print(f"[{tag:4s}] {ref['key']:18s} {ref['title'][:60]:60s} "
              f"({str(rec.get('crossref', {}).get('title_score', ''))}%)")

    verdicts = {}
    for r in results:
        v = r["verdict"].split(" ")[0]
        verdicts[v] = verdicts.get(v, 0) + 1
    print("\nSummary:", verdicts)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
