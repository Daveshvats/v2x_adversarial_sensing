#!/usr/bin/env python3
"""Style-consistency patch for paper/main.tex (post proselint+textstat pass):
  1. centre -> center (4 instances; paper is otherwise US spelling, 14 x center)
  2. correctly-classified -> correctly classified (2x; CMOS 7.86: -ly adverb +
     participle takes no hyphen)
  3. adversarially-trained -> adversarially trained (1x; same rule)
'early-fusion' is deliberately KEPT (noun compound, not an -ly adverb).
Zero content/number changes; idempotent; prints a per-edit diff count.
"""
import re

TEX = "/home/z/my-project/v2x_repo/paper/main.tex"

with open(TEX) as f:
    tex = f.read()

edits = [
    (r"\bcentres\b", "centers"),
    (r"\bcentre\b", "center"),
    (r"correctly-classified", "correctly classified"),
    (r"adversarially-trained", "adversarially trained"),
]

total = 0
for pat, rep in edits:
    n = len(re.findall(pat, tex))
    if n:
        tex = re.sub(pat, rep, tex)
    print(f"  {pat!s:35s} -> {rep!s:25s} : {n} replacement(s)")
    total += n

with open(TEX, "w") as f:
    f.write(tex)
print(f"total edits: {total}")

# verify
with open(TEX) as f:
    v = f.read()
assert "centre" not in v and "correctly-classified" not in v and "adversarially-trained" not in v
print("verification: clean (no centre / -ly hyphens remain)")
