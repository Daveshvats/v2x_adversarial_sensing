"""Clean adaptive_dual_s11_r5.json: remove out-of-protocol cells added by a
default-grid invocation (genie -25/-20, cv2x_mask -50) and restore the
canonical dual per-arm protocol in config provenance.

Canonical dual s7 protocol (must be mirrored exactly for grid identity):
  genie     PSR grid [-50, -45, -40, -35, -30]
  cv2x_mask PSR grid [-45, -40, -35, -30, -25, -20]
"""
import json
import sys

PATH = "results/adaptive_dual_s11_r5.json"
DROP = {
    "untargeted/genie": ["psr=-25dB", "psr=-20dB"],
    "untargeted/cv2x_mask": ["psr=-50dB"],
}
CANON_PSRS = {
    "untargeted/genie": [-50.0, -45.0, -40.0, -35.0, -30.0],
    "untargeted/cv2x_mask": [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0],
}

d = json.load(open(PATH))
removed = []
for arm, keys in DROP.items():
    for k in keys:
        if k in d["runs"].get(arm, {}):
            del d["runs"][arm][k]
            removed.append(f"{arm}/{k}")

# restore config provenance to the canonical dual form
d["config"]["psr_db"] = CANON_PSRS["untargeted/genie"]
d["config"]["protocol_note"] = (
    "per-arm grids mirror adaptive_dual_s7_r5.json exactly: "
    "genie [-50..-30] (5), cv2x_mask [-45..-20] (6); grid identity across "
    "attack seeds is required for comparable MEAPs (w13-style check)"
)

# flag summary as stale so it is not mistaken for final (recomputed on next
# natural driver exit over the cleaned cells)
d["summary"] = {}

tmp = PATH + ".tmp"
with open(tmp, "w") as f:
    json.dump(d, f, indent=2)
import os
os.replace(tmp, PATH)

print("removed:", removed)
for arm, psrs in CANON_PSRS.items():
    have = sorted(d["runs"].get(arm, {}).keys())
    need = [f"psr={p:+.0f}dB" for p in sorted(psrs)]
    print(f"{arm}: have {len(have)}/{len(need)}", have)
sys.exit(0)
