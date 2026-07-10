"""Text-consistency lint for the article + dissertation drafts (CPU, seconds).

Run before committing any docs/draft or docs/dissertation change:
    venv/Scripts/python.exe scripts/check_text_consistency.py

Checks (extend the lists as claims evolve — this is the systematic fix for the class of drift
that produced three stale "four budgets" references after R-F/R-G, and it caught three more
"band [896, ~2,000)" remnants on its first run):
  1. STALE phrases that contradict the current measured state.
  2. [EXPECTED tag inventory per file (so the swap pass after each npz can tick them off).
Exit code 1 on any STALE hit (safe to wire into a pre-commit hook).

DESIGN NOTE (why there is no numeric-invariant check): a first version tried to enforce global
constants (persistence, AE ceiling) and produced only false positives — those quantities are
PROTOCOL-SCOPED (persistence 0.2367 at the k=4 single-step offset vs 0.2552 over the 152-frame
rollout; AE ceilings differ per dataset/metric). Phrase-level linting is the reliable layer;
numbers must be verified against npz, not against each other.
"""
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 console vs Unicode in tag text

ROOT = os.path.join(os.path.dirname(__file__), "..")
FILES = (glob.glob(os.path.join(ROOT, "docs", "draft", "CH*.md"))
         + glob.glob(os.path.join(ROOT, "docs", "dissertation", "rewrite", "*.md")))

# ---- 1. stale phrases (regex, case-insensitive) with the reason they are wrong NOW ----
STALE = [
    (r"four matched budgets|FOUR matched|at four budgets",
     "frontier is SIX budgets since R-F/R-G (2026-07-10)"),
    (r"2,?059\s*(tok|tokens)?\s*\(?interp",
     "the interpolated-2059 hedge is retired: upper edge measured-bracketed to (1792, 2432)"),
    (r"upper (crossing|edge) interpolated",
     "upper edge is measured-bracketed since R-G"),
    (r"single[- ]seed at 512|512 \(seed 0\)",
     "512 is two-seed since R-E"),
    (r"was not re-imported into operating point A",
     "the temporal-only corner IS measured at point A by B2 (strided-flat)"),
    (r"adds tokens, not parameters",
     "corrected to 'backbone-capacity neutral' (+1.4% per-level I/O disclosed) — review 2026-07-10"),
    (r"\[896, ?[≈~]?2,?0\d\d\)",
     "band upper edge is now the measured bracket (1792, 2432), not a point estimate "
     "(canonical: [896, e), e in (1,792, 2,432))"),
]

fail = False
print("== 1. stale phrases ==")
for path in FILES:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, ROOT)
    for pat, why in STALE:
        for m in re.finditer(pat, text, re.IGNORECASE):
            line = text[:m.start()].count("\n") + 1
            # allow explicitly historical/plan mentions
            ctx = text[max(0, m.start() - 200):m.start()]
            if any(k in ctx for k in ("PLAN", "was ", "history", "retired", "superseded",
                                       "AMENDMENT", "stale")):
                continue
            print(f"  STALE {rel}:{line}: '{m.group(0)}' -> {why}")
            fail = True
if not fail:
    print("  (none)")

print("== 2. [EXPECTED inventory ==")
total = 0
for path in FILES:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tags = re.findall(r"\[EXPECTED[^:\]]*", text)
    if tags:
        rel = os.path.relpath(path, ROOT)
        from collections import Counter
        c = Counter(t.strip("[ ") for t in tags)
        print(f"  {rel}: {sum(c.values())} — " + ", ".join(f"{k}×{v}" for k, v in sorted(c.items())))
        total += sum(c.values())
print(f"  total pending: {total}")

# --gate: assembly-time enforcement (approved triage 2026-07-10, adjustment 3): FINAL texts may
# contain NO [EXPECTED]/[PENDING] residue — every shell must be measured, cut, or explicitly
# future work. E8 assembly MUST run `check_text_consistency.py --gate` and require exit 0.
if "--gate" in sys.argv:
    pend = 0
    for path in FILES:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        pend += len(re.findall(r"\[EXPECTED|\[PENDING", text))
    if pend:
        print(f"GATE FAIL: {pend} [EXPECTED]/[PENDING] shells remain — measure, cut, or recast "
              f"as future work before assembly.")
        fail = True
    else:
        print("GATE PASS: no projected-result shells remain.")

sys.exit(1 if fail else 0)
