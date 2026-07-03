"""Unit test for rhvd.DtLabeler (H-grid gate variable) against a brute-force reference on a
synthetic constant-motion trajectory with one digit at a known world column. Exact-match required.

  venv/Scripts/python.exe scripts/test_dt_labeler.py
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from hvp_rev.rhvd import DtLabeler                     # noqa: E402
from hvp_rev.step1 import rel_positions                # noqa: E402

W, SPEED, T, WARM, V = 768, 4, 480, 192, 64
DIGIT_LO, DIGIT_HI = 100, 128                          # world columns of the single digit


def build_fake():
    acts = np.ones((1, T), np.int64)                   # constant rightward pan
    rel = rel_positions(acts[0], SPEED)                # (T,)
    clips = np.zeros((1, T, 64, 64), np.uint8)
    for t in range(T):
        cols = (rel[t] + np.arange(V)) % W             # world cols under the viewport
        vis = np.isin(cols, np.arange(DIGIT_LO, DIGIT_HI))
        clips[0, t, 20:48, vis] = 255                  # digit rows 20..47
    return SimpleNamespace(acts=acts, clips_u8=clips, W=W, speed=SPEED), rel


def brute_dt(rel, tf):
    """Reference: per visible digit column at tf, dt since last PRIOR frame it was visible."""
    def visible(t, c):
        return (c - rel[t]) % W < V
    cols = [c for c in range(DIGIT_LO, DIGIT_HI) if visible(tf, c)]
    out = []
    for c in cols:
        prior = [m for m in range(tf) if visible(m, c)]
        out.append(tf - prior[-1] if prior else 10 ** 9)
    return np.array(out)


def main():
    data, rel = build_fake()
    dtl = DtLabeler(data, WARM)
    checked = reentries = 0
    for tf in range(WARM, T):
        frame = data.clips_u8[0, tf]
        if frame.max() == 0:
            got_dt, got_seen = dtl.label(0, tf)
            assert got_dt == 0.0 and got_seen == 1.0, f"empty frame mislabeled at {tf}"
            continue
        got_dt, got_seen = dtl.label(0, tf)
        ref = brute_dt(rel, tf)
        novel = ref > 6
        if not novel.any():
            assert got_dt == 0.0, f"t{tf}: expected dt=0 (no novel), got {got_dt}"
        else:
            refn = ref[novel]
            finite = refn < 10 ** 8
            exp_seen = float(finite.mean())
            exp_dt = float(np.median(refn[finite])) if finite.any() else 1e9
            assert abs(got_dt - exp_dt) < 1e-6, f"t{tf}: dt {got_dt} != ref {exp_dt}"
            assert abs(got_seen - exp_seen) < 1e-6, f"t{tf}: seen {got_seen} != ref {exp_seen}"
            if exp_dt > 48:
                reentries += 1
        checked += 1
    # sanity of the scenario itself: constant pan at speed 4 on a 768 torus -> the digit re-enters
    # with dt ~= 192 - visible_span; require that we actually exercised the beyond-window path
    assert reentries >= 10, f"scenario exercised only {reentries} beyond-48 re-entry frames"
    print(f"DT-LABELER TEST PASS: {checked} frames checked, {reentries} beyond-48 re-entry frames, "
          f"exact match vs brute force")


if __name__ == "__main__":
    main()
