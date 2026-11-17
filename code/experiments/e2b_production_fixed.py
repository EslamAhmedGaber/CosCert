"""Step 4 (fixed-smoothing part): certified two-sided critical values of the exact fixed-m (Hualde-Iacone) and
cosine-series (EWC) limits.  For each H, kind, k and alpha: x with P(|stat| > x) = alpha, certified bracket [x_lo, x_hi]
(P(stat^2 <= x_lo^2) < 1 - alpha < P(stat^2 <= x_hi^2), both enclosures certified), and the H = 1/2 anchors.
Usage: python e2b_production_fixed.py H1 H2 ...  -> results/E2b_fixed_<H1>_<Hk>.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import lowrank as lr  # noqa: E402
from flint import ctx  # noqa: E402

M_LIST = list(range(1, 17))
B_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
ALPHAS = [0.10, 0.05, 0.01]

if __name__ == "__main__":
    Hs = sys.argv[1:]
    out = {}
    for h in Hs:
        ctx.prec = 128
        t0 = time.perf_counter()
        ga = lr.GramArb(h, 33)
        cells = {}
        for kind, ks in (("HI", M_LIST), ("EWC", B_LIST)):
            for k in ks:
                law = lr.FixedSmoothingLaw(h, kind, k, ga=ga)
                for a in ALPHAS:
                    t1 = time.perf_counter()
                    try:
                        r = law.two_sided_cv_certified(a)
                        cells[f"{kind}|{k}|{a}"] = {"x": r["x"], "x_lo": r["x_lo"], "x_hi": r["x_hi"], "c2": r["c2"],
                                                    "rel_bracket": r["rel_bracket"], "status": "certified",
                                                    "seconds": time.perf_counter() - t1}
                    except Exception as e:      # report, never hide
                        x = law.quantile2(1 - a) ** 0.5
                        cells[f"{kind}|{k}|{a}"] = {"x": x, "status": "computed", "error": repr(e),
                                                    "seconds": time.perf_counter() - t1}
        out[h] = {"H": h, "cells": cells, "seconds": time.perf_counter() - t0}
        print(f"H={h}: {len(cells)} cells, {sum(c['status'] == 'certified' for c in cells.values())} certified, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        with open(os.path.join(WORK, "results", f"E2b_fixed_{Hs[0]}_{Hs[-1]}.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
