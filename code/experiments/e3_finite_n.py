"""Step 5, part A: exact finite-sample rejection rates (protocol AT2).

Every rate is a finite-dimensional exact law (status 'exact', numerical error < 1e-9) and is labelled with
  model (fgn / arfima0 / arfima1 with phi), H (d = H - 1/2), n, procedure (SN; HI m; EWC B), critical-value label
  (own_limit: the limit critical value of the same procedure at the true H; naive_half: the H = 1/2 value;
   hi_table1: Hualde-Iacone Table 1 printed value at delta = H - 1/2 when printed), alpha, and 'finite-sample'.
SN limit critical values are read from results/E2_sn_critical_values.json when present (certified production run);
otherwise they are computed here with certify.BridgeSystem (J = 4096, 'computed') and labelled so.
Usage: python e3_finite_n.py <model> <H...>   (model in fgn, arfima0, arfima1p, arfima1m)  -> results/E3_rates_<model>.csv
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import finite_n as fn, lowrank as lr, certify as cf, laws as lw  # noqa: E402

N_GRID = [50, 100, 200, 500, 1000, 2000]
M_LIST = [1, 2, 3, 4, 6, 8, 12, 16]
B_LIST = [4, 8, 12, 16, 24, 32]
ALPHAS = [0.10, 0.05, 0.01]
SN_HALF = {0.10: 28.330921323509, 0.05: 45.526085752448, 0.01: 100.34563942809}   # exact H = 1/2 (gate G0)


def model_spec(name, H):
    d = H - 0.5
    return {"fgn": ("fgn", H), "arfima0": ("arfima0", d), "arfima1p": ("arfima1", 0.5, d), "arfima1m": ("arfima1", -0.5, d)}[name]


def sn_limit_cv(H, cache={}):
    """Limit critical values of T_H at 1 - alpha; certified production file if available, else computed (J = 4096)."""
    if H in cache:
        return cache[H]
    path = os.path.join(WORK, "results", "E2_sn_critical_values.json")
    if os.path.exists(path):
        tab = json.load(open(path, encoding="utf-8"))
        key = f"{H:.2f}"
        if key in tab.get("cells", {}):
            cache[H] = ({a: tab["cells"][key][f"{1 - a:.3f}"]["q"] for a in ALPHAS}, "certified_production")
            return cache[H]
    if abs(H - 0.5) < 1e-12:
        cache[H] = (SN_HALF, "exact_closed_form")
        return cache[H]
    sysm = cf.BridgeSystem(f"{H:.2f}", 4096)
    out = {}
    for a in ALPHAS:
        p = 1 - a
        f = lambda c: sysm.cdf_T_computed_tilt(c)["point"] - p
        hi = 50.0
        while f(hi) < 0:
            hi *= 2
        out[a] = brentq(f, 1.0, hi, xtol=1e-10, rtol=1e-13)
    cache[H] = (out, "computed_J4096")
    return cache[H]


def hi_table1(H):
    """Printed HI (2017) Table 1 two-sided 5% critical values at delta = H - 1/2, from the ledger (None if not printed)."""
    import pandas as pd
    led = pd.read_csv(os.path.join(WORK, "literature", "ledger_critical_values.csv"))
    sel = led[(led.source_id == "HI2017") & (np.isclose(led.memory_value.astype(float), H - 0.5))]
    if sel.empty:
        return {}
    out = {}
    for _, r in sel.iterrows():
        m = int(str(r["ledger_id"]).rsplit("-m", 1)[1])
        out[m] = float(r["published_value"])
    return out


def main(name, Hs):
    rows = []
    for H in Hs:
        sn_cv, sn_src = sn_limit_cv(H)
        try:
            hi_pub = hi_table1(H)
        except Exception:
            hi_pub = {}
        lims = {}
        for m in M_LIST:
            law = lr.FixedSmoothingLaw(f"{H:.2f}", "HI", m)
            lims[("HI", m)] = {a: math.sqrt(law.quantile2(1 - a)) for a in ALPHAS}
        for B in B_LIST:
            law = lr.FixedSmoothingLaw(f"{H:.2f}", "EWC", B)
            lims[("EWC", B)] = {a: math.sqrt(law.quantile2(1 - a)) for a in ALPHAS}
        for n in N_GRID:
            g = fn.make_acf(model_spec(name, H), n)
            t0 = time.perf_counter()
            sn = fn.SNExact(g)
            for a in ALPHAS:
                for lab, cv, src in (("own_limit", sn_cv[a], sn_src), ("naive_half", SN_HALF[a], "exact_closed_form")):
                    rows.append([name, H, n, "SN", "", lab, src, cv, a, sn.reject(cv), "finite-sample exact"])
            for m in M_LIST:
                lrk = fn.LowRankExact(g, "HI", m)
                for a in ALPHAS:
                    rows.append([name, H, n, "HI", m, "own_limit", "exact_limit", lims[("HI", m)][a], a,
                                 lrk.reject_two_sided(lims[("HI", m)][a]), "finite-sample exact"])
                    tq = stats.t.ppf(1 - a / 2, 2 * m)
                    rows.append([name, H, n, "HI", m, "naive_half", "t_2m", tq, a, lrk.reject_two_sided(tq), "finite-sample exact"])
                if m in hi_pub:
                    rows.append([name, H, n, "HI", m, "hi_table1", "HI2017 Table 1", hi_pub[m], 0.05,
                                 lrk.reject_two_sided(hi_pub[m]), "finite-sample exact"])
            for B in B_LIST:
                lrk = fn.LowRankExact(g, "EWC", B)
                for a in ALPHAS:
                    rows.append([name, H, n, "EWC", B, "own_limit", "exact_limit", lims[("EWC", B)][a], a,
                                 lrk.reject_two_sided(lims[("EWC", B)][a]), "finite-sample exact"])
                    tq = stats.t.ppf(1 - a / 2, B)
                    rows.append([name, H, n, "EWC", B, "naive_half", "t_B", tq, a, lrk.reject_two_sided(tq), "finite-sample exact"])
            print(f"{name} H={H} n={n}: {time.perf_counter() - t0:.1f}s; SN own/naive 5%: "
                  f"{[r[9] for r in rows if r[2] == n and r[1] == H and r[3] == 'SN' and r[8] == 0.05]}", flush=True)
    out = os.path.join(WORK, "results", f"E3_rates_{name}_H{'_'.join(str(h) for h in Hs)}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "H", "n", "procedure", "param", "cv_label", "cv_source", "cv", "alpha", "rejection_rate", "status"])
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], [float(h) for h in sys.argv[2:]])
