"""Step 4: certified critical values of T_H (SN / Lobato / Bartlett b = 1) on an H grid (protocol AT1, P3, P4).

For each H (exact rational, e.g. '0.73') and p in P_LEVELS:
  q        computed quantile (remainder replaced by its mean; double precision)
  [q_lo, q_hi]  certified quantile bracket: certified P(T_H <= q_lo) < p < certified P(T_H <= q_hi)
  c_print  printed critical value (q rounded to the fewest significant digits whose rounding moves the size by <= 2e-5)
  size_at_print = certified enclosure of P(T_H > c_print)   (AT1: inside [alpha - 1e-4, alpha + 1e-4])
Signed McElroy-Politis t (Q(1) = 2 int Btilde^2): T_H = 2 t^2, so the two-sided level-a point of t is sqrt(q(1 - a)/2) and the
one-sided level-a point is sqrt(q(1 - 2a)/2); p = 0.80 and 0.98 are included for that conversion.
Usage: python e2_production_sn.py J H1 H2 ...   -> results/E2_sn_cells_<H1>_<Hk>.json (merged later)
"""
import json
import math
import os
import sys
import time

from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import certify as cf  # noqa: E402
from flint import ctx  # noqa: E402

P_LEVELS = [0.80, 0.90, 0.95, 0.975, 0.98, 0.99]
Q_HALF = {0.80: 15.1, 0.90: 28.33, 0.95: 45.53, 0.975: 66.59, 0.98: 74.19, 0.99: 100.35}


def round_sig(x, d):
    return float(f"{x:.{d}g}")


def one_H(Hs, J):
    ctx.prec = 128
    t0 = time.perf_counter()
    sysm = cf.BridgeSystem(Hs, J)
    tb = time.perf_counter() - t0
    Hf = float(Hs)
    trk = 1.0 / ((2 * Hf + 1) * (Hf + 1)) - 1.0 / 6.0
    cells = {}
    for p in P_LEVELS:
        c0 = time.perf_counter()
        f = lambda c: sysm.cdf_T_computed_tilt(c)["point"] - p
        g0 = Q_HALF[p] / 6.0 / trk
        lo_b, hi_b = 0.4 * g0, 2.5 * g0
        while f(lo_b) > 0:
            lo_b *= 0.5
        while f(hi_b) < 0:
            hi_b *= 2.0
        q = brentq(f, lo_b, hi_b, xtol=1e-13 * hi_b, rtol=1e-14)
        r = sysm.cdf_T_computed_tilt(q)
        h = q * 1e-6
        dens = (sysm.cdf_T_computed_tilt(q + h)["point"] - sysm.cdf_T_computed_tilt(q - h)["point"]) / (2 * h)
        rho = 0.5 * (r["hi"] - r["lo"]) / (dens * q)
        t_comp = time.perf_counter() - c0
        # certified quantile bracket
        cert = None
        fac = 1.5
        for _ in range(5):
            c_lo, c_hi = q * (1 - fac * rho - 1e-12), q * (1 + fac * rho + 1e-12)
            A = sysm.cdf_T_certified_tilt(c_lo)
            B = sysm.cdf_T_certified_tilt(c_hi)
            if A["hi"] < p < B["lo"]:
                cert = (c_lo, c_hi, A, B)
                break
            fac *= 2
        # printed value: fewest significant digits with rounding effect <= 2e-5 in probability
        for d in range(3, 12):
            cp = round_sig(q, d)
            if abs(cp - q) * dens <= 2e-5:
                break
        P = sysm.cdf_T_certified_tilt(cp)
        alpha = 1 - p
        size_lo, size_hi = 1 - P["hi"], 1 - P["lo"]
        cells[f"{p:.3f}"] = {
            "p": p, "alpha": alpha, "q": q, "density": dens, "rel_half_width_computed": rho,
            "q_lo": cert[0] if cert else None, "q_hi": cert[1] if cert else None, "certified_bracket": cert is not None,
            "F_at_q_lo": [cert[2]["lo"], cert[2]["hi"]] if cert else None, "F_at_q_hi": [cert[3]["lo"], cert[3]["hi"]] if cert else None,
            "c_print": cp, "digits": d, "size_at_print": [size_lo, size_hi],
            "AT1_pass": (alpha - 1e-4 <= size_lo) and (size_hi <= alpha + 1e-4),
            "X_radius_max": max(P["X_radius"], cert[2]["X_radius"] if cert else 0, cert[3]["X_radius"] if cert else 0),
            "K": P["K"], "theta_plus": P["theta_plus"], "seconds_computed": t_comp,
            "seconds_total": time.perf_counter() - c0,
            "signed_t_two_sided_point": math.sqrt(q / 2), "status": "certified" if cert else "computed"}
        print(f"H={Hs} p={p}: q={q:.10g} [{cert[0] if cert else float('nan'):.10g}, {cert[1] if cert else float('nan'):.10g}] "
              f"print {cp} size [{size_lo:.7f}, {size_hi:.7f}] AT1 {cells[f'{p:.3f}']['AT1_pass']} "
              f"{time.perf_counter() - c0:.0f}s", flush=True)
    return {"H": Hs, "J": J, "tau": float(sysm.tau.mid()), "cbar": float(sysm.cbar.mid()), "build_s": tb, "cells": cells}


if __name__ == "__main__":
    J = int(sys.argv[1])
    Hs = sys.argv[2:]
    out = {}
    for h in Hs:
        out[h] = one_H(h, J)
        path = os.path.join(WORK, "results", f"E2_sn_cells_{Hs[0]}_{Hs[-1]}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
