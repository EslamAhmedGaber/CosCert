"""Step 3 benchmark (protocol section 6): full cost of a certified T_H critical value as a function of the truncation J.

For H in the argument list, J in {512, ..., 16384}, p in {.90, .95, .99}:
  build time of the c-independent system (Kummer values in arb, head blocks, nested tail Frobenius norms, couplings),
  computed quantile q_J (remainder replaced by its mean), computed remainder bracket [F_lo, F_hi] at q_J, its width in
  probability and the implied relative quantile width (local slope), and the time per computed CDF;
  certified F bracket at q_J for J in CERT_J (width, finite-dimensional radius, K, time).
Usage: python e1_benchmark_J.py 0.5 0.75 ...   -> work/results/E1_benchmark_H<H>.json
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

J_LADDER = [512, 1024, 2048, 4096, 8192, 16384]
CERT_J = [4096, 16384]
P_LEVELS = [0.90, 0.95, 0.99]


def run(Hs):
    for Hs_ in Hs:
        ctx.prec = 128
        res = {"H": Hs_, "rows": []}
        km = cf.Kummer(Hs_, max(J_LADDER))
        guess = {}
        for J in J_LADDER:
            t0 = time.perf_counter()
            sysm = cf.BridgeSystem(Hs_, J, km=km)
            tb = time.perf_counter() - t0
            for p in P_LEVELS:
                f = lambda c: sysm.cdf_T_computed_tilt(c)["point"] - p
                if p in guess:
                    lo_b, hi_b = guess[p] * 0.98, guess[p] * 1.02
                else:   # scaling guess q_p(H) ~ q_p(1/2) (1/6) / tr K_br(H) (within a factor 2 on [0.5, 0.99])
                    Hf = float(Hs_)
                    trk = 1.0 / ((2 * Hf + 1) * (Hf + 1)) - 1.0 / 6.0
                    g0 = {0.90: 28.33, 0.95: 45.53, 0.99: 100.35}[p] / 6.0 / trk
                    lo_b, hi_b = 0.4 * g0, 2.5 * g0
                while f(hi_b) < 0:
                    hi_b *= 2
                while f(lo_b) > 0:
                    lo_b /= 2
                t1 = time.perf_counter()
                q = brentq(f, lo_b, hi_b, xtol=1e-12 * hi_b, rtol=1e-13)
                tq = time.perf_counter() - t1
                guess[p] = q
                t2 = time.perf_counter()
                r = sysm.cdf_T_computed_tilt(q)
                tc = time.perf_counter() - t2
                h = q * 1e-6
                slope = (sysm.cdf_T_computed_tilt(q + h)["point"] - sysm.cdf_T_computed_tilt(q - h)["point"]) / (2 * h)
                row = {"J": J, "p": p, "build_s": tb, "q": q, "quantile_search_s": tq, "cdf_computed_s": tc,
                       "F_lo": r["lo"], "F_hi": r["hi"], "width_F": r["hi"] - r["lo"], "density": slope,
                       "rel_q_width": (r["hi"] - r["lo"]) / (slope * q), "tau": float(sysm.tau.mid()),
                       "cbar": float(sysm.cbar.mid()), "K_double": r["K"]}
                if J in CERT_J:
                    t3 = time.perf_counter()
                    c = sysm.cdf_T_certified_tilt(q)
                    row.update({"cert_lo": c["lo"], "cert_hi": c["hi"], "cert_width": c["width"], "cert_X_radius": c["X_radius"],
                                "cert_K": c["K"], "cert_s": time.perf_counter() - t3, "cert_theta_plus": c["theta_plus"]})
                res["rows"].append(row)
                print(f"H={Hs_} J={J} p={p}: q={q:.10f} widthF={row['width_F']:.2e} relq={row['rel_q_width']:.2e} "
                      f"build {tb:.1f}s cdf {tc:.2f}s" + (f" | cert [{row['cert_lo']:.12f},{row['cert_hi']:.12f}] "
                      f"rad {row['cert_X_radius']:.1e} {row['cert_s']:.0f}s" if J in CERT_J else ""), flush=True)
            del sysm
        out = os.path.join(WORK, "results", f"E1_benchmark_H{Hs_}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    run(sys.argv[1:])
