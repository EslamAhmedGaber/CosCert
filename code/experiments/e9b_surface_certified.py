"""E9b: rigorous held-out size error of the PRINTED response surface (protocol AT3, amendment 7).
For each held-out (odd-hundredth) H and level p, q_hat = exp(sum a_i (H-1/2)^i)/(1-H) from the coefficients rounded to the
printed number of significant digits.  If q_hat lies in the certified bracket [q_lo, q_hi] of E2, monotonicity of F gives
F(q_hat) in [lower(F(q_lo)), upper(F(q_hi))]; otherwise F(q_hat) is enclosed directly by the production certificate
(BridgeSystem, J = 16384, t = 25, tilt 1/4).  Bound = max(p - lower, upper - p), in percentage points.
Output: results/E9_surface_certified.json
"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import certify as cf  # noqa: E402
from flint import ctx  # noqa: E402
RES = os.path.join(WORK, "results")
DIGITS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
J = 16384
def iv(s):
    if isinstance(s, (list, tuple)): return float(s[0]), float(s[1])
    a, b = s.strip("[]").split(","); return float(a), float(b)
SF = json.load(open(os.path.join(RES, "E9_surface.json"))); cells = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
todo, out = {}, []
for p, rec in SF["fits"].items():
    coef = np.array([float(f"{a:.{DIGITS - 1}e}") for a in rec["coef"][: int(rec["K"]) + 1]])
    for h, cc in cells.items():
        H = float(h)
        if round(100 * H) % 2 != 1: continue
        c = cc[p]; qh = float(np.exp(np.polyval(coef[::-1], H - 0.5)) / (1 - H)); pp = float(p)
        r = {"H": h, "p": p, "q_hat_printed": qh, "q_lo": c["q_lo"], "q_hi": c["q_hi"]}
        if c["q_lo"] <= qh <= c["q_hi"]:
            lo, hi = iv(c["F_at_q_lo"])[0], iv(c["F_at_q_hi"])[1]; r.update(route="bracket (monotone)", F_lo=lo, F_hi=hi)
        else:
            todo.setdefault(h, []).append(len(out)); r.update(route="direct certificate")
        out.append(r)
print("cells:", len(out), " direct evaluations needed:", sum(len(v) for v in todo.values()), "at H =", sorted(todo), flush=True)
for h, idx in sorted(todo.items()):
    t0 = time.perf_counter(); ctx.prec = 128; sysm = cf.BridgeSystem(h, J)
    for i in idx:
        P = sysm.cdf_T_certified_tilt(out[i]["q_hat_printed"])
        out[i].update(F_lo=float(P["lo"]), F_hi=float(P["hi"]))
    print(f"H={h}: {len(idx)} cells, {time.perf_counter() - t0:.0f}s", flush=True)
for r in out:
    pp = float(r["p"]); r["bound_pp"] = 100 * max(pp - r["F_lo"], r["F_hi"] - pp, 0.0)
mx = max(out, key=lambda r: r["bound_pp"])
res = {"digits": DIGITS, "J": J, "t": 25, "tilt": 0.25, "cells": out, "max_bound_pp": mx["bound_pp"], "argmax": [mx["H"], mx["p"]],
       "max_bound_pp_by_p": {p: max(r["bound_pp"] for r in out if r["p"] == p) for p in SF["fits"]},
       "n_direct": sum(r["route"] == "direct certificate" for r in out)}
json.dump(res, open(os.path.join(RES, "E9_surface_certified.json"), "w"), indent=1)
print("max certified held-out size error of the printed surface: %.3e pp at H=%s p=%s" % (mx["bound_pp"], mx["H"], mx["p"]), flush=True)
