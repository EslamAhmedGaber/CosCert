"""E9b: rigorous held-out size error of the PRINTED response surface.

For each held-out (odd-hundredth) H and level p, the printed significant-digit coefficients are parsed as exact
rationals in Arb ball arithmetic and evaluated to a ball [q_eval_lo, q_eval_hi] containing the exact real value of
q_hat = exp(sum a_i (H - 1/2)^i) / (1 - H). If [q_eval_lo, q_eval_hi] lies inside the certified bracket [q_lo, q_hi]
of E2, monotonicity of F gives F(q_hat) in [lower(F(q_lo)), upper(F(q_hi))]; otherwise F(q_eval_lo) and F(q_eval_hi)
are enclosed directly by the production certificate (BridgeSystem, J = 16384, t = 25, tilt 1/4, adaptive acb.integral).
Bound = max(p - lower, upper - p), in percentage points.
Output: results/E9_surface_certified.json
"""
from fractions import Fraction
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import certify as cf  # noqa: E402
from flint import arb, ctx  # noqa: E402

RES = os.path.join(WORK, "results")
DIGITS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
J = 16384


def iv(s):
    if isinstance(s, (list, tuple)):
        return float(s[0]), float(s[1])
    a, b = s.strip("[]").split(",")
    return float(a), float(b)


def eval_printed_surface_arb(coef_strs, h_str):
    """Evaluate exp(sum_k c_k (H - 1/2)^k) / (1 - H) in 128-bit Arb from exact printed decimal strings."""
    ctx.prec = 128
    fr_h = Fraction(h_str)
    Ha = arb(fr_h.numerator) / fr_h.denominator
    u = Ha - arb(1) / 2
    # Horner evaluation from exact rational decimal strings
    acc = arb(0)
    for s in reversed(coef_strs):
        fr_c = Fraction(s)
        acc = acc * u + arb(fr_c.numerator) / fr_c.denominator
    return acc.exp() / (1 - Ha)


SF = json.load(open(os.path.join(RES, "E9_surface.json")))
cells = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
todo, out = {}, []
for p, rec in SF["fits"].items():
    coef_strs = [f"{a:.{DIGITS - 1}e}" for a in rec["coef"][: int(rec["K"]) + 1]]
    for h, cc in cells.items():
        H = float(h)
        if round(100 * H) % 2 != 1:
            continue
        c = cc[p]
        q_ball = eval_printed_surface_arb(coef_strs, h)
        q_lo_ev = cf.flo(q_ball)
        q_hi_ev = cf.fup(q_ball)
        qh = float(q_ball.mid())
        r = {
            "H": h, "p": p, "q_hat_printed": qh,
            "q_hat_ball_lo": q_lo_ev, "q_hat_ball_hi": q_hi_ev,
            "q_lo": c["q_lo"], "q_hi": c["q_hi"],
        }
        if c["q_lo"] <= q_lo_ev and q_hi_ev <= c["q_hi"]:
            lo, hi = iv(c["F_at_q_lo"])[0], iv(c["F_at_q_hi"])[1]
            r.update(route="bracket (monotone)", F_lo=lo, F_hi=hi)
        else:
            todo.setdefault(h, []).append(len(out))
            r.update(route="direct certificate")
        out.append(r)

print("cells:", len(out), " direct evaluations needed:", sum(len(v) for v in todo.values()), "at H =", sorted(todo), flush=True)
for h, idx in sorted(todo.items()):
    t0 = time.perf_counter()
    ctx.prec = 128
    sysm = cf.BridgeSystem(h, J)
    for i in idx:
        P_lo = sysm.cdf_T_certified_quad(out[i]["q_hat_ball_lo"])
        P_hi = sysm.cdf_T_certified_quad(out[i]["q_hat_ball_hi"])
        out[i].update(F_lo=float(P_lo["lo"]), F_hi=float(P_hi["hi"]))
    print(f"H={h}: {len(idx)} cells, {time.perf_counter() - t0:.1f}s", flush=True)

for r in out:
    pp = float(r["p"])
    r["bound_pp"] = 100 * max(pp - r["F_lo"], r["F_hi"] - pp, 0.0)
mx = max(out, key=lambda r: r["bound_pp"])
res = {
    "digits": DIGITS, "J": J, "t": 25, "tilt": 0.25, "method": "ball_surface_endpoints_quad",
    "cells": out, "max_bound_pp": mx["bound_pp"], "argmax": [mx["H"], mx["p"]],
    "max_bound_pp_by_p": {p: max(r["bound_pp"] for r in out if r["p"] == p) for p in SF["fits"]},
    "n_direct": sum(r["route"] == "direct certificate" for r in out),
}
json.dump(res, open(os.path.join(RES, "E9_surface_certified.json"), "w"), indent=1)
print("max certified held-out size error of the printed surface: %.3e pp at H=%s p=%s" % (mx["bound_pp"], mx["H"], mx["p"]), flush=True)
