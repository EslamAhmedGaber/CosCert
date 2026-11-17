"""Gate G0 (protocol AT5 / P1): H = 1/2 anchors reproduced by the Paper 3 library.

(a) T_1/2 quantiles from the closed-form characteristic function (mpmath) and from an explicit-eigenvalue Imhof sum
    (N eigenvalues (k pi)^-2 + exact trigamma mean of the omitted tail) agree to >= 10 significant digits;
(b) fixed-smoothing anchors, signed and squared separately: HI tau ~ t_2m and tau^2 ~ F(1, 2m); EWC t ~ t_B, t^2 ~ F(1, B);
    double precision (max error) and certified (containment of the exact reference);
(c) Cramer-von Mises quantiles (KPSS level case) against the published KPSS values; cosh law of int W^2 (closed form vs Imhof);
(d) the certified routes enclose every exact anchor (T_1/2 via certify.BridgeSystem, CvM via the W-law, fixed-m via lowrank).
Writes work/results/E0_anchors.json.
"""
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import polygamma
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import laws as lw, lowrank as lr, certify as cf  # noqa: E402
from flint import ctx  # noqa: E402

ctx.prec = 128
mp.mp.dps = 40
P_LEVELS = [0.90, 0.95, 0.975, 0.99]
out = {"protocol": "PROTOCOL.md AT5 / P1 (gate G0)", "date": time.strftime("%Y-%m-%d %H:%M"), "a": {}, "b": {}, "c": {}, "d": {}}


def sig_digits(x, y):
    if x == y:
        return 17.0
    return float(-math.log10(abs(x - y) / abs(y)))


# ---------------------------------------------------------------- (a) T_1/2
N = 200_000
lam = 1.0 / (np.arange(1, N + 1) * np.pi) ** 2
tail_mean = float(polygamma(1, N + 1)) / np.pi ** 2        # sum_{k>N} (k pi)^-2


def F_imhof(c):
    th = np.concatenate([[1.0], -c * lam])
    return lw.imhof_cdf(th, y=c * tail_mean)[0]


for p in P_LEVELS:
    t0 = time.perf_counter()
    q_cf = float(lw.quantile(lambda c: lw.cdf_T_half(c), p, (5.0, 400.0)))
    t1 = time.perf_counter()
    q_im = brentq(lambda c: F_imhof(c) - p, q_cf * 0.99, q_cf * 1.01, xtol=1e-12, rtol=1e-14)
    t2 = time.perf_counter()
    out["a"][str(p)] = {"q_closed_form": q_cf, "q_imhof": q_im, "sig_digits": sig_digits(q_im, q_cf),
                        "sec_closed_form": t1 - t0, "sec_imhof": t2 - t1, "imhof_N": N}
    print(f"(a) p={p}: closed form {q_cf:.13f}  Imhof {q_im:.13f}  digits {sig_digits(q_im, q_cf):.1f}", flush=True)

# ---------------------------------------------------------------- (b) fixed-smoothing anchors (double and certified)
def F1nu(c, nu):
    c = mp.mpf(c)
    return mp.betainc(mp.mpf(1) / 2, mp.mpf(nu) / 2, 0, c / (c + nu), regularized=True)


for kind, ks in (("HI", list(range(1, 17))), ("EWC", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32])):
    rows = []
    for k in ks:
        law = lr.FixedSmoothingLaw("0.5", kind, k)
        nu = 2 * k if kind == "HI" else k
        for p in (0.90, 0.95, 0.99):
            # squared anchor: P(stat^2 <= F_{1,nu}^{-1}(p)) = p
            c = float(stats.f.ppf(p, 1, nu))
            ref = F1nu(c, nu)
            Fd = law.cdf2(c)
            lo, hi = law.cdf2_certified(c)
            # signed anchor: P(|stat| <= t_nu^{-1}(1/2 + p/2)) = p, i.e. P(stat <= x) = 1/2 + p/2
            x = float(stats.t.ppf(0.5 + p / 2, nu))
            ref_t = F1nu(x * x, nu)                       # P(|t_nu| <= x) = P(F(1, nu) <= x^2)
            Fd_t = 0.5 + 0.5 * law.cdf2(x * x)
            lo_t, hi_t = law.cdf2_certified(x * x)
            rows.append({"k": k, "p": p, "sq_c": c, "sq_err_double": abs(Fd - float(ref)),
                         "sq_cert": [lo, hi], "sq_cert_contains": bool(mp.mpf(lo) <= ref <= mp.mpf(hi)),
                         "signed_x": x, "signed_err_double": abs(Fd_t - (0.5 + 0.5 * float(ref_t))),
                         "signed_cert_contains": bool(mp.mpf(lo_t) <= ref_t <= mp.mpf(hi_t))})
    out["b"][kind] = {"rows": rows,
                      "max_sq_err_double": max(r["sq_err_double"] for r in rows),
                      "max_signed_err_double": max(r["signed_err_double"] for r in rows),
                      "all_certified_contain": all(r["sq_cert_contains"] and r["signed_cert_contains"] for r in rows)}
    print(f"(b) {kind}: max double error squared {out['b'][kind]['max_sq_err_double']:.2e}, signed "
          f"{out['b'][kind]['max_signed_err_double']:.2e}; certified containment {out['b'][kind]['all_certified_contain']}", flush=True)

# ---------------------------------------------------------------- (c) CvM and the cosh law
# closed-form transforms inverted by the tilted Bromwich rule (laws.cdf_cvm_fast / cdf_l2_bm_fast; alias bound e^-36),
# cross-checked by Imhof sums over N explicit eigenvalues plus the exact trigamma mean of the omitted tail.
# (16 Nov 2026: the earlier mpmath quadrature at 30 digits was found to carry a 9e-9 error for the cosh law.)
kpss_published = {"0.9": 0.347, "0.95": 0.463, "0.975": 0.574, "0.99": 0.739}     # KPSS (1990 WP) Table 1, eta_mu, n = 2000
Nb = 200_000
lam_b = 1.0 / ((np.arange(1, Nb + 1) - 0.5) * np.pi) ** 2
tail_b = float(polygamma(1, Nb + 0.5)) / np.pi ** 2                                # sum_{k>Nb} ((k - 1/2) pi)^-2
lam_v = 1.0 / (np.arange(1, Nb + 1) * np.pi) ** 2
tail_v = float(polygamma(1, Nb + 1)) / np.pi ** 2
for p in P_LEVELS:
    q_cvm = brentq(lambda x: lw.cdf_cvm_fast(x) - p, 0.05, 3.0, xtol=1e-15, rtol=1e-15)
    q_cvm_im = brentq(lambda x: lw.imhof_cdf(lam_v, y=x - tail_v)[0] - p, q_cvm * 0.999, q_cvm * 1.001, xtol=1e-15)
    q_l2 = brentq(lambda x: lw.cdf_l2_bm_fast(x) - p, 0.05, 5.0, xtol=1e-15, rtol=1e-15)
    q_l2_im = brentq(lambda x: lw.imhof_cdf(lam_b, y=x - tail_b)[0] - p, q_l2 * 0.999, q_l2 * 1.001, xtol=1e-15)
    pub = kpss_published[str(p)]
    out["c"][str(p)] = {"cvm_quantile": q_cvm, "cvm_quantile_imhof": q_cvm_im, "cvm_digits": sig_digits(q_cvm_im, q_cvm),
                        "kpss_published": pub, "published_equals_rounded": round(q_cvm, 3) == pub,
                        "true_size_at_published": 1.0 - lw.cdf_cvm_fast(pub),
                        "l2bm_quantile_closed": q_l2, "l2bm_quantile_imhof": q_l2_im, "l2bm_digits": sig_digits(q_l2_im, q_l2)}
    print(f"(c) p={p}: CvM {q_cvm:.12f} / Imhof {q_cvm_im:.12f} (KPSS {pub}; size at published "
          f"{out['c'][str(p)]['true_size_at_published']:.6f}); int W^2 {q_l2:.12f} / Imhof {q_l2_im:.12f}", flush=True)

# ---------------------------------------------------------------- (d) certified routes enclose the exact anchors
s5 = cf.BridgeSystem("0.5", 4096)
dT = {}
for p in P_LEVELS:
    q = out["a"][str(p)]["q_closed_form"]
    Fex = float(lw.cdf_T_half(q))
    r = s5.cdf_T_certified_tilt(q)
    dT[str(p)] = {"c": q, "F_exact": Fex, "cert": [r["lo"], r["hi"]], "contains": r["lo"] <= Fex <= r["hi"],
                  "X_radius": r["X_radius"], "width": r["width"], "K": r["K"]}
    print(f"(d) T_1/2 p={p}: certified [{r['lo']:.10f}, {r['hi']:.10f}] contains {dT[str(p)]['contains']}", flush=True)
dW = {}
for p in (0.90, 0.95, 0.99):
    q = out["c"][str(p)]["cvm_quantile"]
    Fex = float(lw.cdf_cvm_fast(q))
    r = s5.cdf_Wfull(q, certified=True)
    dW[str(p)] = {"x": q, "F_exact": Fex, "cert": [r["lo"], r["hi"]], "contains": r["lo"] <= Fex <= r["hi"]}
    print(f"(d) CvM p={p}: certified [{r['lo']:.10f}, {r['hi']:.10f}] contains {dW[str(p)]['contains']}", flush=True)
out["d"] = {"T_half_J4096": dT, "cvm_J4096": dW,
            "fixed_smoothing": {k: v["all_certified_contain"] for k, v in out["b"].items()}}
out["gate_G0_pass"] = (all(v["sig_digits"] >= 10 for v in out["a"].values())
                       and all(v["cvm_digits"] >= 10 and v["l2bm_digits"] >= 10 for v in out["c"].values())
                       and all(v["max_sq_err_double"] < 1e-10 and v["max_signed_err_double"] < 1e-10 for v in out["b"].values())
                       and all(v["contains"] for v in dT.values()) and all(v["contains"] for v in dW.values())
                       and all(out["d"]["fixed_smoothing"].values()))
res_dir = os.path.join(WORK, "results")
os.makedirs(res_dir, exist_ok=True)
with open(os.path.join(res_dir, "E0_anchors.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=float)
print("GATE G0 PASS:", out["gate_G0_pass"], flush=True)
