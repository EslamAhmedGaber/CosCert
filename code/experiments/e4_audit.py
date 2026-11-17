"""Step 6: audit of published critical values in test-size units (protocol section 3, P8).

For every ledger cell inside the certified scope (H >= 1/2) the script reports
  true_size        P(stat > published) under the exact limit law (certified enclosure where available),
  dev_pp           100 * (true_size - nominal alpha)   [percentage points]
  rel_dev          published / exact critical value - 1
  mc_se_pp         Monte Carlo standard error of a rejection frequency at alpha with the source's N (sqrt(a(1-a)/N), in pp)
  disc_size        true size at the published value under the EXACT law of the source's own discretised design
                   (Lobato: white noise, n = 5000; KPSS: iid, n = 2000; McElroy-Politis: fGn grid n = 1000;
                    Hualde-Iacone: Stieltjes sums of the limit integrals on a 10,000-step fGn grid)
  verdict          per protocol P8: 'accurate (within MC error)', 'deviates > 0.1 pp at 5% but within MC error',
                   'unexplained' (beyond 3 MC standard errors and not explained by discretisation).
Sign conventions: squared statistics (T, F-type) are upper-tail; signed t-type values are one-sided upper quantiles
(McElroy-Politis, Abadir-Paruolo, Kiefer-Vogelsang) converted through T = 2 t^2 (Bartlett b = 1) or T = t^2 (AP z_u):
P(t > x) = (1 - F_T(k x^2)) / 2.  Hualde-Iacone values are two-sided: P(|tau| > x).
Usage: python e4_audit.py [part ...]  (parts: half, hi, mp)  -> results/E4_audit_<part>.csv
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import laws as lw, lowrank as lr, finite_n as fn, certify as cf  # noqa: E402
from flint import ctx  # noqa: E402

LED = pd.read_csv(os.path.join(WORK, "literature", "ledger_critical_values.csv"))
OUT = os.path.join(WORK, "results")


def verdict(dev_pp, se_pp, alpha):
    if se_pp and abs(dev_pp) > 3 * se_pp:
        return "unexplained (> 3 MC s.e.)"
    if abs(dev_pp) > 0.1 and abs(alpha - 0.05) < 1e-12:
        return "deviates > 0.1 pp at 5% (within 3 MC s.e.)"
    return "accurate within MC error"


def sn_white_noise_cdf(n):
    return fn.SNExact(fn.make_acf(("wn",), n))


def kpss_exact_cdf_factory(n, trend=False):
    """Exact law of KPSS eta (iid N(0,1), sample size n): eta = n^-2 sum S_t^2 / s^2 with residuals from 1 (or 1, t)."""
    t = np.arange(1, n + 1, dtype=float)
    X = np.ones((n, 1)) if not trend else np.column_stack([np.ones(n), t])
    Qx, _ = np.linalg.qr(X)
    E = np.eye(n) - Qx @ Qx.T                       # residual projector
    G = np.cumsum(E, axis=0)                        # S = D E u
    M = G.T @ G / n ** 2
    mu = np.linalg.eigvalsh(M)                      # M E = M: eigenvectors inside range(E) for nonzero mu
    mu = np.sort(mu)[::-1][: n - X.shape[1]]
    # eta <= x  <=>  u^T (M - x E / n) u <= 0 ; M and E commute, eigenvalues mu_i - x/n on range(E)
    def cdf(x):
        th = mu - x / n
        return lw.imhof_cdf(th)[0]
    return cdf


def part_half():
    rows = []
    q_exact = {}
    for p in (0.80, 0.90, 0.95, 0.975, 0.98, 0.99, 0.995):
        from scipy.optimize import brentq
        q_exact[p] = brentq(lambda c: lw.cdf_T_half_fast(c) - p, 1.0, 400.0, xtol=1e-13, rtol=1e-15) if hasattr(lw, "cdf_T_half_fast") \
            else float(lw.quantile(lambda c: lw.cdf_T_half_fast(c), p, (1.0, 400.0)))
    sn5000 = sn_white_noise_cdf(5000)
    # Lobato (2001): U_1 = T, squared upper tail
    for _, r in LED[LED.source_id == "Lobato2001"].iterrows():
        a = float(r.level_alpha); v = float(r.published_value)
        F = float(lw.cdf_T_half_fast(v)); size = 1 - F
        dsize = sn5000.reject(v)
        se = 100 * math.sqrt(a * (1 - a) / float(r.replications_N))
        rows.append({"ledger_id": r.ledger_id, "source": "Lobato2001", "H": 0.5, "alpha": a, "published": v, "exact_cv": q_exact[round(1 - a, 3)],
                     "rel_dev": v / q_exact[round(1 - a, 3)] - 1, "true_size": size, "dev_pp": 100 * (size - a), "mc_se_pp": se,
                     "disc_size": dsize, "disc_dev_pp": 100 * (dsize - a), "status": "exact (closed form)", "verdict": verdict(100 * (size - a), se, a)})
    # Abadir-Paruolo (1997) z_u and Kiefer-Vogelsang (2002) t*: one-sided signed
    for sid, k in (("AP1997", 1.0), ("KV2002wp", 2.0)):
        for _, r in LED[LED.source_id == sid].iterrows():
            v = float(r.published_value)
            if v <= 0:
                continue                                      # lower quantiles are the mirror images
            a = float(r.level_alpha)
            F = float(lw.cdf_T_half_fast(k * v * v)); size = (1 - F) / 2
            p2 = round(1 - 2 * a, 3)
            exact = math.sqrt(q_exact[p2] / k)
            rows.append({"ledger_id": r.ledger_id, "source": sid, "H": 0.5, "alpha": a, "published": v, "exact_cv": exact,
                         "rel_dev": v / exact - 1, "true_size": size, "dev_pp": 100 * (size - a), "mc_se_pp": None,
                         "disc_size": None, "disc_dev_pp": None, "status": "exact (closed form)",
                         "verdict": "analytic source; difference = printed rounding" if abs(v / exact - 1) < 5e-4 else "check"})
    # KPSS (1990 WP and 1992 journal): CvM (eta_mu) exact; eta_tau computed by finite-n extrapolation
    cdf2000 = kpss_exact_cdf_factory(2000)
    cdf2000t = kpss_exact_cdf_factory(2000, trend=True)
    cdf1000t = kpss_exact_cdf_factory(1000, trend=True)
    for _, r in LED[LED.source_id.isin(["KPSS1990wp", "KPSS1992"])].iterrows():
        a = float(r.level_alpha); v = float(r.published_value)
        se = 100 * math.sqrt(a * (1 - a) / 50000)
        if "tau" in str(r.ledger_id) or "V_2" in str(r.limit_functional) or "tau" in str(r.statistic):
            d2000 = 1 - cdf2000t(v); d1000 = 1 - cdf1000t(v)
            size = d2000 + (d2000 - d1000)                     # Richardson (first order in 1/n), status computed
            status = "computed (Richardson from exact n = 1000, 2000)"
            dsize = d2000
        else:
            size = 1 - lw.cdf_cvm_fast(v); status = "exact (closed form)"
            dsize = 1 - cdf2000(v)
        rows.append({"ledger_id": r.ledger_id, "source": r.source_id, "H": 0.5, "alpha": a, "published": v, "exact_cv": None,
                     "rel_dev": None, "true_size": size, "dev_pp": 100 * (size - a), "mc_se_pp": se, "disc_size": dsize,
                     "disc_dev_pp": 100 * (dsize - a), "status": status, "verdict": verdict(100 * (size - a), se, a)})
    return pd.DataFrame(rows)


def part_hi():
    rows = []
    sub = LED[(LED.source_id == "HI2017") & (LED.memory_value.astype(float) >= -1e-12)]
    for (dlt,), grp in sub.groupby(["memory_value"]):
        H = round(0.5 + float(dlt), 2)
        ga = lr.GramArb(f"{H:.2f}", 33)
        g10k = fn.acf_fgn(H, 10000)
        for _, r in grp.iterrows():
            m = int(str(r.ledger_id).rsplit("-m", 1)[1]); v = float(r.published_value); a = 0.05
            law = lr.FixedSmoothingLaw(f"{H:.2f}", "HI", m, ga=ga)
            lo, hi = law.cdf2_certified(v * v)
            size_lo, size_hi = 1 - hi, 1 - lo
            exact = math.sqrt(law.quantile2(0.95))
            disc = fn.LowRankExact(g10k, "HI", m).reject_two_sided(v)
            se = 100 * math.sqrt(a * (1 - a) / 10000)
            dev = 100 * (0.5 * (size_lo + size_hi) - a)
            rows.append({"ledger_id": r.ledger_id, "source": "HI2017", "H": H, "m": m, "alpha": a, "published": v, "exact_cv": exact,
                         "rel_dev": v / exact - 1, "true_size": 0.5 * (size_lo + size_hi), "true_size_lo": size_lo, "true_size_hi": size_hi,
                         "dev_pp": dev, "mc_se_pp": se, "disc_size": disc, "disc_dev_pp": 100 * (disc - a), "status": "certified",
                         "verdict": verdict(dev, se, a)})
        print(f"HI delta={dlt}: done", flush=True)
    return pd.DataFrame(rows)


def part_mp():
    rows = []
    sub = LED[(LED.source_id == "MP2009TR") & (LED.memory_value.astype(float) >= -1e-12)]
    for (beta,), grp in sub.groupby(["memory_value"]):
        H = round((float(beta) + 1) / 2, 2)
        ctx.prec = 128
        sysm = cf.BridgeSystem(f"{H:.2f}", 16384)
        sn1000 = fn.SNExact(fn.acf_fgn(H, 1000))
        for _, r in grp.iterrows():
            a = float(r.level_alpha); v = float(r.published_value); c = 2 * v * v
            cert = sysm.cdf_T_certified_tilt(c)
            size_lo, size_hi = (1 - cert["hi"]) / 2, (1 - cert["lo"]) / 2
            # exact quantile (computed point value) for the relative deviation
            from scipy.optimize import brentq
            p2 = 1 - 2 * a
            q = brentq(lambda cc: sysm.cdf_T_computed_tilt(cc)["point"] - p2, 0.5 * c, 2.0 * c, xtol=1e-10 * c)
            exact = math.sqrt(q / 2)
            disc = (1 - sn1000.cdf(c)) / 2
            # MC standard error of the simulated quantile (N = 50,000) in size units: sqrt(a (1 - a) / N)
            se = 100 * math.sqrt(a * (1 - a) / 50000)
            dev = 100 * (0.5 * (size_lo + size_hi) - a)
            rows.append({"ledger_id": r.ledger_id, "source": "MP2009TR", "H": H, "beta": float(beta), "alpha": a, "published": v,
                         "exact_cv": exact, "rel_dev": v / exact - 1, "true_size": 0.5 * (size_lo + size_hi), "true_size_lo": size_lo,
                         "true_size_hi": size_hi, "dev_pp": dev, "mc_se_pp": se, "disc_size": disc, "disc_dev_pp": 100 * (disc - a),
                         "status": "certified (size); derived published value", "verdict": verdict(dev, se, a)})
        print(f"MP beta={beta}: done", flush=True)
    return pd.DataFrame(rows)



def part_hi_neg():
    """HI2017 Table 1, delta < 0 (H < 1/2): certified fixed-m limits through the closed Gram, which extends to
    0 < H < 1/2 by analytic continuation in H (both sides analytic; verified numerically 16 Nov 2026)."""
    rows = []
    sub = LED[(LED.source_id == "HI2017") & (LED.memory_value.astype(float) < -1e-12)]
    for (dlt,), grp in sub.groupby(["memory_value"]):
        H = round(0.5 + float(dlt), 2)
        ga = lr.GramArb(f"{H:.2f}", 33)
        g10k = fn.acf_fgn(H, 10000)
        for _, r in grp.iterrows():
            m = int(str(r.ledger_id).rsplit("-m", 1)[1]); v = float(r.published_value); a = 0.05
            law = lr.FixedSmoothingLaw(f"{H:.2f}", "HI", m, ga=ga)
            lo, hi = law.cdf2_certified(v * v)
            size_lo, size_hi = 1 - hi, 1 - lo
            exact = math.sqrt(law.quantile2(0.95))
            disc = fn.LowRankExact(g10k, "HI", m).reject_two_sided(v)
            se = 100 * math.sqrt(a * (1 - a) / 10000)
            dev = 100 * (0.5 * (size_lo + size_hi) - a)
            rows.append({"ledger_id": r.ledger_id, "source": "HI2017", "H": H, "m": m, "alpha": a, "published": v, "exact_cv": exact,
                         "rel_dev": v / exact - 1, "true_size": 0.5 * (size_lo + size_hi), "true_size_lo": size_lo, "true_size_hi": size_hi,
                         "dev_pp": dev, "mc_se_pp": se, "disc_size": disc, "disc_dev_pp": 100 * (disc - a),
                         "status": "certified (Gram continued to H < 1/2)", "verdict": verdict(dev, se, a)})
        print(f"HI delta={dlt}: done", flush=True)
    return pd.DataFrame(rows)


def part_mp_neg():
    """McElroy-Politis beta < 0 (H < 1/2), Bartlett b = 1: the published fit targets the law of their own design
    (fGn grid N = 1000), which is the exact finite-n SN law under fGn; compare with it ('exact design law', not a limit)."""
    rows = []
    sub = LED[(LED.source_id == "MP2009TR") & (LED.memory_value.astype(float) < -1e-12)]
    for (beta,), grp in sub.groupby(["memory_value"]):
        H = round((float(beta) + 1) / 2, 2)
        sn1000 = fn.SNExact(fn.acf_fgn(H, 1000))
        sn500 = fn.SNExact(fn.acf_fgn(H, 500))
        from scipy.optimize import brentq
        for _, r in grp.iterrows():
            a = float(r.level_alpha); v = float(r.published_value); c = 2 * v * v
            disc = (1 - sn1000.cdf(c)) / 2
            d500 = (1 - sn500.cdf(c)) / 2
            q1000 = brentq(lambda cc: sn1000.cdf(cc) - (1 - 2 * a), 0.2 * c, 5 * c, xtol=1e-10 * c)
            se = 100 * math.sqrt(a * (1 - a) / 50000)
            dev = 100 * (disc - a)
            rows.append({"ledger_id": r.ledger_id, "source": "MP2009TR", "H": H, "beta": float(beta), "alpha": a, "published": v,
                         "exact_cv": math.sqrt(q1000 / 2), "rel_dev": v / math.sqrt(q1000 / 2) - 1, "true_size": disc,
                         "dev_pp": dev, "mc_se_pp": se, "disc_size": disc, "disc_dev_pp": dev, "n500_size": d500,
                         "status": "exact law of the published design (N = 1000 fGn grid); limit not certified for H < 1/2",
                         "verdict": verdict(dev, se, a)})
        print(f"MP beta={beta}: done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["half", "hi", "mp"]
    for part in parts:
        t0 = time.perf_counter()
        df = {"half": part_half, "hi": part_hi, "mp": part_mp, "hi_neg": part_hi_neg, "mp_neg": part_mp_neg}[part]()
        df.to_csv(os.path.join(OUT, f"E4_audit_{part}.csv"), index=False)
        print(f"part {part}: {len(df)} rows, {time.perf_counter() - t0:.0f}s", flush=True)
