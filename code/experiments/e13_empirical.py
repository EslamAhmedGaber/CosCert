"""E13: Empirical illustration (Section 6): the long-run mean of the Nile annual minima, AD 622-1284 (n = 663).

For the Roda Gauge minima (data/nile_minima_622_1284.csv) the script computes
  - the Robinson (1995) local Whittle (LW) and demeaned exact local Whittle (ELW; Shimotsu and Phillips 2005,
    Shimotsu 2010) memory estimates with bandwidth floor(n^0.65), and the one-sided 90 % upper confidence bound
    H_up = min(0.99, H_LW + z_0.90 / (2 sqrt(m))) of Section 4.4;
  - for SN, HI (m = 4, 16) and EWC (B = 8, 32): nominal 95 % confidence intervals for the mean with short-memory
    (H = 1/2) critical values and with CosCert critical values at H_LW, H_ELW and H_up;
  - the exact finite-sample probability (Proposition 7) that each interval misses the mean under the fitted Gaussian
    ARFIMA(0, d_LW, 0) model at n = 663;
  - the test of the illustrative level mu0 = 1100 cm, with certified limiting p-values at H_LW (rounded to four
    decimals) for the CosCert tests;
  - the ADG-MAC (Abadir, Distaso and Giraitis 2009) interval and test, and the rate-free self-normalised subsampling
    test (Politis, Romano and Wolf 1999; Shao 2010) with its interval obtained by test inversion: the subsample
    statistics are centred at the sample mean and do not depend on mu0, so the accepted set is an explicit interval.
Outputs: results/E13_empirical.json, results/E13_empirical.csv
"""
import csv
import json
import math
import os
import sys
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import certify as cf, finite_n as fn, lowrank as lr  # noqa: E402
from flint import ctx  # noqa: E402

DATA = os.path.join(WORK, "data")
RES = os.path.join(WORK, "results")

SN_HALF_05 = 45.526085752448
MU0 = 1100.0
TESTS = ("SN", "HI4", "HI16", "EWC8", "EWC32")


def load_series(csv_name, col_name):
    with open(os.path.join(DATA, csv_name), newline="", encoding="utf-8") as fh:
        return np.asarray([float(r[col_name]) for r in csv.DictReader(fh)], dtype=float)


def load_splines():
    grid = json.load(open(os.path.join(RES, "E3b_fixed_cv_grid.json")))
    H = np.array(grid["H"])
    cv = {k: CubicSpline(H, np.array(v)) for k, v in grid["cv"].items()}
    tab = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))
    Hs = sorted(float(h) for h in tab["cells"])
    sncv = CubicSpline(Hs, [tab["cells"][f"{h:.2f}"]["0.950"]["q"] for h in Hs])
    return cv, sncv


def se_components(x):
    """Sample mean and the standard errors of SN, HI(4, 16) and EWC(8, 32), so that each statistic is
    (xbar - mu0) / se (for SN, T_n = ((xbar - mu0) / se_SN)^2)."""
    n = x.size
    xbar = float(x.mean())
    S = np.cumsum(x - xbar)
    se = {"SN": math.sqrt(float(np.sum(S * S) / n ** 2) / n)}
    I2 = fn.periodogram_fourier(x[None, :], 16)[0]
    se["HI4"] = math.sqrt(float(I2[:4].mean()) / n)
    se["HI16"] = math.sqrt(float(I2[:16].mean()) / n)
    t = np.arange(1, n + 1, dtype=float)
    for B in (8, 32):
        Cm = np.sqrt(2.0 / n) * np.cos(np.pi * np.outer(np.arange(1, B + 1), t - 0.5) / n)
        Lam = Cm @ x
        se[f"EWC{B}"] = math.sqrt(float(np.mean(Lam * Lam)) / n)
    return xbar, se


def subsampling_interval(x, b, level=0.95):
    """Interval {mu0 : subsampling p-value >= 1 - level} for the rate-free SN subsampling test of p3.finite_n.
    The block statistics are centred at xbar, so the p-value depends on mu0 only through T_n(mu0)."""
    n = x.size
    xbar = float(x.mean())
    S = np.cumsum(x - xbar)
    Cn = float(np.sum(S * S) / n ** 2)
    W = np.lib.stride_tricks.sliding_window_view(x - xbar, b)
    wbar = W.mean(axis=1)
    Sw = np.cumsum(W - wbar[:, None], axis=1)
    Tb = b * wbar ** 2 / np.maximum((Sw * Sw).sum(axis=1) / b ** 2, 1e-300)
    k = int(math.ceil((1.0 - level) * Tb.size))        # accept iff #{Tb >= T_n} >= k
    t_star = float(np.sort(Tb)[::-1][k - 1])
    half = math.sqrt(t_star * Cn / n)
    return xbar - half, xbar + half, t_star


def main():
    cv, sncv = load_splines()
    x = load_series("nile_minima_622_1284.csv", "min_level_cm")
    n = x.size
    mlw = int(math.floor(n ** 0.65))
    b_sub = int(round(n ** 0.60))
    se_H = 1.0 / (2.0 * math.sqrt(mlw))
    d_lw, twopi_G = fn.local_whittle(x[None, :], mlw, return_G=True)
    d_lw, twopi_G = float(d_lw[0]), float(twopi_G[0])
    d_elw = float(fn.exact_local_whittle(x[None, :], mlw)[0])
    H_lw = float(np.clip(d_lw + 0.5, 0.50, 0.99))
    H_elw = float(np.clip(d_elw + 0.5, 0.50, 0.99))
    H_up = float(np.clip(H_lw + stats.norm.ppf(0.90) * se_H, 0.50, 0.99))

    xbar, ses = se_components(x)
    diff = xbar - MU0
    stat = {"SN": (diff / ses["SN"]) ** 2}
    stat.update({k: diff / ses[k] for k in TESTS[1:]})

    # certified limiting p-values at H_LW (rounded to four decimals as an exact rational for Arb)
    H_str = f"{H_lw:.4f}"
    ctx.prec = 128
    box = cf.BridgeSystem(H_str, 4096).cdf_T_certified_quad(stat["SN"])
    p_cc = {"SN": (max(0.0, 1.0 - box["hi"]), min(1.0, 1.0 - box["lo"]))}
    box_half = cf.BridgeSystem("0.50", 2048).cdf_T_certified_quad(stat["SN"])
    p_nv = {"SN": 1.0 - 0.5 * (box_half["lo"] + box_half["hi"])}
    ga = lr.GramArb(H_str, 33)
    for k, (kind, q) in {"HI4": ("HI", 4), "HI16": ("HI", 16), "EWC8": ("EWC", 8), "EWC32": ("EWC", 32)}.items():
        lo_cdf, hi_cdf = lr.FixedSmoothingLaw(H_str, kind, q, ga=ga).cdf2_certified(stat[k] ** 2)
        p_cc[k] = (max(0.0, 1.0 - hi_cdf), min(1.0, 1.0 - lo_cdf))
        p_nv[k] = float(2.0 * (1.0 - stats.t.cdf(abs(stat[k]), 2 * q if kind == "HI" else q)))

    # exact finite-sample laws under the fitted Gaussian ARFIMA(0, d_LW, 0) model (Proposition 7)
    g_arf = fn.acf_arfima0(d_lw, n)
    exact = {"SN": fn.SNExact(g_arf), "HI4": fn.LowRankExact(g_arf, "HI", 4), "HI16": fn.LowRankExact(g_arf, "HI", 16),
             "EWC8": fn.LowRankExact(g_arf, "EWC", 8), "EWC32": fn.LowRankExact(g_arf, "EWC", 32)}

    c_naive = {"SN": SN_HALF_05, "HI4": float(stats.t.ppf(0.975, 8)), "HI16": float(stats.t.ppf(0.975, 32)),
               "EWC8": float(stats.t.ppf(0.975, 8)), "EWC32": float(stats.t.ppf(0.975, 32))}
    key = {"HI4": "HI|4|0.05", "HI16": "HI|16|0.05", "EWC8": "EWC|8|0.05", "EWC32": "EWC|32|0.05"}

    def cutoff(k, H):
        return float(sncv(H)) if k == "SN" else float(cv[key[k]](H))

    tests = {}
    for k in TESTS:
        cuts = {"naive": c_naive[k], "lw": cutoff(k, H_lw), "elw": cutoff(k, H_elw), "up": cutoff(k, H_up)}
        half = {r: (math.sqrt(c) if k == "SN" else c) * ses[k] for r, c in cuts.items()}
        size = {r: 100.0 * float(exact[k].reject(c) if k == "SN" else exact[k].reject_two_sided(c)) for r, c in cuts.items()}
        s_abs = stat[k] if k == "SN" else abs(stat[k])
        rej = {r: bool(s_abs > c) for r, c in cuts.items()}
        tests[k] = {
            "stat": float(stat[k]), "se": float(ses[k]),
            "cv": cuts, "ci95": {r: [xbar - h, xbar + h] for r, h in half.items()},
            "ci_ratio": {r: half[r] / half["naive"] for r in ("lw", "elw", "up")},
            "noncoverage_exact_pct": size, "reject_mu0": rej,
            "p_naive": float(p_nv[k]), "p_coscert_lw": 0.5 * (p_cc[k][0] + p_cc[k][1]),
            "p_coscert_lw_box": [float(p_cc[k][0]), float(p_cc[k][1])],
            "reversal_lw": bool(rej["naive"] != rej["lw"]),
        }

    # ADG-MAC (its standard error does not depend on mu0: the periodogram at Fourier frequencies j >= 1 is mean-free)
    t_mac = float(fn.adg_mac_stat((x - MU0)[None, :], mlw, dhat=np.array([d_lw]), twopi_G=np.array([twopi_G]))[0])
    se_mac = abs(diff / t_mac)
    z975 = float(stats.norm.ppf(0.975))
    tests["ADG_MAC"] = {"stat": t_mac, "se": se_mac, "cv": z975, "p_value": float(2.0 * stats.norm.sf(abs(t_mac))),
                        "reject_mu0": bool(abs(t_mac) > z975), "ci95": [xbar - z975 * se_mac, xbar + z975 * se_mac]}
    # rate-free SN subsampling: p-value at mu0 and the inverted interval, checked against the library p-value
    p_sub = float(fn.subsampling_sn_pvalue((x - MU0)[None, :], b=b_sub)[0])
    lo_s, hi_s, t_star = subsampling_interval(x, b_sub)
    eps = 1e-6 * (hi_s - lo_s)
    p_in = [float(fn.subsampling_sn_pvalue((x - m)[None, :], b=b_sub)[0]) for m in (lo_s + eps, hi_s - eps)]
    p_out = [float(fn.subsampling_sn_pvalue((x - m)[None, :], b=b_sub)[0]) for m in (lo_s - eps, hi_s + eps)]
    assert min(p_in) >= 0.05 > max(p_out), (p_in, p_out)
    tests["Subsampling_SN"] = {"stat": float(stat["SN"]), "b_sub": b_sub, "p_value": p_sub, "reject_mu0": bool(p_sub < 0.05),
                               "ci95": [lo_s, hi_s], "t_star": t_star}

    res = {"id": "nile_minima", "label": "Nile annual minima, Roda Gauge, AD 622-1284", "n": n, "unit": "cm",
           "mu0": MU0, "xbar": xbar, "sd": float(x.std(ddof=1)), "min": float(x.min()), "max": float(x.max()),
           "m_lw": mlw, "b_sub": b_sub, "d_lw": d_lw, "H_lw": H_lw, "d_elw": d_elw, "H_elw": H_elw,
           "se_H": se_H, "H_up90": H_up, "H_cert": H_str, "tests": tests}
    print(f"Nile: n={n} xbar={xbar:.2f} H_lw={H_lw:.4f} H_elw={H_elw:.4f} H_up={H_up:.4f} se_H={se_H:.4f}")
    for k in TESTS:
        t = tests[k]
        print(f"  {k:5s} stat={t['stat']:8.3f} cv n/lw/up={t['cv']['naive']:.3f}/{t['cv']['lw']:.3f}/{t['cv']['up']:.3f} "
              f"CI nv=[{t['ci95']['naive'][0]:.1f},{t['ci95']['naive'][1]:.1f}] cc=[{t['ci95']['lw'][0]:.1f},{t['ci95']['lw'][1]:.1f}] "
              f"ratio lw/up={t['ci_ratio']['lw']:.2f}/{t['ci_ratio']['up']:.2f} miss% nv/lw/up={t['noncoverage_exact_pct']['naive']:.2f}/"
              f"{t['noncoverage_exact_pct']['lw']:.3f}/{t['noncoverage_exact_pct']['up']:.4f} p nv/cc={t['p_naive']:.4f}/{t['p_coscert_lw']:.4f}")
    print(f"  ADG-MAC t={t_mac:.3f} p={tests['ADG_MAC']['p_value']:.4f} CI={tests['ADG_MAC']['ci95']}")
    print(f"  SubSN b={b_sub} p={p_sub:.4f} CI={[lo_s, hi_s]}")

    json.dump({"cases": [res]}, open(os.path.join(RES, "E13_empirical.json"), "w", encoding="utf-8"), indent=1)
    with open(os.path.join(RES, "E13_empirical.csv"), "w", newline="", encoding="utf-8") as fh:
        columns = ["test", "stat", "cv_naive", "cv_coscert_lw", "cv_coscert_elw", "cv_coscert_up",
                   "ci95_naive_lo", "ci95_naive_hi", "ci95_lw_lo", "ci95_lw_hi", "ci95_up_lo", "ci95_up_hi",
                   "ratio_lw", "ratio_up", "miss_naive_pct", "miss_lw_pct", "miss_up_pct", "p_naive", "p_coscert_lw",
                   "reject_naive", "reject_lw", "reject_elw", "reject_up",
                   "cv_reference", "ci95_reference_lo", "ci95_reference_hi", "p_reference", "reject_reference"]
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for k in TESTS:
            t = tests[k]
            row = {"test": k, "stat": t["stat"], "cv_naive": t["cv"]["naive"],
                   "p_naive": t["p_naive"], "p_coscert_lw": t["p_coscert_lw"]}
            for rule in ("lw", "elw", "up"):
                row[f"cv_coscert_{rule}"] = t["cv"][rule]
            for rule in ("naive", "lw", "up"):
                row[f"ci95_{rule}_lo"], row[f"ci95_{rule}_hi"] = t["ci95"][rule]
                row[f"miss_{rule}_pct"] = t["noncoverage_exact_pct"][rule]
            for rule in ("lw", "up"):
                row[f"ratio_{rule}"] = t["ci_ratio"][rule]
            for rule in ("naive", "lw", "elw", "up"):
                row[f"reject_{rule}"] = t["reject_mu0"][rule]
            writer.writerow(row)
        for k in ("ADG_MAC", "Subsampling_SN"):
            t = tests[k]
            writer.writerow({"test": k, "stat": t["stat"], "cv_reference": t.get("cv", ""),
                             "ci95_reference_lo": t["ci95"][0], "ci95_reference_hi": t["ci95"][1],
                             "p_reference": t["p_value"], "reject_reference": t["reject_mu0"]})
    print("wrote results/E13_empirical.json and results/E13_empirical.csv", flush=True)


if __name__ == "__main__":
    main()
