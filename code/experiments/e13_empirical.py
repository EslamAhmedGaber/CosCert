"""E13: Empirical applications (Section 6) across three real-world long-memory series.

Series (all contiguous, zero missing values, stored in data/):
  1. AusElec-NSW  : NSW daily YoY log electricity demand growth (2012-03-01..2015-02-28, n=1095), H0: mu = 0%.
  2. London-PM2.5 : London GR9 daily log PM2.5 (2017-01-01..2018-03-30, n=454), H0: mu = log(10) [and log(15)].
  3. Nile-Minima  : Nilometer annual minimum levels (622-1284 AD, n=663), H0: mu = 1100 cm.

For each series computes:
  - Robinson (1995) LW and Shimotsu & Phillips (2005) / Shimotsu (2010) demeaned ELW memory estimates H_hat and s.e.
  - Self-normalised (SN), fixed-m (HI, m=4, 16), equal-weighted cosine (EWC, B=8, 32), ADG-MAC (2009), and rate-free
    self-normalised subsampling (b = round(n^0.6)) test statistics and p-values.
  - Naive short-memory (H=1/2) vs CosCert (at H_LW, H_ELW, and conservative upper-CI H_up) 5% critical values,
    numerically enclosed limiting plug-in p-values, and 95% confidence intervals with exact CI width expansion ratios.
  - Exact finite-sample ARFIMA(0, d_hat, 0) sizes of both naive and CosCert limit cutoffs via Proposition 6 (SNExact
    and LowRankExact).
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


def load_series(csv_name, col_name):
    vals = []
    with open(os.path.join(DATA, csv_name), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vals.append(float(r[col_name]))
    return np.asarray(vals, dtype=float)


def load_splines():
    grid = json.load(open(os.path.join(RES, "E3b_fixed_cv_grid.json")))
    H = np.array(grid["H"])
    cv = {k: CubicSpline(H, np.array(v)) for k, v in grid["cv"].items()}
    tab = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))
    Hs = sorted(float(h) for h in tab["cells"])
    sncv = CubicSpline(Hs, [tab["cells"][f"{h:.2f}"]["0.950"]["q"] for h in Hs])
    return cv, sncv


def se_components(x):
    """Return xbar and the unscaled denominator standard errors for SN, HI(4,16), EWC(8,32) so stat = (xbar-mu0)/se
    (for SN: T_n = ((xbar - mu0) / se_sn)^2)."""
    n = x.size
    xbar = float(x.mean())
    S = np.cumsum(x - xbar)
    Cn = float(np.sum(S * S) / (n ** 2))
    se_sn = math.sqrt(Cn / n)
    I2 = fn.periodogram_fourier(x[None, :], 16)[0]
    se_hi4 = math.sqrt(float(I2[:4].mean()) / n)
    se_hi16 = math.sqrt(float(I2[:16].mean()) / n)
    t = np.arange(1, n + 1, dtype=float)
    se_ewc = {}
    for B in (8, 32):
        Cm = np.sqrt(2.0 / n) * np.cos(np.pi * np.outer(np.arange(1, B + 1), t - 0.5) / n)
        Lam = Cm @ x
        se_ewc[B] = math.sqrt(float(np.mean(Lam * Lam)) / n)
    return xbar, {"SN": se_sn, "HI4": se_hi4, "HI16": se_hi16, "EWC8": se_ewc[8], "EWC32": se_ewc[32]}


def analyze_case(case_id, label, x, mu0, mu0_label, unit, cv, sncv):
    n = x.size
    mlw = int(math.floor(n ** 0.65))
    b_sub = int(round(n ** 0.60))
    se_H = 1.0 / (2.0 * math.sqrt(mlw))
    d_lw, twopi_G = fn.local_whittle(x[None, :], mlw, return_G=True)
    d_lw = float(d_lw[0])
    twopi_G = float(twopi_G[0])
    d_elw = float(fn.exact_local_whittle(x[None, :], mlw)[0])
    H_lw = float(np.clip(d_lw + 0.5, 0.50, 0.99))
    H_elw = float(np.clip(d_elw + 0.5, 0.50, 0.99))
    H_up = float(np.clip(H_lw + stats.norm.ppf(0.90) * se_H, 0.50, 0.99))

    xbar, ses = se_components(x)
    diff = xbar - mu0
    stat_sn = (diff / ses["SN"]) ** 2
    stat_hi4 = diff / ses["HI4"]
    stat_hi16 = diff / ses["HI16"]
    stat_ewc8 = diff / ses["EWC8"]
    stat_ewc32 = diff / ses["EWC32"]

    # ADG-MAC (2009)
    t_mac = float(fn.adg_mac_stat((x - mu0)[None, :], mlw, dhat=np.array([d_lw]), twopi_G=np.array([twopi_G]))[0])
    p_mac = float(2.0 * (1.0 - stats.norm.cdf(abs(t_mac))))
    se_mac = abs(diff / t_mac) if abs(t_mac) > 1e-12 else float("nan")

    # Rate-free self-normalised subsampling
    p_sub = float(fn.subsampling_sn_pvalue((x - mu0)[None, :], b=b_sub)[0])

    # Certified exact p-values at H_lw (rounded to 4 decimals as exact fraction for Arb)
    H_str = f"{H_lw:.4f}"
    ctx.prec = 128
    bs_sn = cf.BridgeSystem(H_str, 4096)
    p_sn_cert_box = bs_sn.cdf_T_certified_quad(stat_sn) if stat_sn > 0.0 else {"lo": 0.0, "hi": 0.0}
    p_sn_cert = (max(0.0, 1.0 - p_sn_cert_box["hi"]), min(1.0, 1.0 - p_sn_cert_box["lo"]))
    bs_half = cf.BridgeSystem("0.50", 2048)
    p_sn_naive_box = bs_half.cdf_T_certified_quad(stat_sn) if stat_sn > 0.0 else {"lo": 0.0, "hi": 0.0}
    p_sn_naive = 1.0 - 0.5 * (p_sn_naive_box["lo"] + p_sn_naive_box["hi"])

    ga = lr.GramArb(H_str, 33)
    laws = {
        "HI4": lr.FixedSmoothingLaw(H_str, "HI", 4, ga=ga),
        "HI16": lr.FixedSmoothingLaw(H_str, "HI", 16, ga=ga),
        "EWC8": lr.FixedSmoothingLaw(H_str, "EWC", 8, ga=ga),
        "EWC32": lr.FixedSmoothingLaw(H_str, "EWC", 32, ga=ga),
    }

    # Exact finite-sample ARFIMA(0, d_lw, 0) null size of naive vs CosCert limit cutoffs (Proposition 6)
    g_arf = fn.acf_arfima0(d_lw, n)
    sn_ex = fn.SNExact(g_arf)
    hi4_ex = fn.LowRankExact(g_arf, "HI", 4)
    hi16_ex = fn.LowRankExact(g_arf, "HI", 16)
    ewc8_ex = fn.LowRankExact(g_arf, "EWC", 8)
    ewc32_ex = fn.LowRankExact(g_arf, "EWC", 32)

    z975 = stats.norm.ppf(0.975)
    t_naive = {
        "SN": SN_HALF_05,
        "HI4": float(stats.t.ppf(0.975, 8)),
        "HI16": float(stats.t.ppf(0.975, 32)),
        "EWC8": float(stats.t.ppf(0.975, 8)),
        "EWC32": float(stats.t.ppf(0.975, 32)),
        "ADG_MAC": float(z975),
    }
    cv_lw = {
        "SN": float(sncv(H_lw)),
        "HI4": float(cv["HI|4|0.05"](H_lw)),
        "HI16": float(cv["HI|16|0.05"](H_lw)),
        "EWC8": float(cv["EWC|8|0.05"](H_lw)),
        "EWC32": float(cv["EWC|32|0.05"](H_lw)),
        "ADG_MAC": float(z975),
    }
    cv_elw = {
        "SN": float(sncv(H_elw)),
        "HI4": float(cv["HI|4|0.05"](H_elw)),
        "HI16": float(cv["HI|16|0.05"](H_elw)),
        "EWC8": float(cv["EWC|8|0.05"](H_elw)),
        "EWC32": float(cv["EWC|32|0.05"](H_elw)),
    }
    cv_up = {
        "SN": float(sncv(H_up)),
        "HI4": float(cv["HI|4|0.05"](H_up)),
        "HI16": float(cv["HI|16|0.05"](H_up)),
        "EWC8": float(cv["EWC|8|0.05"](H_up)),
        "EWC32": float(cv["EWC|32|0.05"](H_up)),
    }

    tests = {}
    for key, stat_v, dof_n, ex_obj in [
        ("SN", stat_sn, None, sn_ex),
        ("HI4", stat_hi4, 8, hi4_ex),
        ("HI16", stat_hi16, 32, hi16_ex),
        ("EWC8", stat_ewc8, 8, ewc8_ex),
        ("EWC32", stat_ewc32, 32, ewc32_ex),
    ]:
        c_nv = t_naive[key]
        c_cc = cv_lw[key]
        c_el = cv_elw[key]
        c_up = cv_up[key]
        if key == "SN":
            hw_nv = math.sqrt(c_nv) * ses[key]
            hw_cc = math.sqrt(c_cc) * ses[key]
            hw_up = math.sqrt(c_up) * ses[key]
            rej_nv = bool(stat_v > c_nv)
            rej_cc = bool(stat_v > c_cc)
            rej_el = bool(stat_v > c_el)
            rej_up = bool(stat_v > c_up)
            p_nv = p_sn_naive
            p_cc = 0.5 * (p_sn_cert[0] + p_sn_cert[1])
            p_cc_lo, p_cc_hi = p_sn_cert
            fs_size_nv = float(ex_obj.reject(c_nv))
            fs_size_cc = float(ex_obj.reject(c_cc))
        else:
            hw_nv = c_nv * ses[key]
            hw_cc = c_cc * ses[key]
            hw_up = c_up * ses[key]
            rej_nv = bool(abs(stat_v) > c_nv)
            rej_cc = bool(abs(stat_v) > c_cc)
            rej_el = bool(abs(stat_v) > c_el)
            rej_up = bool(abs(stat_v) > c_up)
            p_nv = float(2.0 * (1.0 - stats.t.cdf(abs(stat_v), dof_n)))
            lo_cdf, hi_cdf = laws[key].cdf2_certified(stat_v * stat_v) if stat_v != 0.0 else (0.0, 0.0)
            p_cc_lo, p_cc_hi = max(0.0, 1.0 - hi_cdf), min(1.0, 1.0 - lo_cdf)
            p_cc = 0.5 * (p_cc_lo + p_cc_hi)
            fs_size_nv = float(ex_obj.reject_two_sided(c_nv))
            fs_size_cc = float(ex_obj.reject_two_sided(c_cc))
        tests[key] = {
            "stat": float(stat_v),
            "se": float(ses[key]),
            "cv_naive": c_nv,
            "cv_coscert_lw": c_cc,
            "cv_coscert_elw": c_el,
            "cv_coscert_up": c_up,
            "p_naive": p_nv,
            "p_coscert_lw": p_cc,
            "p_coscert_lw_lo": p_cc_lo,
            "p_coscert_lw_hi": p_cc_hi,
            "reject_naive": rej_nv,
            "reject_coscert_lw": rej_cc,
            "reject_coscert_elw": rej_el,
            "reject_coscert_up": rej_up,
            "reversal_lw": bool(rej_nv != rej_cc),
            "ci95_naive": [xbar - hw_nv, xbar + hw_nv],
            "ci95_coscert_lw": [xbar - hw_cc, xbar + hw_cc],
            "ci95_coscert_up": [xbar - hw_up, xbar + hw_up],
            "ci_width_ratio": hw_cc / hw_nv,
            "exact_arfima_size_naive_pct": 100.0 * fs_size_nv,
            "exact_arfima_size_coscert_pct": 100.0 * fs_size_cc,
        }

    tests["ADG_MAC"] = {
        "stat": t_mac, "se": se_mac, "cv": float(z975), "p_value": p_mac,
        "reject": bool(abs(t_mac) > z975),
        "ci95": [xbar - z975 * se_mac, xbar + z975 * se_mac],
    }
    tests["Subsampling_SN"] = {
        "stat": stat_sn, "b_sub": b_sub, "p_value": p_sub, "reject": bool(p_sub < 0.05),
    }

    return {
        "id": case_id, "label": label, "n": n, "unit": unit,
        "mu0": mu0, "mu0_label": mu0_label,
        "xbar": xbar, "sd": float(x.std(ddof=1)),
        "m_lw": mlw, "b_sub": b_sub,
        "d_lw": d_lw, "H_lw": H_lw, "d_elw": d_elw, "H_elw": H_elw,
        "se_H": se_H, "H_up90": H_up,
        "tests": tests,
    }


def main():
    cv, sncv = load_splines()
    x_aus = load_series("aus_elec_nsw_yoy_2012_2015.csv", "yoy_log_growth_pct")
    x_lon = load_series("london_gr9_pm25_2017_2018.csv", "log_pm25")
    x_nile = load_series("nile_minima_622_1284.csv", "min_level_cm")

    cases = [
        ("aus_elec_nsw", "Australian Electricity Demand (NSW YoY %, 2012-2015)", x_aus, 0.0, "0%", "%"),
        ("london_pm25_who05", "London Urban PM2.5 (GR9 log ug/m^3, 2017-2018; reference 10)", x_lon, math.log(10.0), "log(10)", "log ug/m^3"),
        ("london_pm25_who21", "London Urban PM2.5 (GR9 log ug/m^3, 2017-2018; reference 15)", x_lon, math.log(15.0), "log(15)", "log ug/m^3"),
        ("nile_minima", "Nile River Annual Minima (Roda Gauge, 622-1284 AD)", x_nile, 1100.0, "1100 cm", "cm"),
    ]
    results = []
    flat_rows = []
    for cid, lab, x, mu0, mu0_lab, unit in cases:
        res = analyze_case(cid, lab, x, mu0, mu0_lab, unit, cv, sncv)
        results.append(res)
        print(f"\n=== {lab} (n={res['n']}, xbar={res['xbar']:.4f}, mu0={mu0:.4f}, H_lw={res['H_lw']:.3f}, H_elw={res['H_elw']:.3f}, se={res['se_H']:.3f}) ===")
        for k in ("SN", "HI4", "HI16", "EWC8", "EWC32"):
            t = res["tests"][k]
            print(
                f"  {k:5s}: stat={t['stat']:7.2f} | naive_cv={t['cv_naive']:6.2f} (p={t['p_naive']:.4f}, rej={str(t['reject_naive']):5s}, exSz={t['exact_arfima_size_naive_pct']:5.1f}%) | "
                f"CC_cv={t['cv_coscert_lw']:7.2f} (p={t['p_coscert_lw']:.4f}, rej={str(t['reject_coscert_lw']):5s}, exSz={t['exact_arfima_size_coscert_pct']:4.2f}%) | "
                f"CI_ratio={t['ci_width_ratio']:.2f}x | rev={t['reversal_lw']}"
            )
            flat_rows.append([
                cid, res["n"], res["xbar"], mu0, res["H_lw"], res["H_elw"], res["se_H"], k,
                t["stat"], t["cv_naive"], t["cv_coscert_lw"], t["cv_coscert_elw"], t["cv_coscert_up"],
                t["p_naive"], t["p_coscert_lw"], t["reject_naive"], t["reject_coscert_lw"], t["reversal_lw"],
                t["ci95_naive"][0], t["ci95_naive"][1], t["ci95_coscert_lw"][0], t["ci95_coscert_lw"][1],
                t["ci_width_ratio"], t["exact_arfima_size_naive_pct"], t["exact_arfima_size_coscert_pct"],
            ])
        tm = res["tests"]["ADG_MAC"]
        ts = res["tests"]["Subsampling_SN"]
        print(f"  ADG  : stat={tm['stat']:7.2f} (p={tm['p_value']:.4f}, rej={tm['reject']})")
        print(f"  SubSN: b={ts['b_sub']:3d} (p={ts['p_value']:.4f}, rej={ts['reject']})")

    out_json = os.path.join(RES, "E13_empirical.json")
    json.dump({"cases": results}, open(out_json, "w", encoding="utf-8"), indent=1)
    out_csv = os.path.join(RES, "E13_empirical.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "case_id", "n", "xbar", "mu0", "H_lw", "H_elw", "se_H", "test",
            "stat", "cv_naive", "cv_coscert_lw", "cv_coscert_elw", "cv_coscert_up",
            "p_naive", "p_coscert_lw", "reject_naive", "reject_coscert_lw", "reversal_lw",
            "ci95_naive_lo", "ci95_naive_hi", "ci95_cc_lo", "ci95_cc_hi",
            "ci_width_ratio", "exact_arfima_size_naive_pct", "exact_arfima_size_coscert_pct",
        ])
        w.writerows(flat_rows)
    print(f"\nwrote {out_json} and {out_csv}", flush=True)


if __name__ == "__main__":
    main()
