"""Extended finite-sample plug-in, competitor benchmark, and exact HI (2017) oracle calculations.

Subcommands:
  n50_ewc       : Monte Carlo plug-in rejection rates at n = 50 (all procedures) and n = 100..2000 for EWC B in {8, 32}
                  using the exact same RNG streams as e3b_mc.py (R = 40,000), closing the Figure 2 n=50 and B=8,32 gaps.
                  Output: results/E3d_plugin_n50_ewc.csv
  competitors   : Head-to-head feasible long-memory mean-test benchmark (null size and local power at c = 2.8)
                  comparing CosCert plug-in (LW, ELW, conservative upper-CI) against ADG-MAC (Abadir, Distaso & Giraitis
                  2009), rate-free self-normalised subsampling (Politis, Romano & Wolf 1999), and naive H = 1/2 cutoffs.
                  Output: results/E3d_competitors.csv
  hi2017_oracle : Exact finite-sample oracle and naive t_{2m} rejection rates (via LowRankExact) for Hualde & Iacone
                  (2017) Tables 3-5 designs (T = 64, 256, 1024; m = 1..16).
                  Output: results/E3d_hi2017_exact_oracle.csv
  all           : Run all three subcommands sequentially.
"""
import csv
import json
import math
import os
import sys
import time
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import finite_n as fn  # noqa: E402

RES = os.path.join(WORK, "results")
ALPHAS = (0.10, 0.05, 0.01)
SN_HALF = {0.10: 28.330921323509, 0.05: 45.526085752448, 0.01: 100.34563942809}
BATCH = 4000
MODELS = ("fgn", "arfima0", "arfima1p", "arfima1m")


def model_spec(name, H):
    d = H - 0.5
    return {
        "fgn": ("fgn", H),
        "arfima0": ("arfima0", d),
        "arfima1p": ("arfima1", 0.5, d),
        "arfima1m": ("arfima1", -0.5, d),
    }[name]


def load_splines():
    grid = json.load(open(os.path.join(RES, "E3b_fixed_cv_grid.json")))
    H = np.array(grid["H"])
    cv = {k: CubicSpline(H, np.array(v)) for k, v in grid["cv"].items()}
    rho = {k: CubicSpline(H, np.array(v)) for k, v in grid["rho"].items()}
    tab = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))
    Hs = sorted(float(h) for h in tab["cells"])
    sncv = {a: CubicSpline(Hs, [tab["cells"][f"{h:.2f}"][f"{1 - a:.3f}"]["q"] for h in Hs]) for a in ALPHAS}
    return cv, rho, sncv


def wilson(k, n, z=2.5758):
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / den
    hw = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return c - hw, c + hw


def mc_batches(g, R, seed, batch=BATCH):
    smp = fn.CirculantSampler(g)
    rng = np.random.Generator(np.random.PCG64(seed))
    done = 0
    while done < R:
        nb = min(batch, (R - done + 1) // 2)
        X = smp.sample(nb, rng)
        X = X[: R - done]
        done += X.shape[0]
        yield X


def var_xbar(gamma):
    """Exact Var(xbar_n) = (1/n) * (gamma_0 + 2 sum_{h=1}^{n-1} (1 - h/n) gamma_h)."""
    n = gamma.size
    h = np.arange(1, n, dtype=float)
    return float((gamma[0] + 2.0 * np.sum((1.0 - h / n) * gamma[1:])) / n)


# ============================================================================ Part 1: n=50 + EWC B=8, 32 plug-in grid
def run_n50_ewc(R=40_000, bw=0.65):
    cv, rho, sncv = load_splines()
    rows = []
    t_all = time.perf_counter()
    for H in (0.5, 0.6, 0.7, 0.8, 0.9):
        for model in MODELS:
            m_id = {"fgn": 1, "arfima0": 2, "arfima1p": 3, "arfima1m": 4}[model]
            for n in (50, 100, 200, 500, 1000, 2000):
                g = fn.make_acf(model_spec(model, H), n)
                mlw = int(math.floor(n ** bw))
                seed = 31000000 + int(100 * H) * 10000 + n + m_id * 1000000
                batch = max(100, min(BATCH, int(2_000_000 // n)))
                cnt = {}
                Rtot = 0
                dhs = []
                # At n=50 run all procedures; at n>=100 run EWC B in (8, 32) on the identical stream
                if n == 50:
                    hi_list = (1, 4, 16)
                    ewc_list = (4, 8, 16, 32)
                    do_sn = True
                else:
                    hi_list = ()
                    ewc_list = (8, 32)
                    do_sn = False
                for X in mc_batches(g, R, seed, batch):
                    dh = fn.local_whittle(X, mlw)
                    dhs.append(dh)
                    Hh = np.clip(dh + 0.5, 0.01, 0.99)
                    Hs = np.clip(Hh, 0.50, 0.99)
                    st = fn.stats_from_samples(X, m_list=hi_list, B_list=ewc_list)
                    Rtot += X.shape[0]
                    for a in ALPHAS:
                        rules = {}
                        if do_sn:
                            rules[("SN", 0, "plugin_exact")] = (st["SN"], sncv[a](Hs))
                            rules[("SN", 0, "naive_half")] = (st["SN"], SN_HALF[a])
                            rules[("SN", 0, "oracle_exact")] = (st["SN"], float(sncv[a](max(H, 0.5))))
                        for kind, ks, dof in (("HI", hi_list, 2), ("EWC", ewc_list, 1)):
                            for k in ks:
                                s = np.abs(st[(kind, k)])
                                tq = stats.t.ppf(1 - a / 2, dof * k)
                                rules[(kind, k, "plugin_exact")] = (s, cv[f"{kind}|{k}|{a:.2f}"](Hh))
                                rules[(kind, k, "plugin_exact_trunc")] = (s, cv[f"{kind}|{k}|{a:.2f}"](Hs))
                                rules[(kind, k, "plugin_rescaled_t")] = (
                                    s, tq / np.sqrt(np.clip(rho[f"{kind}|{k}"](Hh), 1e-12, None))
                                )
                                rules[(kind, k, "naive_half")] = (s, tq)
                                rules[(kind, k, "oracle_exact")] = (s, float(cv[f"{kind}|{k}|{a:.2f}"](H)))
                        for key, (s, c) in rules.items():
                            cnt[key + (a,)] = cnt.get(key + (a,), 0) + int(np.sum(s > c))
                dh = np.concatenate(dhs)
                for (proc, k, lab, a), c in sorted(cnt.items()):
                    lo, hi = wilson(c, Rtot)
                    rows.append([
                        model, H, n, proc, k or "", lab, a, Rtot, seed, c / Rtot, lo, hi,
                        float(dh.mean()), float(dh.std()), "MC",
                    ])
            print(f"n50_ewc H={H:.1f} {model} done ({time.perf_counter() - t_all:.1f}s)", flush=True)
    out = os.path.join(RES, "E3d_plugin_n50_ewc.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "H", "n", "procedure", "param", "rule", "alpha", "R", "seed", "mc_rate",
            "ci99_lo", "ci99_hi", "dhat_mean", "dhat_sd", "status",
        ])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {time.perf_counter() - t_all:.1f}s)", flush=True)


# ============================================================================ Part 2: Head-to-head competitor benchmark
def run_competitors(R=20_000, c_alt=2.8, alpha=0.05):
    cv, _, sncv = load_splines()
    z_norm = stats.norm.ppf(1.0 - alpha / 2.0)       # 1.9599639845...
    z_up = stats.norm.ppf(0.90)                      # one-sided 90% upper confidence bound on H
    t8_naive = stats.t.ppf(1.0 - alpha / 2.0, 8)     # HI m=4 short-memory cutoff
    sn_naive = SN_HALF[alpha]
    rows = []
    t_all = time.perf_counter()
    for n in (100, 500):
        mlw = int(math.floor(n ** 0.65))
        se_lw = 1.0 / (2.0 * math.sqrt(mlw))
        b_sub = int(round(n ** 0.60))                # 16 at n=100, 42 at n=500
        batch = 500 if n == 500 else 1000
        for H in (0.5, 0.7, 0.9):
            for model in MODELS:
                t0 = time.perf_counter()
                g = fn.make_acf(model_spec(model, H), n)
                vx = var_xbar(g)
                mu_alt = c_alt * math.sqrt(vx)
                m_id = {"fgn": 1, "arfima0": 2, "arfima1p": 3, "arfima1m": 4}[model]
                seed = 77000000 + int(100 * H) * 10000 + n + m_id * 100000
                cnt_sz, cnt_pw = {}, {}
                Rtot = 0
                d_lw_all, d_elw_all = [], []
                # Precompute cosine design matrix for EWC B=8
                t_idx = np.arange(1, n + 1)
                C_ewc = np.sqrt(2.0 / n) * np.cos(np.pi * np.outer(np.arange(1, 9), t_idx - 0.5) / n)
                for X in mc_batches(g, R, seed, batch=batch):
                    B_cur = X.shape[0]
                    Rtot += B_cur
                    # Mean-invariant quantities (identical under null X and shifted X + mu_alt)
                    d_lw, twopi_G = fn.local_whittle(X, mlw, return_G=True)
                    d_elw = fn.exact_local_whittle(X, mlw)
                    d_lw_all.append(d_lw)
                    d_elw_all.append(d_elw)
                    H_lw = np.clip(d_lw + 0.5, 0.01, 0.99)
                    H_elw = np.clip(d_elw + 0.5, 0.01, 0.99)
                    H_lw_sn = np.clip(H_lw, 0.50, 0.99)
                    H_up = np.clip(H_lw + z_up * se_lw, 0.01, 0.99)
                    H_up_sn = np.clip(H_up, 0.50, 0.99)

                    xbar0 = X.mean(axis=1)
                    xbar1 = xbar0 + mu_alt

                    # SN denominator
                    S_full = np.cumsum(X - xbar0[:, None], axis=1)
                    Cn = (S_full * S_full).sum(axis=1) / (n ** 2)
                    sn0 = n * (xbar0 ** 2) / Cn
                    sn1 = n * (xbar1 ** 2) / Cn

                    # HI m=4 denominator
                    I2 = fn.periodogram_fourier(X, 4)
                    se_hi = np.sqrt(I2.mean(axis=1) / n)
                    hi0 = np.abs(xbar0) / se_hi
                    hi1 = np.abs(xbar1) / se_hi

                    # EWC B=8 denominator
                    Lam = X @ C_ewc.T
                    se_ewc = np.sqrt((Lam ** 2).mean(axis=1) / n)
                    ewc0 = np.abs(xbar0) / se_ewc
                    ewc1 = np.abs(xbar1) / se_ewc

                    # ADG-MAC (2009) denominator (uses d_lw and twopi_G)
                    from scipy.special import gammaln
                    d_clip = np.clip(d_lw, -0.49, 0.49)
                    log_cf = gammaln(1.0 - 2.0 * d_clip) - np.log(2.0 * d_clip + 1.0) - gammaln(1.0 + d_clip) - gammaln(1.0 - d_clip)
                    se_mac = np.sqrt(np.maximum(twopi_G * np.exp(log_cf), 1e-300)) * (n ** (d_clip - 0.5))
                    mac0 = np.abs(xbar0) / se_mac
                    mac1 = np.abs(xbar1) / se_mac

                    # Rate-free self-normalised subsampling (centred blocks invariant to mu_alt)
                    Xc = X - xbar0[:, None]
                    nblk = n - b_sub + 1
                    s0, s1 = Xc.strides
                    W = np.lib.stride_tricks.as_strided(Xc, shape=(B_cur, nblk, b_sub), strides=(s0, s1, s1))
                    wbar = W.mean(axis=2)
                    Sw = np.cumsum(W - wbar[:, :, None], axis=2)
                    Cb = (Sw * Sw).sum(axis=2) / (b_sub ** 2)
                    Tb = b_sub * (wbar ** 2) / np.maximum(Cb, 1e-300)
                    p_sub0 = np.mean(Tb >= sn0[:, None], axis=1)
                    p_sub1 = np.mean(Tb >= sn1[:, None], axis=1)

                    # Critical values
                    c_sn_lw = sncv[alpha](H_lw_sn)
                    c_sn_up = sncv[alpha](H_up_sn)
                    # every feasible rule truncates the memory estimate to [1/2, 0.99], as SN and Section 4.3 do (21 Nov)
                    c_hi_lw = cv["HI|4|0.05"](H_lw_sn)
                    c_hi_elw = cv["HI|4|0.05"](np.clip(H_elw, 0.50, 0.99))
                    c_hi_up = cv["HI|4|0.05"](H_up_sn)
                    c_ewc_lw = cv["EWC|8|0.05"](H_lw_sn)

                    decisions = {
                        "SN_CosCert_LW": (sn0 > c_sn_lw, sn1 > c_sn_lw),
                        "SN_CosCert_UpperCI": (sn0 > c_sn_up, sn1 > c_sn_up),
                        "HI4_CosCert_LW": (hi0 > c_hi_lw, hi1 > c_hi_lw),
                        "HI4_CosCert_ELW": (hi0 > c_hi_elw, hi1 > c_hi_elw),
                        "HI4_CosCert_UpperCI": (hi0 > c_hi_up, hi1 > c_hi_up),
                        "EWC8_CosCert_LW": (ewc0 > c_ewc_lw, ewc1 > c_ewc_lw),
                        "ADG_MAC_LW": (mac0 > z_norm, mac1 > z_norm),
                        "Subsampling_SN": (p_sub0 < alpha, p_sub1 < alpha),
                        "Naive_SN_half": (sn0 > sn_naive, sn1 > sn_naive),
                        "Naive_HI4_t8": (hi0 > t8_naive, hi1 > t8_naive),
                    }
                    for name, (r0, r1) in decisions.items():
                        cnt_sz[name] = cnt_sz.get(name, 0) + int(np.sum(r0))
                        cnt_pw[name] = cnt_pw.get(name, 0) + int(np.sum(r1))

                dlw = np.concatenate(d_lw_all)
                delw = np.concatenate(d_elw_all)
                for name in cnt_sz:
                    k0, k1 = cnt_sz[name], cnt_pw[name]
                    lo0, hi0_ = wilson(k0, Rtot)
                    lo1, hi1_ = wilson(k1, Rtot)
                    rows.append([
                        model, H, n, name, alpha, c_alt, mlw, b_sub, Rtot, seed,
                        k0 / Rtot, lo0, hi0_, k1 / Rtot, lo1, hi1_,
                        float(dlw.mean()), float(dlw.std()), float(delw.mean()), float(delw.std()), "MC",
                    ])
                print(
                    f"comp n={n} H={H:.1f} {model:8s}: HI_LW={100*cnt_sz['HI4_CosCert_LW']/Rtot:5.2f}/{100*cnt_pw['HI4_CosCert_LW']/Rtot:5.1f}  "
                    f"HI_ELW={100*cnt_sz['HI4_CosCert_ELW']/Rtot:5.2f}/{100*cnt_pw['HI4_CosCert_ELW']/Rtot:5.1f}  "
                    f"HI_Up={100*cnt_sz['HI4_CosCert_UpperCI']/Rtot:5.2f}/{100*cnt_pw['HI4_CosCert_UpperCI']/Rtot:5.1f}  "
                    f"ADG={100*cnt_sz['ADG_MAC_LW']/Rtot:5.2f}/{100*cnt_pw['ADG_MAC_LW']/Rtot:5.1f}  "
                    f"Sub={100*cnt_sz['Subsampling_SN']/Rtot:5.2f}/{100*cnt_pw['Subsampling_SN']/Rtot:5.1f}  ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
    out = os.path.join(RES, "E3d_competitors.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "model", "H", "n", "method", "alpha", "c_alt", "m_lw", "b_sub", "R", "seed",
            "size_rate", "size_ci99_lo", "size_ci99_hi", "power_rate", "power_ci99_lo", "power_ci99_hi",
            "d_lw_mean", "d_lw_sd", "d_elw_mean", "d_elw_sd", "status",
        ])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {time.perf_counter() - t_all:.1f}s)", flush=True)


# ============================================================================ Part 3: Exact HI (2017) Tables 3-5 oracle
def run_hi2017_exact_oracle():
    cv, _, _ = load_splines()
    rows = []
    specs = [("3", 0.0, 0.0), ("4", -0.3, 0.5), ("5", 0.3, -0.5)]
    for tab, delta, phi in specs:
        H = 0.5 + delta
        for T in (64, 256, 1024):
            g = fn.make_acf(("arfima1", phi, delta), T)
            for m in range(1, 17):
                lrk = fn.LowRankExact(g, "HI", m)
                c_or = float(cv[f"HI|{m}|0.05"](H))
                c_t = stats.t.ppf(0.975, 2 * m)
                r_or = float(lrk.reject_two_sided(c_or))
                r_t = float(lrk.reject_two_sided(c_t))
                rows.append([f"HI2017 Table {tab}", delta, phi, T, m, c_or, r_or, c_t, r_t, "exact"])
    out = os.path.join(RES, "E3d_hi2017_exact_oracle.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "delta", "phi", "T", "m", "cv_oracle", "exact_oracle_rate", "cv_t2m", "exact_t2m_rate", "status"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("hi2017_oracle", "all"):
        run_hi2017_exact_oracle()
    if cmd in ("competitors", "all"):
        run_competitors()
    if cmd in ("n50_ewc", "all"):
        run_n50_ewc()
