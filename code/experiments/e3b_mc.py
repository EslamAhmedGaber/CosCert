"""Step 5, part B -- Monte Carlo (every number produced here is labelled 'MC', with replications R and seed).

  validate : AT4 -- MC rejection rates of SN, HI (m = 1, 4, 16) and EWC (B = 4, 16) at the SAME critical values used by
             the exact finite-n computation (E3 csv), with Wilson 99 % intervals; pass if the exact rate lies inside.
  hi2017   : reproduction of Hualde & Iacone (2017) Tables 3-5: Gaussian ARFIMA(1, delta, 0), delta estimated by local
             Whittle with bandwidth floor(T^0.65), plug-in two-sided 5 % critical values; m = 1..16, T = 64/256/1024.
             Two plug-in rules: 'exact' (our exact fixed-m limit quantile at the estimate, continuous in delta) and
             'hi_table' (linear interpolation of HI's printed Table 1 in delta, as the closest reading of their rule).
  plugin   : feasible procedures on the paper's design grid (fgn, arfima0, arfima1p, arfima1m; H = .5..9; n = 100..2000):
             SN (certified T_H table interpolated in H; estimate clipped to [0.50, 0.99]), HI m and EWC B with exact plug-in
             critical values, HI m / EWC B with the closed-form rescaled-t rule t / sqrt(rho(H_hat)), and the naive H = 1/2 values.
usage: python e3b_mc.py validate <model> | hi2017 <table> | plugin <model> <H>
"""
import csv, json, math, os, sys, time
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import finite_n as fn, lowrank as lr, gram as gr  # noqa: E402

RES = os.path.join(WORK, "results")
ALPHAS = (0.10, 0.05, 0.01)
HGRID = np.round(np.arange(0.01, 0.995, 0.01), 2)
HI_M = list(range(1, 17)); EWC_B = [4, 8, 16, 32]
SN_HALF = {0.10: 28.330921323509, 0.05: 45.526085752448, 0.01: 100.34563942809}
BATCH = 4000                      # complex draws per batch -> 2 * BATCH series


def model_spec(name, H):
    d = H - 0.5
    return {"fgn": ("fgn", H), "arfima0": ("arfima0", d), "arfima1p": ("arfima1", 0.5, d), "arfima1m": ("arfima1", -0.5, d)}[name]


def exp_den(H, kind, k):
    if kind == "SN":
        return 1.0 / ((2 * H + 1) * (H + 1)) - 1.0 / 6.0
    Sg = gr.hi_fixed_m_cov(H, k) if kind == "HI" else gr.ewc_cov(H, k)
    return float(np.trace(np.asarray(Sg)[1:, 1:])) / k


def fixed_grid():
    """Exact fixed-smoothing limit quantiles q(H) (two-sided, t-scale) and rho(H) on H = .01..99 (double precision)."""
    path = os.path.join(RES, "E3b_fixed_cv_grid.json")
    if os.path.exists(path):
        return json.load(open(path))
    out = {"H": HGRID.tolist(), "cv": {}, "rho": {}}
    for h in HGRID:
        ga = lr.GramArb(f"{h:.2f}", 33)          # arb Gram: valid for every 0 < H < 1 (analytic continuation below 1/2)
        for kind, ks in (("HI", HI_M), ("EWC", EWC_B)):
            for k in ks:
                law = lr.FixedSmoothingLaw(f"{h:.2f}", kind, k, ga=ga)
                # E_{1/2}[denominator] = 1 for HI and EWC, so rho(H) = E_H[denominator] = tr(Sigma_components) / k
                out["rho"].setdefault(f"{kind}|{k}", []).append(float(np.trace(np.asarray(law.Sd, float)[1:, 1:])) / k)
                for a in ALPHAS:
                    out["cv"].setdefault(f"{kind}|{k}|{a:.2f}", []).append(math.sqrt(law.quantile2(1 - a)))
        print(f"grid H={h:.2f}", flush=True)
    json.dump(out, open(path, "w")); return out


def splines(grid):
    H = np.array(grid["H"])
    cv = {k: CubicSpline(H, np.array(v)) for k, v in grid["cv"].items()}
    rho = {k: CubicSpline(H, np.array(v)) for k, v in grid["rho"].items()}
    return cv, rho


def sn_table_spline():
    tab = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))
    Hs = sorted(float(h) for h in tab["cells"])
    if Hs[-1] < 0.99 - 1e-9:
        raise SystemExit("E2_sn_critical_values.json does not reach H = 0.99 yet (merge pending) -- not running SN plug-in")
    return {a: CubicSpline(Hs, [tab["cells"][f"{h:.2f}"][f"{1 - a:.3f}"]["q"] for h in Hs]) for a in ALPHAS}


def wilson(k, n, z=2.5758):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - hw, c + hw


def mc_batches(g, R, seed, batch=BATCH):
    smp = fn.CirculantSampler(g); rng = np.random.Generator(np.random.PCG64(seed)); done = 0
    while done < R:
        nb = min(batch, (R - done + 1) // 2)
        X = smp.sample(nb, rng); X = X[: R - done]
        done += X.shape[0]
        yield X


def part_validate(model, R=200_000):
    import pandas as pd
    E3 = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in os.listdir(RES) if f.startswith(f"E3_rates_{model}_")])
    rows = []
    for H in (0.6, 0.8):
        for n in (50, 500):
            sel = E3[(np.isclose(E3.H, H)) & (E3.n == n) & (E3.alpha == 0.05) & (E3.cv_label.isin(["own_limit", "naive_half"]))]
            want = [("SN", None), ("HI", 1), ("HI", 4), ("HI", 16), ("EWC", 4), ("EWC", 16)]
            cells = []
            for proc, k in want:
                for lab in ("own_limit", "naive_half"):
                    s = sel[(sel.procedure == proc) & (sel.cv_label == lab) & ((pd.to_numeric(sel.param, errors="coerce") == k) if k else True)]
                    if len(s):
                        cells.append((proc, k, lab, float(s.cv.iloc[0]), float(s.rejection_rate.iloc[0])))
            seed = int(1e6 * H) + 1000 * n + {"fgn": 1, "arfima0": 2, "arfima1p": 3, "arfima1m": 4}[model]
            cnt = np.zeros(len(cells)); t0 = time.perf_counter(); Rtot = 0
            for X in mc_batches(fn.make_acf(model_spec(model, H), n), R, seed):
                st = fn.stats_from_samples(X, m_list=(1, 4, 16), B_list=(4, 16)); Rtot += X.shape[0]
                for i, (proc, k, lab, cv, ex) in enumerate(cells):
                    cnt[i] += np.sum(st["SN"] > cv) if proc == "SN" else np.sum(np.abs(st[(proc, k)]) > cv)
            for i, (proc, k, lab, cv, ex) in enumerate(cells):
                lo, hi = wilson(cnt[i], Rtot); p = cnt[i] / Rtot
                rows.append([model, H, n, proc, k or "", lab, cv, 0.05, Rtot, seed, p, ex, (p - ex) / math.sqrt(ex * (1 - ex) / Rtot), lo, hi,
                             bool(lo <= ex <= hi), "MC"])
            print(f"validate {model} H={H} n={n}: {time.perf_counter() - t0:.0f}s  max|z|={max(abs(r[12]) for r in rows[-len(cells):]):.2f}", flush=True)
    out = os.path.join(RES, f"E3b_validate_{model}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "H", "n", "procedure", "param", "cv_label", "cv", "alpha", "R", "seed", "mc_rate", "exact_rate", "z",
                                        "ci99_lo", "ci99_hi", "exact_inside_ci99", "status"]); w.writerows(rows)
    print("wrote", out)


def hi_table1_interp():
    import pandas as pd
    led = pd.read_csv(os.path.join(WORK, "literature", "ledger_critical_values.csv"))
    sel = led[led.source_id == "HI2017"].copy()
    sel["m"] = sel.ledger_id.astype(str).str.rsplit("-m", n=1).str[1].astype(int); sel["d"] = sel.memory_value.astype(float)
    return {m: (np.array(g.sort_values("d").d), np.array(g.sort_values("d").published_value.astype(float))) for m, g in sel.groupby("m")}


def part_hi2017(table, R=100_000):
    spec = {"3": (0.0, 0.0), "4": (-0.3, 0.5), "5": (0.3, -0.5)}[table]
    delta, phi = spec; H = 0.5 + delta
    cv, rho = splines(fixed_grid()); hit = hi_table1_interp()
    rows = []
    for T in (64, 256, 1024):
        g = fn.make_acf(("arfima1", phi, delta), T); mlw = int(math.floor(T ** 0.65))
        seed = 20170000 + 1000 * int(table) + T
        cnt = {}; Rtot = 0; t0 = time.perf_counter(); dhat_all = []
        # default batch: the stored E3b_hi2017_table*.csv were produced with it, so a re-run reproduces them bit for bit
        # (fix 16 Nov: a copied memory-cap line referred to the plug-in loop variable n, undefined here)
        batch = BATCH
        for X in mc_batches(g, R, seed, batch):
            dh = fn.local_whittle(X, mlw); dhat_all.append(dh); Hh = np.clip(dh + 0.5, 0.01, 0.99)
            st = fn.stats_from_samples(X, m_list=tuple(HI_M)); Rtot += X.shape[0]
            for m in HI_M:
                a = np.abs(st[("HI", m)])
                c_ex = cv[f"HI|{m}|0.05"](Hh); dd, vv = hit[m]; c_tab = np.interp(dh, dd, vv)
                c_or = float(cv[f"HI|{m}|0.05"](H)); c_t = stats.t.ppf(0.975, 2 * m)
                c_rs = c_t / np.sqrt(np.clip(rho[f"HI|{m}"](Hh), 1e-12, None))
                for lab, c in (("plugin_exact", c_ex), ("plugin_hi_table", c_tab), ("oracle_exact", c_or), ("naive_t2m", c_t), ("plugin_rescaled_t", c_rs)):
                    cnt[(m, lab)] = cnt.get((m, lab), 0) + int(np.sum(a > c))
        dh = np.concatenate(dhat_all)
        for (m, lab), k in sorted(cnt.items()):
            lo, hi = wilson(k, Rtot)
            rows.append([f"HI2017 Table {table}", delta, phi, T, m, lab, 0.05, Rtot, seed, k / Rtot, lo, hi, float(dh.mean()), float(dh.std()), "MC"])
        print(f"HI Table {table} T={T}: {time.perf_counter() - t0:.0f}s  mean dhat={dh.mean():.3f} sd={dh.std():.3f}", flush=True)
    out = os.path.join(RES, f"E3b_hi2017_table{table}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["design", "delta", "phi", "T", "m", "rule", "alpha", "R", "seed", "mc_rate", "ci99_lo", "ci99_hi",
                                        "dhat_mean", "dhat_sd", "status"]); w.writerows(rows)
    print("wrote", out)


def part_plugin(model, H, R=40_000, bw=0.65):
    cv, rho = splines(fixed_grid()); sncv = sn_table_spline()
    rows = []
    for n in (100, 200, 500, 1000, 2000):
        g = fn.make_acf(model_spec(model, H), n); mlw = int(math.floor(n ** bw))   # local Whittle bandwidth (production 0.65)
        seed = 31000000 + int(100 * H) * 10000 + n + {"fgn": 1, "arfima0": 2, "arfima1p": 3, "arfima1m": 4}[model] * 1000000
        cnt = {}; Rtot = 0; t0 = time.perf_counter(); dhs = []
        # memory cap (16 Nov, host RAM ~2 GiB free): at most 2e6 complex draws x n per batch; affects only this mode's stream
        batch = max(100, min(BATCH, int(2_000_000 // n)))
        for X in mc_batches(g, R, seed, batch):
            dh = fn.local_whittle(X, mlw); dhs.append(dh); Hh = np.clip(dh + 0.5, 0.01, 0.99); Hs = np.clip(Hh, 0.50, 0.99)
            # EWC B = 8, 32 added 21 Nov (pairing B = 2m with HI m = 4, 16); same streams, so earlier columns are unchanged
            st = fn.stats_from_samples(X, m_list=(1, 4, 16), B_list=(4, 8, 16, 32)); Rtot += X.shape[0]
            for a in ALPHAS:
                rules = {("SN", 0, "plugin_exact"): (st["SN"], sncv[a](Hs)), ("SN", 0, "naive_half"): (st["SN"], SN_HALF[a]),
                         ("SN", 0, "oracle_exact"): (st["SN"], float(sncv[a](max(H, 0.5))))}
                for kind, ks, dof in (("HI", (1, 4, 16), 2), ("EWC", (4, 8, 16, 32), 1)):
                    for k in ks:
                        s = np.abs(st[(kind, k)]); tq = stats.t.ppf(1 - a / 2, dof * k)
                        rules[(kind, k, "plugin_exact")] = (s, cv[f"{kind}|{k}|{a:.2f}"](Hh))
                        # same truncation as the SN rule (H-hat clipped to [1/2, 0.99]): long memory assumed a priori (v2, 16 Nov)
                        rules[(kind, k, "plugin_exact_trunc")] = (s, cv[f"{kind}|{k}|{a:.2f}"](Hs))
                        rules[(kind, k, "plugin_rescaled_t")] = (s, tq / np.sqrt(np.clip(rho[f"{kind}|{k}"](Hh), 1e-12, None)))
                        rules[(kind, k, "naive_half")] = (s, tq)
                        rules[(kind, k, "oracle_exact")] = (s, float(cv[f"{kind}|{k}|{a:.2f}"](H)))
                for key, (s, c) in rules.items():
                    cnt[key + (a,)] = cnt.get(key + (a,), 0) + int(np.sum(s > c))
        dh = np.concatenate(dhs)
        for (proc, k, lab, a), c in sorted(cnt.items()):
            lo, hi = wilson(c, Rtot)
            rows.append([model, H, n, proc, k or "", lab, a, Rtot, seed, c / Rtot, lo, hi, float(dh.mean()), float(dh.std()), "MC"])
        print(f"plugin {model} H={H} n={n}: {time.perf_counter() - t0:.0f}s  dhat {dh.mean():.3f}+-{dh.std():.3f}", flush=True)
    # same seeds for every bandwidth (common random numbers): differences between bandwidths are positively correlated
    # estimates, so their Monte Carlo error is smaller than for independent runs, but it is not zero
    out = os.path.join(RES, f"E3b_plugin_v2_{model}_H{H:.1f}.csv" if bw == 0.65 else f"E3b_plugin_bw{bw:.2f}_{model}_H{H:.1f}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model", "H", "n", "procedure", "param", "rule", "alpha", "R", "seed", "mc_rate", "ci99_lo", "ci99_hi",
                                        "dhat_mean", "dhat_sd", "status"]); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    part = sys.argv[1]
    if part == "validate":
        part_validate(sys.argv[2])
    elif part == "hi2017":
        part_hi2017(sys.argv[2])
    elif part == "grid":
        fixed_grid()
    elif part == "plugin_bw":
        part_plugin(sys.argv[2], float(sys.argv[3]), bw=float(sys.argv[4]))
    else:
        part_plugin(sys.argv[2], float(sys.argv[3]))
