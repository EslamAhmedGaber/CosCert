"""Write manuscript/tables/p3_numbers.tex (every number in the text, as \\pthreedef{key}{value}) and every table body
manuscript/tables/tab_*.tex, from work/results and work/literature only.  Missing inputs leave keys undefined (reported);
nothing is typed by hand.  Usage: python make_numbers.py
"""
import os, sys, json, glob, math, re
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
RES = os.path.join(WORK, "results"); LIT = os.path.join(WORK, "literature"); TAB = os.path.join(WORK, "manuscript", "tables")
os.makedirs(TAB, exist_ok=True)
NUM, MISSING, CHECKS = {}, [], []


# ------------------------------------------------------------------ formatting
def put(key, val):
    NUM[key] = val


def fixed(x, d):
    return f"{x:.{d}f}"


def sci(x, d=1):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    if round(abs(m), d) >= 10:
        m /= 10; e += 1
    return rf"${m:.{d}f}\times10^{{{e}}}$"


def grouped(n):
    return f"{int(n):,}".replace(",", r"\,")


def sig_str(x, s):
    if x == 0:
        return "0"
    dec = s - 1 - int(math.floor(math.log10(abs(x))))
    return f"{x:.{max(dec, 0)}f}" if dec > 0 else f"{round(x, dec):.0f}"


def cert_round(q, lo, hi, smax=6, smin=3):
    """Largest s <= smax such that lo and hi round to the same s-significant-digit value; returns that string."""
    for s in range(smax, smin - 1, -1):
        if sig_str(lo, s) == sig_str(hi, s):
            return sig_str(q, s)
    return sig_str(q, smin) + r"^{\ast}"


def merged_json(pattern, sub=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        d = json.load(open(f))
        for k, v in d.items():
            out[k] = v[sub] if sub else v
    return out


def need(cond, key):
    if not cond:
        MISSING.append(key)
    return cond


# ------------------------------------------------------------------ load
SNF = os.path.join(RES, "E2_sn_critical_values.json")
SN = json.load(open(SNF))["cells"] if os.path.exists(SNF) else {}
FX = merged_json("E2b_fixed_*.json", "cells")
H_SN = sorted(SN, key=float)

# ------------------------------------------------------------------ section 3: SN critical values
if need(SN, "SN table"):
    put("sn-n-H", str(len(H_SN))); put("sn-n-cells", str(sum(len(v) for v in SN.values())))
    rw = max((c["q_hi"] - c["q_lo"]) / (2 * c["q"]) for v in SN.values() for c in v.values())
    put("sn-max-relwidth", sci(rw, 1))
    # certified size error of the printed values and of the brackets (17 Nov, audit C2: quantify "error of the critical value")
    pe = max(max(abs(c["size_at_print"][0] - (1 - float(p))), abs(c["size_at_print"][1] - (1 - float(p))))
             for v in SN.values() for p, c in v.items())
    be = max(max(c["F_at_q_hi"][1] - float(p), float(p) - c["F_at_q_lo"][0]) for v in SN.values() for p, c in v.items())
    put("sn-print-maxerr", fixed(100 * pe, 3) if pe >= 1e-5 else sci(100 * pe, 1)); put("sn-bracket-maxerr", sci(100 * be, 1))
    for h in H_SN:
        for p, c in SN[h].items():
            put(f"sn-q-{h}-{p}", cert_round(c["q"], c["q_lo"], c["q_hi"]))
    q05 = SN["0.50"]["0.950"]["q"]
    for h in ("0.80", "0.90", "0.95", "0.99"):
        if h in SN:
            put(f"sn-infl-{h}-0.950", sig_str(SN[h]["0.950"]["q"] / q05, 3))
            put(f"sn-tinfl-{h}-0.950", sig_str(math.sqrt(SN[h]["0.950"]["q"] / q05), 3))
    for h in ("0.99",):
        need(h in SN, f"sn-q-{h}-*")
if need(FX, "fixed table"):
    put("fx-n-cells", grouped(sum(1 for v in FX.values() for c in v.values() if c.get("status") == "certified")))

E0F = os.path.join(RES, "E0_anchors.json")
if need(os.path.exists(E0F), "E0"):
    e0 = json.load(open(E0F)); dg = [v["sig_digits"] for v in e0["a"].values()]
    put("g0-min-digits", str(int(math.floor(min(dg))))); put("g0-max-digits", str(int(math.floor(max(dg)))))

# ------------------------------------------------------------------ naive and rescaled (E6, E6b)
NSN = merged_json("E6_naive_sn_*.json"); NFX = merged_json("E6_naive_fixed_*.json")
RFX = merged_json("E6b_rescaled_fixed_*.json"); RSN = merged_json("E6b_rescaled_sn_*.json")
if need(NSN and NFX, "E6"):
    for h in ("0.70", "0.90"):
        if h in NSN:
            put(f"naive-sn-{h}-0.05", fixed(100 * NSN[h]["0.05"]["size_computed"], 1))
    for m in (1, 4, 16):
        put(f"naive-hi{m}-0.90", fixed(100 * NFX["0.90"][f"HI|{m}|0.05"]["size_computed"], 1))
    vals = [NSN["0.90"]["0.05"]["size_computed"]] + [NFX["0.90"][f"{k}|0.05"]["size_computed"] for k in
                                                     ("HI|1", "HI|4", "HI|16", "EWC|2", "EWC|8", "EWC|32")]
    put("naive-min-0.90", fixed(100 * min(vals), 1)); put("naive-max-0.90", fixed(100 * max(vals), 1))
if need(RFX, "E6b fixed"):
    def rs_err(hmax, a):
        e = 0.0
        for h, rec in RFX.items():
            if float(h) <= hmax + 1e-9:
                for k, c in rec.items():
                    if k.endswith(f"|{a:.2f}"):
                        e = max(e, abs(c["size_lo"] - a), abs(c["size_hi"] - a))
        return 100 * e
    for a in (0.10, 0.05, 0.01):
        put(f"rs-maxerr-090-{a:.2f}", fixed(rs_err(0.90, a), 2))
    put("rs-maxerr-099-0.05", fixed(rs_err(0.99, 0.05), 2))
    # table: max deviation by H band, family and level
    bands = [(0.50, 0.70), (0.71, 0.80), (0.81, 0.90), (0.91, 0.99)]
    lines = [r"\begin{tabular}{@{}lcccccc@{}}\toprule", r" & \multicolumn{3}{c}{HI, $m\le16$} & \multicolumn{3}{c}{EWC, tabulated $B$}\\",
             r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", r"$H$ & $10\,\%$ & $5\,\%$ & $1\,\%$ & $10\,\%$ & $5\,\%$ & $1\,\%$\\\midrule"]
    for lo, hi in bands:
        row = [f"{lo:.2f}--{hi:.2f}"]
        for fam in ("HI", "EWC"):
            for a in (0.10, 0.05, 0.01):
                e = max(max(abs(c["size_lo"] - a), abs(c["size_hi"] - a)) for h, rec in RFX.items() if lo - 1e-9 <= float(h) <= hi + 1e-9
                        for k, c in rec.items() if k.startswith(fam + "|") and k.endswith(f"|{a:.2f}"))
                row.append(fixed(100 * e, 2))
        lines.append(" & ".join(row) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(TAB, "tab_rescaled.tex"), "w").write("\n".join(lines) + "\n")
if need(RSN, "E6b sn"):
    for h in ("0.75", "0.95"):
        if h in RSN:
            c = RSN[h]["0.05"]; put(f"rs-sn-{h}-0.05", fixed(100 * 0.5 * (c["size_lo"] + c["size_hi"]), 2))

SURF = os.path.join(RES, "E9_surface.json")
if need(os.path.exists(SURF), "E9 surface"):
    sf = json.load(open(SURF)); K9 = int(sf["K"])
    put("rs-surface-K", str(K9)); put("rs-surface-ncoef", str(K9 + 1))
    put("rs-surface-fit-maxerr", sci(sf["max_heldout_fit_err_pp"], 1)); put("rs-surface-guar-maxerr", sci(sf["max_heldout_guaranteed_err_pp"], 1))
    # printed digits: the fewest significant digits whose rounding moves the held-out size by < 1e-5 pp at every level
    Hh = np.array([float(h) for h in H_SN if round(100 * float(h)) % 2 == 1])
    def size_shift(p, coef_r, coef):
        x = Hh - 0.5; f = np.array([SN[f"{h:.2f}"][p]["density"] for h in Hh])
        q1 = np.exp(np.polyval(np.array(coef_r)[::-1], x)) / (1 - Hh); q0 = np.exp(np.polyval(np.array(coef)[::-1], x)) / (1 - Hh)
        return float(np.max(100 * f * np.abs(q1 - q0)))
    s_need = None
    for s_ in range(5, 17):
        if all(size_shift(p, [float(f"{a:.{s_ - 1}e}") for a in rec["coef"][:K9 + 1]], rec["coef"][:K9 + 1]) < 1e-5 for p, rec in sf["fits"].items()):
            s_need = s_; break
    put("rs-surface-digits", str(s_need))
    CHECKS.append((f"response-surface coefficients printed to {s_need} significant digits move the size by < 1e-5 pp", s_need is not None))
    ps = list(sf["fits"])
    lines = [r"\begin{tabular}{@{}l" + "r" * len(ps) + r"@{}}\toprule",
             r"& \multicolumn{" + str(len(ps)) + r"}{c}{level $p$ (two-sided $\alpha=1-p$)}\\\cmidrule(l){2-" + str(len(ps) + 1) + "}",
             "coefficient & " + " & ".join(f"${float(p):g}$" for p in ps) + r"\\\midrule"]
    for i in range(K9 + 1):
        lines.append(f"$a_{{p,{i}}}$ & " + " & ".join(f"{sf['fits'][p]['coef'][i]:.{s_need - 1}e}".replace("e", r"\,\mathrm{e}") .join(["$", "$"]) if False else
                                                     f"${float(f'{sf['fits'][p]['coef'][i]:.{s_need - 1}e}'):.{s_need}g}$" for p in ps) + r"\\")
    lines.append(r"\midrule fit error (pp)\textsuperscript{a} & " + " & ".join(sci(sf["fits"][p]["max_heldout_fit_err_pp"], 1) for p in ps) + r"\\")
    SFC = os.path.join(RES, "E9_surface_certified.json")
    if need(os.path.exists(SFC), "E9b certified surface"):
        sc = json.load(open(SFC))
        assert int(sc["digits"]) == int(s_need), "certified surface used different printed digits"
        put("rs-surface-cert-maxerr", sci(sc["max_bound_pp"], 1)); put("rs-surface-cert-argH", sc["argmax"][0])
        put("rs-surface-cert-ndirect", str(sc["n_direct"])); put("rs-surface-cert-ncells", str(len(sc["cells"])))
        lines.append(r"certified bound (pp)\textsuperscript{b} & " + " & ".join(sci(sc["max_bound_pp_by_p"][p], 1) for p in ps) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    open(os.path.join(TAB, "tab_surface.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ E12: limit as H -> 1 (Proposition edge)
E12f = os.path.join(RES, "E12_edge_limit.json")
if need(os.path.exists(E12f), "E12"):
    e12 = json.load(open(E12f))
    for p_ in ("0.900", "0.950", "0.990"):
        put(f"edge-C-{p_}", sig_str(float(e12["C_p"][p_]), 4))
    put("edge-extrap-maxrel", fixed(100 * max(e12["extrap_rel_dev"].values()), 2))
    put("edge-rho-min", fixed(float(e12["rho_over_1mH_min"]), 2)); put("edge-mu1-share", f"{100 * e12['mu1_share']:.0f}")
    CHECKS.append(("E12 trace of k1 equals 7/36 to 1e-10", abs(e12["trace_k1_numeric"] - 7 / 36) < 1e-10))

# ------------------------------------------------------------------ tables of critical values
if SN:
    rows_short = [h for h in ("0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95", "0.99") if h in SN]
    P_SHORT = ("0.800", "0.900", "0.950", "0.990")
    L = [r"\begin{tabular}{@{}lrrrr@{}}\toprule", r"$H$ & $q_{0.80}$ & $q_{0.90}$ & $q_{0.95}$ & $q_{0.99}$\\\midrule"]
    for h in rows_short:
        L.append(h + " & " + " & ".join(cert_round(SN[h][p]["q"], SN[h][p]["q_lo"], SN[h][p]["q_hi"]) for p in P_SHORT) + r"\\")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_sn_cv_short.tex"), "w").write("\n".join(L) + "\n")
    P_ALL = ("0.800", "0.900", "0.950", "0.975", "0.980", "0.990")
    L = [r"\begin{longtable}{@{}lrrrrrr@{}}", r"\caption{Quantiles $q_p(H)$ of $T_H$, all values (certified).}\label{tab:sncvfull}\\\toprule",
         r"$H$ & $q_{0.80}$ & $q_{0.90}$ & $q_{0.95}$ & $q_{0.975}$ & $q_{0.98}$ & $q_{0.99}$\\\midrule\endfirsthead",
         r"\caption[]{(continued)}\\\toprule $H$ & $q_{0.80}$ & $q_{0.90}$ & $q_{0.95}$ & $q_{0.975}$ & $q_{0.98}$ & $q_{0.99}$\\\midrule\endhead",
         r"\midrule\multicolumn{7}{r@{}}{\footnotesize\itshape continued on the next page}\\\endfoot", r"\bottomrule\endlastfoot"]
    for h in H_SN:
        L.append(h + " & " + " & ".join(cert_round(SN[h][p]["q"], SN[h][p]["q_lo"], SN[h][p]["q_hi"]) if p in SN[h] else "--" for p in P_ALL) + r"\\")
    L.append(r"\end{longtable}"); open(os.path.join(TAB, "tab_sn_cv_full.tex"), "w").write("\n".join(L) + "\n")
if FX:
    def fxv(h, kind, k, a="0.05"):
        c = FX[h][f"{kind}|{k}|{float(a)}"] if f"{kind}|{k}|{float(a)}" in FX[h] else FX[h][f"{kind}|{k}|{a}"]
        return cert_round(c["x"], c["x_lo"], c["x_hi"], smax=5)
    HS = [h for h in ("0.50", "0.60", "0.70", "0.80", "0.90", "0.95", "0.99") if h in FX]
    L = [r"\begin{tabular}{@{}l" + "r" * 10 + r"@{}}\toprule",
         r" & \multicolumn{5}{c}{HI, $m$} & \multicolumn{5}{c}{EWC, $B$}\\\cmidrule(lr){2-6}\cmidrule(l){7-11}",
         r"$H$ & 1 & 2 & 4 & 8 & 16 & 2 & 4 & 8 & 16 & 32\\\midrule"]
    for h in HS:
        L.append(h + " & " + " & ".join([fxv(h, "HI", m) for m in (1, 2, 4, 8, 16)] + [fxv(h, "EWC", B) for B in (2, 4, 8, 16, 32)]) + r"\\")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_fixed_cv_short.tex"), "w").write("\n".join(L) + "\n")
    HF = [h for h in sorted(FX, key=float) if round(float(h) * 100) % 5 == 0 or h == "0.99"]
    L = []
    for kind, ks, title in (("HI", (1, 2, 3, 4, 6, 8, 12, 16), r"fixed-$m$ statistic (HI), $m$ frequencies"),
                            ("EWC", (1, 2, 4, 8, 16, 32), r"EWC statistic, $B$ cosines")):
        L += [r"\begin{tabular}{@{}l" + "r" * len(ks) + r"@{}}\toprule",
              r" & \multicolumn{" + str(len(ks)) + "}{c}{" + title + r"}\\\cmidrule(l){2-" + str(len(ks) + 1) + "}",
              r"$H$ & " + " & ".join(str(k) for k in ks) + r"\\\midrule"]
        for h in HF:
            L.append(h + " & " + " & ".join(fxv(h, kind, k) for k in ks) + r"\\")
        L.append(r"\bottomrule\end{tabular}")
        if kind == "HI":
            L.append(r"\par\medskip")
    open(os.path.join(TAB, "tab_fixed_cv_full.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ section 4: finite samples (E3, E3b, E3c)
E3f = sorted(glob.glob(os.path.join(RES, "E3_rates_*.csv")))
if need(E3f, "E3"):
    E3 = pd.concat([pd.read_csv(f) for f in E3f], ignore_index=True)
    E3["k"] = pd.to_numeric(E3.param, errors="coerce").fillna(0).astype(int); E3["size"] = 100 * E3.rejection_rate
    O = E3[(E3.cv_label == "own_limit") & np.isclose(E3.alpha, 0.05)]
    def fsz(model, proc, k, H, n):
        s = O[(O.model == model) & (O.procedure == proc) & (O.k == k) & np.isclose(O.H, H) & (O.n == n)]
        return float(s["size"].iloc[0])
    put("fgn-sn-0.70-500-dev", sci(abs(fsz("fgn", "SN", 0, 0.7, 500) - 5.0), 1))
    put("fn-maxdev-pure", fixed(O[O.model.isin(["fgn", "arfima0"])]["size"].sub(5).abs().max(), 2))
    put("fn-maxdev-2000", fixed(O[O.n == 2000]["size"].sub(5).abs().max(), 2))
    small = O[(((O.procedure == "HI") & (O.k <= 4)) | ((O.procedure == "EWC") & (O.k <= 8))) & (O.n >= 100)]
    put("fn-maxdev-small-100", fixed(small["size"].sub(5).abs().max(), 2))
    for model, proc, k, H, n in (("arfima1p", "HI", 16, 0.5, 50), ("arfima1p", "HI", 16, 0.9, 50), ("arfima1p", "SN", 0, 0.5, 50),
                                 ("arfima1p", "HI", 1, 0.5, 50), ("arfima1m", "HI", 16, 0.5, 50), ("arfima1p", "HI", 4, 0.7, 100),
                                 ("arfima1p", "SN", 0, 0.7, 100)):
        put(f"fn-{model}-{proc}{k if k else ''}-{H:.2f}-{n}", fixed(fsz(model, proc, k, H, n), 1))
    # check: distortion |size - 5| increases with m (HI) at every (model, H, n)
    mono = True
    for (model, H, n), g in O[O.procedure == "HI"].groupby(["model", "H", "n"]):
        d = g.sort_values("k")["size"].sub(5).abs().values
        mono &= bool(np.all(np.diff(d) >= -0.02))
    CHECKS.append(("HI distortion non-decreasing in m at every (model,H,n) [tol 0.02pp]", mono))
    # table: H = 0.7, procedures x n, per model
    procs = [("SN", 0, "SN"), ("HI", 1, r"HI, $m=1$"), ("HI", 4, r"HI, $m=4$"), ("HI", 16, r"HI, $m=16$"), ("EWC", 8, r"EWC, $B=8$"), ("EWC", 32, r"EWC, $B=32$")]
    NS = (50, 100, 200, 500, 1000, 2000)
    L = [r"\begin{tabular}{@{}ll" + "r" * len(NS) + r"@{}}\toprule", "model & test & " + " & ".join(f"$n={n}$" for n in NS) + r"\\\midrule"]
    names = {"fgn": "fGn", "arfima0": r"ARFIMA$(0,d,0)$", "arfima1p": r"ARFIMA$(1,d,0)$, $\phi=0.5$", "arfima1m": r"ARFIMA$(1,d,0)$, $\phi=-0.5$"}
    for model in ("fgn", "arfima0", "arfima1p", "arfima1m"):
        for i, (proc, k, lab) in enumerate(procs):
            L.append((rf"\multirow{{{len(procs)}}}{{*}}{{{names[model]}}}" if i == 0 else "") + f" & {lab} & " +
                     " & ".join(fixed(fsz(model, proc, k, 0.7, n), 2) for n in NS) + r"\\")
        L.append(r"\midrule" if model != "arfima1m" else "")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_finite.tex"), "w").write("\n".join(L) + "\n")

Vf = sorted(glob.glob(os.path.join(RES, "E3b_validate_*.csv")))
if need(Vf, "E3b validate"):
    V = pd.concat([pd.read_csv(f) for f in Vf], ignore_index=True)
    put("val-R", grouped(int(V.R.max()))); put("val-n-cells", str(len(V))); put("val-n-inside", str(int(V.exact_inside_ci99.sum())))
    put("val-z-mean", fixed(V.z.mean(), 2)); put("val-z-sd", fixed(V.z.std(), 2))
T2f = os.path.join(RES, "E3c_hi2017_table2.csv")
if need(os.path.exists(T2f), "E3c"):
    T2 = pd.read_csv(T2f).dropna(subset=["published_rate"])
    put("hi17-t2-maxz", fixed(T2.z.abs().max(), 2)); put("hi17-t2-n", str(len(T2)))

R35f = [os.path.join(RES, f"E3b_hi2017_table{t}.csv") for t in (3, 4, 5)]
LRR = pd.read_csv(os.path.join(LIT, "ledger_rejection_rates.csv"))
if need(all(os.path.exists(f) for f in R35f), "E3b hi2017"):
    R35 = pd.concat([pd.read_csv(f) for f in R35f], ignore_index=True); R35["tab"] = R35.design.str.extract(r"Table (\d)").astype(int)
    pub = LRR[(LRR.source_id == "HI2017") & LRR.location.str.contains("Table [345]") & LRR.procedure.str.startswith("fixed-m")].copy()
    pub["tab"] = pub.location.str.extract(r"Table (\d)").astype(int); pub["m"] = pub.procedure.str.extract(r"m=(\d+)").astype(int)
    M = R35.merge(pub[["tab", "n", "m", "published_rate", "replications_N"]], left_on=["tab", "T", "m"], right_on=["tab", "n", "m"])
    s = M[M.rule == "plugin_hi_table"].copy()
    s["z"] = (s.mc_rate - s.published_rate) / np.sqrt(s.published_rate * (1 - s.published_rate) / pd.to_numeric(s.replications_N, errors="coerce") + s.mc_rate * (1 - s.mc_rate) / s.R)
    put("hi17-R", grouped(int(R35.R.max()))); put("hi17-ncells", str(len(s)))
    put("hi17-meandiff", ("+" if (s.mc_rate - s.published_rate).mean() >= 0 else "") + fixed(100 * (s.mc_rate - s.published_rate).mean(), 2))
    put("hi17-share-z2", str(int(round(100 * (s.z.abs() > 2).mean())))); put("hi17-maxdiff", fixed(100 * (s.mc_rate - s.published_rate).abs().max(), 1))
    put("hi17-T5-64-16-plugin", fixed(100 * float(M[(M.tab == 5) & (M["T"] == 64) & (M.m == 16) & (M.rule == "plugin_exact")].mc_rate.iloc[0]), 1))
    L = [r"\begin{tabular}{@{}llrrrrrrr@{}}\toprule", r"design & $T$ & $m$ & published & HI table & exact plug-in & rescaled $t$ & oracle & naive\\\midrule"]
    for t in (3, 4, 5):
        d0, p0 = R35[R35.tab == t][["delta", "phi"]].iloc[0]
        for T in (64, 256, 1024):
            for i, m in enumerate((1, 4, 16)):
                g = M[(M.tab == t) & (M["T"] == T) & (M.m == m)].set_index("rule")
                lab = rf"T{t}: $\delta={d0:g}$, $\phi={p0:g}$" if (T == 64 and i == 0) else ""
                L.append(f"{lab} & {T if i == 0 else ''} & {m} & " + " & ".join(fixed(100 * float(v), 1) for v in
                         (g.published_rate.iloc[0], g.mc_rate["plugin_hi_table"], g.mc_rate["plugin_exact"], g.mc_rate["plugin_rescaled_t"],
                          g.mc_rate["oracle_exact"], g.mc_rate["naive_t2m"])) + r"\\")
        if t != 5:
            L.append(r"\midrule")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_hi17.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ section 5: power (E7)
Pf = sorted(glob.glob(os.path.join(RES, "E7_power_H*.csv")))
if need(Pf, "E7"):
    P = pd.concat([pd.read_csv(f) for f in Pf], ignore_index=True); P["k"] = pd.to_numeric(P.param, errors="coerce").fillna(0).astype(int)
    def pw(proc, k, H, c=2.8):
        s = P[(P.procedure == proc) & (P.k == k) & np.isclose(P.H, H) & np.isclose(P.c, c)]
        return float(s.power.iloc[0])
    for proc, k, H in (("HI", 1, 0.75), ("HI", 4, 0.75), ("HI", 16, 0.75), ("HI", 16, 0.5), ("HI", 16, 0.9), ("SN", 0, 0.5),
                       ("SN", 0, 0.9), ("HI", 4, 0.7), ("SN", 0, 0.7)):
        put(f"pw-{proc}{k if k else ''}-{H:.2f}", fixed(pw(proc, k, H), 3))
    put("pw-z", fixed(float(P[P.procedure.str.startswith("oracle") & np.isclose(P.c, 2.8)].power.iloc[0]), 3))
    put("pw-c0-maxdev", sci(float((P[np.isclose(P.c, 0.0)].power - 0.05).abs().max()), 1))
    dmax = 0.0
    for m in (1, 2, 4, 8, 16):
        a = P[(P.procedure == "HI") & (P.k == m)].set_index(["H", "c"]).power; b = P[(P.procedure == "EWC") & (P.k == 2 * m)].set_index(["H", "c"]).power
        dmax = max(dmax, float((a - b).abs().max()))
    put("pw-hi-ewc-maxdiff", fixed(dmax, 3))
    nctf = os.path.join(RES, "E7c_nct_check.json")          # H = 1/2 check against the noncentral t (quadrature reference)
    if need(os.path.exists(nctf), "E7c"):
        nc_ = json.load(open(nctf))
        put("pw-nct-maxdev", sci(nc_["max_abs_dev"], 1)); put("pw-nct-ncells", str(nc_["cells"]))
    # power alone (17 Nov, audit C2): fixed-smoothing test minus Bartlett/SN over all H and c > 0 of the grid
    snp = P[P.procedure == "SN"].set_index(["H", "c"]).power
    for proc, k in (("HI", 1), ("HI", 2), ("HI", 4), ("EWC", 2), ("EWC", 4), ("EWC", 8)):
        d_ = P[(P.procedure == proc) & (P.k == k)].set_index(["H", "c"]).power
        d_ = (d_ - snp.reindex(d_.index)); d_ = d_[d_.index.get_level_values("c") > 0]
        put(f"pw-vsSN-{proc}{k}-npos", str(int((d_ > 0).sum()))); put("pw-vsSN-ncells", str(len(d_)))
        put(f"pw-vsSN-{proc}{k}-min", fixed(float(d_.min()), 3))
    jl = os.path.join(RES, "E7b_sn_power_Jladder.json")     # SN power: convergence in the Galerkin truncation J
    if need(os.path.exists(jl), "E7b"):
        J_ = json.load(open(jl))
        if need(len(J_["H"]) == 6, "E7b complete"):
            put("pw-jl-256-512", sci(max(v["maxdiff_vs_512"]["256"] for v in J_["H"].values()), 1))
            put("pw-jl-512-1024", sci(max(v["maxdiff_vs_1024"]["512"] for v in J_["H"].values()), 1))
            put("pw-jl-128-512", sci(max(v["maxdiff_vs_512"]["128"] for v in J_["H"].values()), 1))
    if E3f:
        cells = 0; dom = 0
        for (model, H, n), g in O.groupby(["model", "H", "n"]):
            if not np.any(np.isclose(P.H, H)):
                continue
            sn = g[g.procedure == "SN"]["size"].iloc[0]; h4 = g[(g.procedure == "HI") & (g.k == 4)]["size"].iloc[0]
            cells += 1; dom += int(abs(h4 - 5) <= abs(sn - 5) and pw("HI", 4, H) >= pw("SN", 0, H))
        put("dominance-ncells", str(cells)); put("dominance-share", f"{dom}")
        CHECKS.append((f"dominance: HI m=4 vs SN in {dom}/{cells} cells", True))

# ------------------------------------------------------------------ section 6: audit (E4)
A = {}
for part in ("half", "hi", "hi_neg", "mp", "mp_neg"):
    f = os.path.join(RES, f"E4_audit_{part}.csv")
    if need(os.path.exists(f), f"E4 {part}"):
        A[part] = pd.read_csv(f)
LCV = pd.read_csv(os.path.join(LIT, "ledger_critical_values.csv"))
put("ledger-n-cv", str(len(LCV))); put("ledger-n-rr", str(len(LRR)))
if len(A) == 5:
    ALL = pd.concat([A[p].assign(part=p) for p in A], ignore_index=True)
    # the 1990 KPSS working paper prints the same eight values as the 1992 article: count them once (referee, 16 Nov)
    wp = ALL[ALL.source == "KPSS1990wp"]; jr = ALL[ALL.source == "KPSS1992"]
    assert sorted(wp.published.round(6)) == sorted(jr.published.round(6)), "KPSS working-paper values differ from the article"
    put("audit-n-entries", str(len(ALL))); put("audit-n-dup", str(len(wp)))
    ALL = ALL[ALL.source != "KPSS1990wp"].reset_index(drop=True)
    ALL["z"] = ALL.dev_pp / ALL.mc_se_pp
    put("audit-n-values", str(len(ALL))); put("audit-n-sources", str(ALL.source.replace({"KPSS1990wp": "KPSS1992"}).nunique()))
    put("audit-n-within3", str(int((ALL.z.abs() <= 3).sum())))
    h = A["half"]
    apkv = h[h.source.isin(["AP1997", "KV2002wp"])]
    put("audit-apkv-maxrel", sci(float(apkv.rel_dev.abs().max()), 1))
    lob = h[h.source == "Lobato2001"].sort_values("alpha", ascending=False)
    put("audit-lob-sizes", ", ".join(fixed(100 * v, 3) for v in lob.true_size))
    kt = h[h.ledger_id.astype(str).str.contains("tau", case=False) & np.isclose(h.alpha, 0.05) & (h.source == "KPSS1992")]
    if need(len(kt), "audit-kpss-tau"):
        r = kt.iloc[0]; put("audit-kpss-tau-05", fixed(100 * r.true_size, 2)); put("audit-kpss-tau-05-z", fixed(r.dev_pp / r.mc_se_pp, 1))
        put("audit-kpss-tau-05-exact", sig_str(float(r.exact_cv), 4))
    hp, hn = A["hi"], A["hi_neg"]
    put("audit-hi-pos-n", str(len(hp))); put("audit-hi-pos-min", fixed(100 * hp.true_size.min(), 2)); put("audit-hi-pos-max", fixed(100 * hp.true_size.max(), 2))
    put("audit-hi-pos-gt01", str(int((hp.dev_pp.abs() > 0.1).sum())))
    put("audit-hi-neg-n", str(len(hn))); put("audit-hi-neg-min", fixed(100 * hn.true_size.min(), 2)); put("audit-hi-neg-max", fixed(100 * hn.true_size.max(), 2))
    put("audit-hi-neg-gt2se", str(int((hn.dev_pp.abs() > 2 * hn.mc_se_pp).sum())))
    put("audit-hi-neg-gt3se", str(int((hn.dev_pp.abs() > 3 * hn.mc_se_pp).sum())))
    put("audit-hi-disc-max", sci(float(pd.concat([hp, hn]).eval("abs(disc_dev_pp - dev_pp)").max()), 1))
    put("audit-hi-2se", fixed(2 * float(hp.mc_se_pp.iloc[0]), 2))
    mp, mn = A["mp"], A["mp_neg"]
    mpp = mp[mp.H >= 0.6 - 1e-9]
    put("audit-mp-pos-n", str(len(mpp))); put("audit-mp-pos-min", fixed(mpp.dev_pp.min(), 2)); put("audit-mp-pos-max", fixed(mpp.dev_pp.max(), 2))
    r = mp[np.isclose(mp.H, 0.9) & np.isclose(mp.alpha, 0.10)]
    if need(len(r), "audit-mp-H090-a10"):
        put("audit-mp-H090-a10", fixed(100 * float(r.true_size.iloc[0]), 2))
    put("audit-mp-neg-min", fixed(mn.dev_pp.min(), 2)); put("audit-mp-neg-max", fixed(mn.dev_pp.max(), 2))
    # verdict classes (P8 + protocol Amendments 5 and 8). Precedence: (1) values beyond 3 s.e. in the limit: C if the exact law
    # of the finite design inferred for the source puts them within 3 s.e. (HI delta < 0), otherwise E; (2) D for every value
    # of a block (source x statistic x memory regime H = 1/2, > 1/2, < 1/2) in which all values are within 3 s.e., all deviations
    # have one sign, k >= 3, and |sum z| / sqrt(k) > 3 (values treated as independent); (3) B if > 0.1 pp from 5 % at the 5 %
    # level, otherwise A. The rule is applied uniformly (17 Nov audit C1); it reproduces the earlier assignment.
    def _block(r):
        reg = "half" if abs(r.H - 0.5) < 1e-9 else ("pos" if r.H > 0.5 else "neg")
        stat = "eta_tau" if "eta_tau" in str(r.ledger_id) else ("eta_mu" if "eta_mu" in str(r.ledger_id) else "")
        return f"{r.source}|{stat}|{reg}"
    ALL["block"] = ALL.apply(_block, axis=1)
    Dblocks, Dinfo = set(), {}
    for b_, g_ in ALL.groupby("block"):
        z_ = g_.z.replace([np.inf, -np.inf], np.nan)
        if z_.isna().any() or len(g_) < 3 or (z_.abs() > 3).any():
            continue
        one_sign = bool((g_.dev_pp > 0).all() or (g_.dev_pp < 0).all())
        jz = float(z_.sum() / math.sqrt(len(z_)))
        Dinfo[b_] = (one_sign, jz)
        if one_sign and abs(jz) > 3:
            Dblocks.add(b_)
    def _cls(r):
        five = abs(r.alpha - 0.05) < 1e-12
        if r.part == "hi_neg":
            v = str(r.verdict_final)
            if v.startswith("explained"): return "C"
            if v.startswith("unexplained"): return "E"
            return "B" if (five and abs(r.dev_pp) > 0.1) else "A"
        if abs(r.z) > 3: return "E"
        if r.block in Dblocks: return "D"
        return "B" if (five and abs(r.dev_pp) > 0.1) else "A"
    ALL["cls"] = ALL.apply(_cls, axis=1)
    for k_ in "ABCDE":
        put(f"audit-n-{k_}", str(int((ALL.cls == k_).sum())))
    put("audit-n-AB", str(int(ALL.cls.isin(["A", "B"]).sum())))
    # the uniform rule selects two blocks (17 Nov): McElroy-Politis H > 1/2 and Hualde-Iacone delta = 0 (Student t_{2m} row)
    assert Dblocks == {"MP2009TR||pos", "HI2017||half"}, f"class D blocks changed: {sorted(Dblocks)}"
    mpD = ALL[ALL.block == "MP2009TR||pos"]
    put("audit-D-jointz", fixed(Dinfo["MP2009TR||pos"][1], 1))
    rz = mpD.groupby("H").z.mean()                      # one simulated row = one observation (conservative)
    put("audit-D-jointz-rows", fixed(float(rz.sum() / math.sqrt(len(rz))), 1))
    put("audit-D-nrows", str(len(rz)))
    hiD = ALL[ALL.block == "HI2017||half"]
    put("audit-D-hi0-n", str(len(hiD)))
    put("audit-D-hi0-jointz", fixed(abs(Dinfo["HI2017||half"][1]), 1))
    put("audit-D-hi0-rowz", fixed(abs(float(hiD.z.mean())), 1))
    put("audit-D-hi0-min", fixed(100 * float(hiD.true_size.min()), 2)); put("audit-D-hi0-max", fixed(100 * float(hiD.true_size.max()), 2))
    put("audit-D-hi0-maxz", fixed(float(hiD.z.abs().max()), 1))
    lb = [k for k in Dinfo if k.startswith("Lobato")]
    if need(len(lb) == 1, "audit-lob-jointz"):
        put("audit-lob-jointz", fixed(Dinfo[lb[0]][1], 1))
    kt = [k for k in Dinfo if "eta_tau" in k]
    if need(len(kt) == 1, "audit-kpsstau-jointz"):
        put("audit-kpsstau-jointz", fixed(Dinfo[kt[0]][1], 1))
    rnd = os.path.join(RES, "E4_mp_beta-0.4_rounding.json")
    if need(os.path.exists(rnd), "E4 rounding"):
        rr_ = json.load(open(rnd))
        put("audit-mp-T3-round-maxshift", fixed(max(x["max_shift_pp"] for x in rr_), 2))
        put("audit-mp-T3-round-mindev", fixed(min(abs(x["dev_pp"]) for x in rr_), 2))
    hn2 = A["hi_neg"]
    put("audit-hi-neg-design-gt2", str(int((hn2.design_z.abs() > 2).sum()))); put("audit-hi-neg-design-gt3", str(int((hn2.design_z.abs() > 3).sum())))
    put("audit-hi-neg-design-maxz", fixed(float(hn2.design_z.abs().max()), 2))
    put("audit-hi-neg-maxdev", fixed(float(hn2.dev_pp.max()), 1)); put("audit-hi-neg-maxz", fixed(float((hn2.dev_pp / hn2.mc_se_pp).abs().max()), 0))
    DES = pd.read_csv(os.path.join(RES, "E4_audit_hi_neg_design.csv"))
    d16 = DES[np.isclose(DES.H, 0.01) & (DES.m == 16)].set_index("n")
    for n_, kk in ((1000, "T1e3"), (10000, "T1e4"), (100000, "T1e5")):
        put(f"audit-hi-neg-m16-{kk}", fixed(100 * float(d16.loc[n_, "size_arfima0"]), 1))
    put("audit-hi-neg-m16-limit", fixed(100 * float(d16.loc[10000, "size_limit"]), 1)); put("audit-hi-neg-m16-fgn", fixed(100 * float(d16.loc[10000, "size_fgn_n1e4"]), 1))
    r1 = hn2[np.isclose(hn2.H, 0.01) & (hn2.m == 1)].iloc[0]; put("audit-hi-neg-m1-limit", fixed(100 * float(r1.true_size), 1))
    put("audit-hi-neg-design-min", fixed(100 * float(hn2.design_size.min()), 2)); put("audit-hi-neg-design-max", fixed(100 * float(hn2.design_size.max()), 2))
    mn2 = A["mp_neg"].assign(z=lambda d: d.dev_pp / d.mc_se_pp)
    t3 = mn2[np.isclose(mn2.beta, -0.4)]; t1 = mn2[np.isclose(mn2.beta, -0.8)]
    put("audit-mp-T3-dev10", fixed(float(t3[np.isclose(t3.alpha, 0.10)].dev_pp.iloc[0]), 2)); put("audit-mp-T3-size05", fixed(100 * float(t3[np.isclose(t3.alpha, 0.05)].true_size.iloc[0]), 2))
    put("audit-mp-T3-maxz", fixed(float(t3.z.abs().max()), 0)); put("audit-mp-T1-minz", fixed(float(t1.z.abs().min()), 1)); put("audit-mp-T1-maxz", fixed(float(t1.z.abs().max()), 1))
    put("audit-mp-T1-devmin", fixed(float(t1.dev_pp.min()), 2)); put("audit-mp-T1-devmax", fixed(float(t1.dev_pp.max()), 2))
    mpp2 = mp[mp.H >= 0.6 - 1e-9].assign(z=lambda d: d.dev_pp / d.mc_se_pp); put("audit-mp-pos-maxz", fixed(float(mpp2.z.max()), 2))
    put("audit-mp-pos-npos", str(int((mpp2.dev_pp > 0).sum())))
    # verdict table by source
    lab = {"Lobato2001": r"Lobato~\fixcite{Lob01}", "AP1997": r"Abadir--Paruolo~\fixcite{AP97}", "KV2002wp": r"Kiefer--Vogelsang~\fixcite{KV02}",
           "KPSS1990wp": r"KPSS~\fixcite{KPSS92}, WP", "KPSS1992": r"KPSS~\fixcite{KPSS92}", "HI2017": r"Hualde--Iacone~\fixcite{HI17}",
           "MP2009TR": r"McElroy--Politis~\fixcite{MP09}"}
    L = [r"\begin{tabular}{@{}llrrrrrrrr@{}}\toprule",
         r"& & & & & \multicolumn{5}{c}{verdict class}\\\cmidrule(l){6-10}",
         r"source & range & $N$ & deviation (pp) & s.e.\ (pp) & A & B & C & D & E\\\midrule"]
    for (src, part), g in ALL.groupby(["source", "part"], sort=False):
        rng = {"half": r"$H=\frac12$", "hi": r"$\delta\ge0$", "hi_neg": r"$\delta<0$", "mp": r"$\beta\ge0$", "mp_neg": r"$\beta<0$"}[part]
        cc = [int((g.cls == k_).sum()) for k_ in "ABCDE"]
        se_ = (r"exact\textsuperscript{a}" if g.mc_se_pp.isna().all() else f"{g.mc_se_pp.min():.2f}"
               + (f"--{g.mc_se_pp.max():.2f}" if g.mc_se_pp.max() - g.mc_se_pp.min() > 0.005 else ""))
        L.append(f"{lab.get(src, src)} & {rng} & {len(g)} & {g.dev_pp.min():+.2f} to {g.dev_pp.max():+.2f} & {se_}" + " & "
                 + " & ".join(str(c_) if c_ else r"--" for c_ in cc) + r"\\")
    tot = [int((ALL.cls == k_).sum()) for k_ in "ABCDE"]
    L.append(r"\midrule all & & " + f"{len(ALL)} & & & " + " & ".join(str(c_) for c_ in tot) + r"\\")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_audit.tex"), "w").write("\n".join(L) + "\n")
    # per-value tables
    L = [r"\begin{longtable}{@{}llrrrrrc@{}}", r"\caption{Audit at $H=\frac12$: every value. Class as in \cref{tab:audit}.}\label{tab:audithalf}\\\toprule",
         r"source & id & $\alpha$ & published & exact & size (\%) & $z$ & class\\\midrule\endfirsthead",
         r"\caption[]{(continued)}\\\toprule source & id & $\alpha$ & published & exact & size (\%) & $z$ & class\\\midrule\endhead",
         r"\midrule\multicolumn{8}{r@{}}{\footnotesize\itshape continued on the next page}\\\endfoot", r"\bottomrule\endlastfoot"]
    for _, r in ALL[ALL.part == "half"].iterrows():
        L.append(f"{lab.get(r.source, r.source)} & \\texttt{{{str(r.ledger_id).replace('_', chr(92) + '_')}}} & {r.alpha:g} & {r.published:g} & {sig_str(float(r.exact_cv), 6)} & "
                 f"{100 * r.true_size:.3f} & " + ("--" if pd.isna(r.mc_se_pp) else f"{r.dev_pp / r.mc_se_pp:+.2f}") + f" & {r.cls}" + r"\\")
    L.append(r"\end{longtable}"); open(os.path.join(TAB, "tab_audit_half.tex"), "w").write("\n".join(L) + "\n")
    hdr = r"id & $H$ & $\alpha$ & published & exact & size (\%) & $z$ & design (\%) & class\\\midrule"
    L = [r"\begin{longtable}{@{}lrrrrrrrc@{}}",
         r"\caption{Audit for $H\ne\frac12$: every value. True size: certified limit size, except McElroy--Politis at $\beta<0$, where it is"
         r" the exact size under their own $1000$-step fGn design. Design size: exact size under the source's finite design"
         r" (Hualde--Iacone $\delta<0$: stationary ARFIMA$(0,\delta,0)$, $T=10\,000$). Class as in \cref{tab:audit}.}\label{tab:auditfull}\\\toprule",
         hdr + r"\endfirsthead", r"\caption[]{(continued)}\\\toprule " + hdr + r"\endhead",
         r"\midrule\multicolumn{9}{r@{}}{\footnotesize\itshape continued on the next page}\\\endfoot", r"\bottomrule\endlastfoot"]
    for part in ("hi_neg", "hi", "mp_neg", "mp"):
        sub_ = ALL[ALL.part == part]
        for _, r in sub_.sort_values(["H"] + (["m"] if part.startswith("hi") else ["alpha"])).iterrows():
            ds = f"{100 * r.design_size:.2f}" if part == "hi_neg" else "--"
            L.append(f"\\texttt{{{str(r.ledger_id).replace('_', chr(92) + '_')}}} & {r.H:.2f} & {r.alpha:g} & {r.published:g} & {sig_str(float(r.exact_cv), 5)} & {100 * r.true_size:.2f} & "
                     f"{r.dev_pp / r.mc_se_pp:+.2f} & {ds} & {r.cls}" + r"\\")
    L.append(r"\end{longtable}"); open(os.path.join(TAB, "tab_audit_full.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ section 7: computation (E1, E1b, E5)
E1 = {os.path.basename(f)[14:-5]: json.load(open(f)) for f in glob.glob(os.path.join(RES, "E1_benchmark_H*.json"))}
if need(E1, "E1"):
    put("e1-width-max-16384", sci(max(r["width_F"] for d in E1.values() for r in d["rows"] if r["J"] == 16384), 1))
    dev = 0.0
    for d in E1.values():
        H = float(d["H"]); rows = sorted([r for r in d["rows"] if abs(r["p"] - 0.9) < 1e-9], key=lambda r: r["J"])
        J = np.array([r["J"] for r in rows if r["J"] <= 8192], float); w = np.array([r["width_F"] for r in rows if r["J"] <= 8192])
        dev = max(dev, float(np.max(np.abs(np.diff(np.log(w)) / np.diff(np.log(J)) + 0.5 + 2 * H))))
    put("e1-slope-maxdev", fixed(dev, 2))
E1b = os.path.join(RES, "E1b_nodes.json")
if need(os.path.exists(E1b), "E1b"):
    rr = [r["ratio"] for r in json.load(open(E1b))["rows"]]
    put("e1-tilt-node-ratio-min", sig_str(min(rr), 2)); put("e1-tilt-node-ratio-max", sig_str(max(rr), 2))
E5f = os.path.join(RES, "E5_methods.csv")
E5m = os.path.join(RES, "E5_methods_meta.json")
if need(os.path.exists(E5f) and os.path.exists(E5m), "E5"):
    # revised 17 Nov (audit B1-B4): error intervals against enclosures, documented domains, cold / repeated / amortised cost
    E5 = pd.read_csv(E5f); M5 = json.load(open(E5m))
    IMPL = [("Ruben series)", "certruben"), ("tilted Bromwich in arb", "certbromwich"), ("exact law", "exact"),
            ("Imhof (scipy", "imhof"), ("Bromwich tilted midpoint", "bromwich"), ("gx2 imhof", "gx2imhof"), ("gx2 auto", "gx2auto"),
            ("gx2 ifft", "gx2ifft"), ("gx2 ray", "gx2ray"), ("gx2 pearson", "gx2pearson"), ("Ruben-Beta", "ruben"),
            ("Lugannani", "lr"), ("leading-20", "lead20"), ("Monte Carlo", "mc"), ("davies", "davies"),
            ("CompQuadForm::imhof", "cqfimhof"), ("liu (LTZ", "ltz"), ("hbe", "hbe"), ("lpb4", "lpb4"),
            ("pchisqsum saddlepoint", "svsaddle"), ("pchisqsum satterthwaite", "satt")]
    METH = {"certruben": "ruben", "ruben": "ruben", "certbromwich": "bromwich", "exact": "bromwich", "bromwich": "bromwich",
            "imhof": "imhof", "cqfimhof": "imhof", "gx2imhof": "imhof", "gx2auto": "imhof", "davies": "davies", "gx2ifft": "fft",
            "gx2ray": "ray", "gx2pearson": "pearson", "ltz": "ltz", "hbe": "hbe", "lpb4": "lpb4", "lr": "saddlepoint",
            "svsaddle": "saddlepoint", "satt": "satterthwaite", "lead20": "leading-k + Satterthwaite", "mc": "Monte Carlo"}
    def implof(m):
        for k, v in IMPL:
            if k in m:
                return v
        raise ValueError(f"E5: unmapped method {m}")
    E5["impl"] = E5.method.map(implof); E5["meth"] = E5.impl.map(METH)
    main5 = E5[~E5.domain.isin(["diagnostic", "timing only"])]
    put("e5-n-impl", str(main5.impl.nunique())); put("e5-n-methods", str(main5.meth.nunique()))
    put("e5-n-targets-rows", str(len(main5)))
    APX = ["lr", "gx2pearson", "hbe", "lpb4", "svsaddle", "satt", "lead20"]
    apx = main5[main5.impl.isin(APX) & (main5.domain == "documented domain") & main5.value.notna()]
    put("e5-approx-maxerr-pp", fixed(100 * apx.err_hi.max(), 1))
    CHECKS.append(("E5: largest in-domain approximation error is gx2 pearson",
                   apx.loc[apx.err_hi.idxmax()].impl == "gx2pearson"))
    def mx(impl, tgt=None):
        s = main5[(main5.impl == impl) & main5.value.notna()]
        if tgt:
            s = s[s.target.str.startswith(tgt)]
        return float(s.err_hi.max())
    put("e5-lr-err-pp", fixed(100 * mx("lr"), 2)); put("e5-satt-err-pp", fixed(100 * mx("satt"), 1))
    put("e5-hbe-err-pp", fixed(100 * mx("hbe"), 2)); put("e5-lpb4-err-pp", fixed(100 * mx("lpb4"), 2))
    put("e5-svsaddle-err-pp", fixed(100 * mx("svsaddle"), 2))
    put("e5-3mom-T1-pp", fixed(100 * mx("gx2pearson", "T1"), 1)); put("e5-ltz-T1-pp", fixed(100 * mx("ltz", "T1"), 1))
    pe34 = main5[(main5.impl == "gx2pearson") & main5.target.str.startswith(("T3", "T4"))]
    CHECKS.append(("E5: gx2 pearson returns 1 on T3 and T4", bool((pe34.value >= 1.0).all())))
    lt34 = main5[(main5.impl == "ltz") & main5.target.str.startswith(("T3", "T4"))]
    CHECKS.append(("E5: LTZ (outside its domain) returns 0 on T3 and T4", bool((lt34.value <= 0.0).all())))
    put("e5-lead20-err", sci(mx("lead20"), 1))
    exact_d = main5[main5.impl.isin(["imhof", "cqfimhof", "gx2imhof", "gx2auto", "davies", "ruben", "bromwich"]) &
                    main5.target.str.startswith(("T1", "T2")) & main5.value.notna()]
    put("e5-T12-exact-maxerr", sci(float(exact_d.err_hi.max()), 0))
    put("e5-T12-exact-maxs", sci(float(exact_d.seconds_method.max()), 0))
    cr = main5[main5.impl == "certruben"]
    put("e5-T12-cert-s", fixed(float(cr.seconds_method.max()), 2)); put("e5-T12-cert-width", sci(float((cr.ref_hi - cr.ref_lo).max()), 0))
    # outer-quadrature diagnostic: Gauss-Hermite (64 nodes) minus adaptive, same inner method and target
    dg = E5[E5.domain == "diagnostic"].copy(); dg["key"] = dg.method.str.split("|").str[0].str.strip() + "#" + dg.target
    ad = E5[E5.method.str.contains("adaptive over Z", regex=False)].copy(); ad["key"] = ad.method.str.split("|").str[0].str.strip() + "#" + ad.target
    jj = dg.merge(ad[["key", "value"]], on="key", suffixes=("_gh", "_ad")).dropna(subset=["value_gh", "value_ad"])
    put("e5-gh-maxdiff-pp", fixed(100 * float((jj.value_gh - jj.value_ad).abs().max()), 2))
    # T3: cold, repeated and amortised cost
    t3m = M5["targets"]["T3 SN n=1000 ARFIMA(0,0.2,0)"]; am = t3m["amortisation"]
    t3 = main5[main5.target.str.startswith("T3")]
    put("e5-T3-setup-s", fixed(t3m["SNExact_setup_s"], 2)); put("e5-T3-query-s", fixed(t3m["SNExact_repeat_query_s"], 2))
    put("e5-T3-eig-s", fixed(t3m["seconds_cholesky"] + t3m["seconds_ratio_eigs"], 2))
    put("e5-T3-imhof-s", fixed(float(t3[t3.impl == "imhof"].seconds_total.iloc[0]), 2))
    put("e5-T3-ifft-s", f"{float(t3[t3.impl == 'gx2ifft'].seconds_total.iloc[0]):.0f}")
    put("e5-amort-m", str(am["m"])); put("e5-amort-ours-s", fixed(am["SNExact_total_s"], 1))
    put("e5-amort-imhof-s", fixed(am["Imhof (scipy quad)_total_s"], 1))
    put("e5-amort-ratio", fixed(am["Imhof (scipy quad)_total_s"] / am["SNExact_total_s"], 1))
    put("e5-amort-maxdiff", sci(am["Imhof (scipy quad)_max_abs_diff_vs_SNExact"], 0))
    # T4
    t4 = main5[main5.target == "T4 T_H H=0.75 via Galerkin J=512"]
    put("e5-T4-galerkin-err", sci(float(t4[t4.impl == "imhof"].err_hi.iloc[0]), 1))
    put("e5-T4-galerkin-s", fixed(float(t4[t4.impl == "imhof"].seconds_total.iloc[0]), 1))
    c4 = main5[main5.target == "T4 T_H H=0.75 (limit)"].iloc[0]
    put("e5-T4-cert-s", fixed(float(c4.seconds_total), 1)); put("e5-T4-cert-eval-s", fixed(float(c4.seconds_method), 1))
    put("e5-T4-ref-width", sci(float(c4.ref_hi - c4.ref_lo), 1))
    r20 = main5[(main5.target == "T4 T_H H=0.75 via Galerkin J=2048") & (main5.impl == "gx2ray")]
    put("e5-T4-ray2048-s", f"{float(r20.seconds_method.iloc[0]):.0f}")
    fails = E5[(E5.value.isna()) & ~E5.domain.isin(["timing only"])]
    put("e5-n-fail", str(len(fails)))
    env5 = M5["environment"]
    put("e5-env", f"{env5['logical_cpus']} logical CPUs, {env5['platform'].split('-')[0]} {env5['platform'].split('-')[1]}, Python {env5['python']}, R {env5.get('R', '?').split()[2] if 'R' in env5 else '?'}")

# budget table (E1 at J = 16384 and E2 production radii)
if E1 and SN:
    L = [r"\begin{tabular}{@{}lrrrr@{}}\toprule", r"$H$ & enclosure width & exceptional events $2e^{-t}$ & ball radius (finite part) & alias, truncation\\\midrule"]
    for hk in ("0.50", "0.75", "0.95"):
        d = E1.get(hk) or E1.get(str(float(hk)))
        if d is None:
            continue
        r = [x for x in d["rows"] if x["J"] == 16384 and abs(x["p"] - 0.9) < 1e-9][0]
        rad = SN[hk]["0.900"]["X_radius_max"]
        L.append(f"{hk} & {sci(r['width_F'], 1)} & {sci(2 * math.exp(-25), 1)} & {sci(2 * rad, 1)} & $\\le$ {sci(4e-14, 0)}" + r"\\")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_budget.tex"), "w").write("\n".join(L) + "\n")


# sources and experiments tables
SRC = [("Lobato2001", r"\citet{Lob01}", "journal", "Table 1, p.~1067"), ("AP1997", r"\citet{AP97}", "journal (JSTOR)", "Table I, p.~677"),
       ("KV2002wp", r"\citet{KV02}", r"working paper, rev.\ Feb.\ 2001", "Table I, p.~4"), ("KPSS1992", r"\citet{KPSS92}", "journal; 1990 working paper", "Table 1, p.~166"),
       ("HI2017", r"\citet{HI17}", "accepted manuscript", "Tables 1--5"), ("MP2009TR", r"\citet{MP09}", r"technical report, 1 Dec.\ 2009", "Tables 1--9")]
L = [r"\begin{tabular}{@{}llll@{}}\toprule", r"ledger id & reference & version read & location\\\midrule"]
for sid, ref, ver, loc in SRC:
    L.append(f"\\texttt{{{sid}}} & {ref} & {ver} & {loc}" + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_sources.tex"), "w").write("\n".join(L) + "\n")
EXP = [("E0, E0d", "e0\\_anchors.py, e0d\\_revalidate.py", "$H=\\frac12$ anchors, gate G0; part (d) with the final certificate"),
       ("E1, E1b", "e1\\_benchmark\\_J.py, e1b\\_nodes.py", "truncation benchmark, node counts"),
       ("E2, E2b, E2r", "e2\\_production\\_sn.py, e2b\\_production\\_fixed.py, e2r\\_revalidate.py", "certified critical values; recomputation with the final certificate"),
       ("E3", "e3\\_finite\\_n.py", "exact finite-sample sizes"),
       ("E3b, E3c", "e3b\\_mc.py, e3c\\_hi2017\\_table2.py, e3c\\_clip\\_check.py", "Monte Carlo validation, plug-in, HI reproductions; eigenvalue clipping"),
       ("E4", "e4\\_audit.py", "audit"),
       ("E5", "e5\\_methods.py, rq\\_methods.R", "computational methods"),
       ("E6, E6b", "e6\\_naive\\_size.py, e6b\\_rescaled.py", "naive and rescaled sizes"),
       ("E7, E7b, E7c", "e7\\_power.py, e7b\\_power\\_jladder.py, e7c\\_nct\\_check.py", "local power; $J$ ladder; noncentral-$t$ check"),
       ("E8, E9, E9b", "e8\\_spectrum.py, e9\\_surface.py, e9b\\_surface\\_certified.py", "spectrum, response surface, its certified held-out bound"),
       ("E10", "e10\\_matched\\_example.py", "matched finite-grid example"),
       ("E11", "make\\_numbers.py", "independent checks (\\cref{tab:reproduction})"),
       ("E12", "e12\\_edge\\_limit.py", r"limit constants as $H\to1$")]
L = [r"\begin{tabular}{@{}l>{\raggedright}p{0.50\textwidth}>{\raggedright\arraybackslash}p{0.33\textwidth}@{}}\toprule", r"experiment & script & content\\\midrule"]
for e, s_, c in EXP:
    L.append(f"{e} & \\texttt{{{s_}}} & {c}" + r"\\")
L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_experiments.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ matched example (E10) and estimated H (E3b plug-in v2)
E10f = os.path.join(RES, "E10_matched_example.csv")
if need(os.path.exists(E10f), "E10"):
    X10 = pd.read_csv(E10f)
    put("mx-maxdiff-AB", sci(float(X10.A_minus_B_pp.abs().max()), 1)); put("mx-maxdiff-user", sci(float(X10.routeA_minus_user_pp.dropna().abs().max()), 1))
    NS6 = merged_json("E6_naive_sn_*.json")
    L = [r"\begin{tabular}{@{}llrrr@{}}\toprule",
         r"& & \multicolumn{2}{c}{exact, finite grid} & limit\\\cmidrule(lr){3-4}\cmidrule(l){5-5}",
         r"$H$ & critical value used & $n=64$ & $n=128$ & certified\\\midrule"]
    for H_ in (0.50, 0.90, 0.99):
        g = X10[np.isclose(X10.H, H_) & np.isclose(X10.cutoff, 45.52608575, rtol=0, atol=1e-9)].set_index("n")
        lim = NS6.get(f"{H_:.2f}", {}).get("0.05", {}) if NS6 else {}
        if lim and "size_lo" in lim:
            limtxt = fixed(100 * 0.5 * (lim["size_lo"] + lim["size_hi"]), 4)
        elif H_ == 0.5:
            limtxt = "5.0000"
        else:
            limtxt = "--"
        L.append(f"{H_:.2f} & $H=\\frac12$ value $45.526$ & {100 * g.loc[64, 'size_routeA_pct'] / 100:.4f} & {100 * g.loc[128, 'size_routeA_pct'] / 100:.4f} & {limtxt}" + r"\\")
        put(f"mx-{H_:.2f}-64", fixed(float(g.loc[64, "size_routeA_pct"]), 4)); put(f"mx-{H_:.2f}-128", fixed(float(g.loc[128, "size_routeA_pct"]), 4))
    for cval, lab_, key_ in ((6829.9, r"rounded certified value $6829.9$", "mx-0.99-128-round"),
                             (6829.9404006874, r"certified value $6829.9404$", "mx-0.99-128-own")):
        g = X10[np.isclose(X10.H, 0.99) & np.isclose(X10.cutoff, cval, rtol=0, atol=1e-9)]
        assert len(g) == 1, (cval, len(g))
        L.append(r"0.99 & " + lab_ + " & -- & " + f"{float(g.size_routeA_pct.iloc[0]):.5f} & 5.0000" + r"\\")
        put(key_, fixed(float(g.size_routeA_pct.iloc[0]), 5))
    put("mx-cv-0.99", "6829.94")
    # gap between finite-grid and certified limit sizes (Brownian cut-off rows)
    gaps = {64: [], 128: []}
    for H_ in (0.50, 0.90, 0.99):
        g = X10[np.isclose(X10.H, H_) & np.isclose(X10.cutoff, 45.52608575, rtol=0, atol=1e-9)].set_index("n")
        lim = NS6.get(f"{H_:.2f}", {}).get("0.05", {}); limv = 5.0 if H_ == 0.5 else 100 * 0.5 * (lim["size_lo"] + lim["size_hi"])
        for n_ in (64, 128):
            gaps[n_].append((abs(float(g.loc[n_, "size_routeA_pct"]) - limv), H_))
    for n_ in (64, 128):
        put(f"mx-gap-{n_}", sci(max(gaps[n_])[0], 1)); put(f"mx-gap-{n_}-H", f"{max(gaps[n_])[1]:.2f}")
        put(f"mx-gap-{n_}-long", sci(max(v for v, h in gaps[n_] if h > 0.5), 1))
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_matched.tex"), "w").write("\n".join(L) + "\n")

PLf = sorted(glob.glob(os.path.join(RES, "E3b_plugin_v2_*.csv")))
if need(len(PLf) == 20, "E3b plugin v2"):
    PL = pd.concat([pd.read_csv(f) for f in PLf], ignore_index=True)
    PL["k"] = pd.to_numeric(PL.param, errors="coerce").fillna(0).astype(int); PL["pct"] = 100 * PL.mc_rate
    P5 = PL[np.isclose(PL.alpha, 0.05)]
    FEAS = P5[((P5.procedure == "SN") & (P5.rule == "plugin_exact")) | ((P5.procedure != "SN") & (P5.rule == "plugin_exact_trunc"))]
    def pz(model, proc, k, H_, n_, D=FEAS):
        s = D[(D.model == model) & (D.procedure == proc) & (D.k == k) & np.isclose(D.H, H_) & (D.n == n_)]
        return float(s.pct.iloc[0])
    put("pl-R", grouped(int(PL.R.max())))
    for model in ("fgn", "arfima0", "arfima1p", "arfima1m"):
        for proc, k, lab in (("SN", 0, "SN"), ("HI", 4, "HI4"), ("HI", 16, "HI16")):
            for H_ in (0.7, 0.9):
                for n_ in (100, 2000):
                    put(f"pl-{model}-{lab}-{H_:.1f}-{n_}", fixed(pz(model, proc, k, H_, n_), 1))
    pure = FEAS[FEAS.model.isin(["fgn", "arfima0"]) & (FEAS.H >= 0.6 - 1e-9)]
    put("pl-pure-min-100", fixed(float(pure[pure.n == 100].pct.min()), 1)); put("pl-pure-max-100", fixed(float(pure[pure.n == 100].pct.max()), 1))
    put("pl-pure-min-2000", fixed(float(pure[pure.n == 2000].pct.min()), 1)); put("pl-pure-max-2000", fixed(float(pure[pure.n == 2000].pct.max()), 1))
    # bandwidth sensitivity (referee, 16 Nov): same seeds, local Whittle bandwidth n^0.50 / n^0.65 / n^0.80, pure memory, H in {0.7, 0.9}
    bwr = []
    for bw_ in ("0.50", "0.65", "0.80"):
        for m_ in ("fgn", "arfima0"):
            for h_ in ("0.7", "0.9"):
                f_ = os.path.join(RES, f"E3b_plugin_v2_{m_}_H{h_}.csv") if bw_ == "0.65" else os.path.join(RES, f"E3b_plugin_bw{bw_}_{m_}_H{h_}.csv")
                if os.path.exists(f_):
                    bwr.append(pd.read_csv(f_).assign(bw=bw_))
    if need(len(bwr) == 12, "bandwidth sensitivity runs"):
        BWD = pd.concat(bwr, ignore_index=True); BWD = BWD[np.isclose(BWD.alpha, 0.05)]
        BWD = BWD[((BWD.procedure == "SN") & (BWD.rule == "plugin_exact")) | ((BWD.procedure != "SN") & (BWD.rule == "plugin_exact_trunc"))].copy()
        BWD["pct"] = 100 * BWD.mc_rate; BWD["k"] = pd.to_numeric(BWD.param, errors="coerce").fillna(0).astype(int)
        for bw_ in ("0.50", "0.65", "0.80"):
            for n_ in (100, 2000):
                s_ = BWD[(BWD.bw == bw_) & (BWD.n == n_)].pct
                put(f"bw-{bw_[2:]}-min-{n_}", fixed(float(s_.min()), 1)); put(f"bw-{bw_[2:]}-max-{n_}", fixed(float(s_.max()), 1))
        s_ = BWD[(BWD.bw == "0.80") & (BWD.n >= 500)]
        put("bw-80-fgn-maxdev-500", fixed(float(s_[s_.model == "fgn"].pct.sub(5).abs().max()), 1))
        put("bw-80-arfima0-max-500", fixed(float(s_[s_.model == "arfima0"].pct.max()), 1))
        mono_bw = True
        for (bw_, m_, h_, n_), g_ in BWD[BWD.procedure == "HI"].groupby(["bw", "model", "H", "n"]):
            d_ = g_.sort_values("k").pct.sub(5).abs().values
            if d_.max() > 1.0:
                mono_bw &= bool(np.all(np.diff(d_) >= -0.15))
        CHECKS.append(("bandwidth: where the HI feasible distortion exceeds 1 pp it is non-decreasing in m [tol 0.15pp]", mono_bw))
    ar1p = FEAS[FEAS.model == "arfima1p"]
    put("pl-ar1p-min", fixed(float(ar1p.pct.min()), 1)); put("pl-ar1p-max", fixed(float(ar1p.pct.max()), 1))
    ar1m = FEAS[FEAS.model == "arfima1m"]; put("pl-ar1m-max-100", fixed(float(ar1m[ar1m.n == 100].pct.max()), 1))
    unt = P5[(P5.rule == "plugin_exact") & (P5.procedure == "HI") & (P5.k == 4) & (P5.model == "fgn") & np.isclose(P5.H, 0.5) & (P5.n == 100)]
    put("pl-HI4-H05-untrunc", fixed(float(unt.pct.iloc[0]), 1)); put("pl-HI4-H05-trunc", fixed(pz("fgn", "HI", 4, 0.5, 100), 1))
    # SN least distorted among feasible tests under pure memory at H >= 0.6?
    sn_best = all(pz(mo, "SN", 0, H_, n_) <= min(pz(mo, "HI", 4, H_, n_), pz(mo, "HI", 16, H_, n_)) + 1e-9
                  for mo in ("fgn", "arfima0") for H_ in (0.6, 0.7, 0.8, 0.9) for n_ in (100, 200, 500, 1000, 2000))
    CHECKS.append(("plug-in SN size <= plug-in HI(4), HI(16) size for pure memory, H >= 0.6, all n", sn_best))
    cells_ = [(mo, H_, n_) for mo in ("fgn", "arfima0") for H_ in (0.6, 0.7, 0.8, 0.9) for n_ in (100, 200, 500, 1000, 2000)]
    put("pl-n-pure-cells", str(len(cells_)))
    put("pl-share-hi1-le-sn", str(int(round(100 * np.mean([pz(mo, "HI", 1, H_, n_) <= pz(mo, "SN", 0, H_, n_) for mo, H_, n_ in cells_])))))
    put("pl-share-ewc4-le-sn", str(int(round(100 * np.mean([pz(mo, "EWC", 4, H_, n_) <= pz(mo, "SN", 0, H_, n_) for mo, H_, n_ in cells_])))))
    put("pl-share-sn-le-hi4", str(int(round(100 * np.mean([pz(mo, "SN", 0, H_, n_) <= min(pz(mo, "HI", 4, H_, n_), pz(mo, "HI", 16, H_, n_)) for mo, H_, n_ in cells_])))))
    # oracle vs exact (second validation of the exact laws)
    O = PL[PL.rule == "oracle_exact"].copy()
    E3a = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E3_rates_*.csv"))], ignore_index=True)
    E3a = E3a[E3a.cv_label == "own_limit"].copy(); E3a["k"] = pd.to_numeric(E3a.param, errors="coerce").fillna(0).astype(int)
    O = O.merge(E3a[["model", "H", "n", "procedure", "k", "alpha", "rejection_rate"]], on=["model", "H", "n", "procedure", "k", "alpha"])
    O["z"] = (O.mc_rate - O.rejection_rate) / np.sqrt(O.rejection_rate * (1 - O.rejection_rate) / O.R)
    O["inside"] = (O.rejection_rate >= O.ci99_lo) & (O.rejection_rate <= O.ci99_hi)
    put("or-n-cells", grouped(len(O))); put("or-inside-pct", fixed(100 * O.inside.mean(), 1)); put("or-z-mean", fixed(O.z.mean(), 2))
    put("or-z-sd", fixed(O.z.std(), 2)); put("or-maxz", fixed(O.z.abs().max(), 2))
    CHECKS.append(("oracle cells: at most 1 in 12 outside 99 % and none beyond 4 s.e. (AT4)", bool((1 - O.inside.mean()) <= 1 / 12 and O.z.abs().max() < 4)))
    # table: feasible size at 5 %
    L = [r"\begin{tabular}{@{}llrrrrrr@{}}\toprule", r"& & \multicolumn{3}{c}{$H=0.7$} & \multicolumn{3}{c}{$H=0.9$}\\\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"model & test & $n=100$ & $500$ & $2000$ & $n=100$ & $500$ & $2000$\\\midrule"]
    MODN = {"fgn": "fGn", "arfima0": r"ARFIMA$(0,d,0)$", "arfima1p": r"ARFIMA$(1,d,0)$, $\phi=0.5$", "arfima1m": r"ARFIMA$(1,d,0)$, $\phi=-0.5$"}
    for mo in ("fgn", "arfima0", "arfima1p", "arfima1m"):
        for i, (proc, k, lab) in enumerate((("SN", 0, "SN"), ("HI", 4, r"HI, $m=4$"), ("HI", 16, r"HI, $m=16$"), ("EWC", 16, r"EWC, $B=16$"))):
            vals = [pz(mo, proc, k, H_, n_) for H_ in (0.7, 0.9) for n_ in (100, 500, 2000)]
            L.append((MODN[mo] if i == 0 else "") + f" & {lab} & " + " & ".join(fixed(v, 1) for v in vals) + r"\\")
        if mo != "arfima1m":
            L.append(r"\addlinespace")
    L.append(r"\bottomrule\end{tabular}"); open(os.path.join(TAB, "tab_plugin.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ E11: independent checks of the headline numbers (Appendix E)
# Revised 17 Nov 2026 (audit D2, editorial pass with the scientific-writing-studio table checks): rows grouped by the kind of
# evidence; every cell is computed here from result files (column 'sources'); the table is written to tab_reproduction.tex
# and the same rows to results/E11_reproduction.csv.
def _m(x, d):                     # number in math mode, fixed decimals
    return f"${x:.{d}f}$"
def _s(x, d=1):                   # scientific, math mode (sci already returns $...$)
    return sci(x, d)
E11rows = []
def add11(cat, qty, prim, indep, agree, sources, ok=True):
    E11rows.append(dict(category=cat, quantity=qty, primary=prim, independent=indep, agreement=agree, sources=sources, ok=bool(ok)))
E0f = os.path.join(RES, "E0_anchors.json")
E0d = json.load(open(E0f)) if os.path.exists(E0f) else {}
CAT = [("enclosure", "Enclosure compatibility", "an independent value lies inside a certified enclosure"),
       ("implementation", "Independent implementation", "the same quantity from separate code or another algorithm"),
       ("asymptotic", "Asymptotic consistency", "exact finite-sample or near-edge values approach the certified limit"),
       ("probabilistic", "Probabilistic check", "simulation against exact values"),
       ("convergence", "Convergence check", "refinement of a discretisation parameter"),
       ("reproducibility", "Reproducibility", "a re-run with the same or with corrected code")]
if need(bool(E0d) and SN, "E11 inputs"):
    # --- enclosure compatibility
    lv = [("0.900", "0.9"), ("0.950", "0.95"), ("0.975", "0.975"), ("0.990", "0.99")]
    ins = [SN["0.50"][p]["q_lo"] <= E0d["a"][k]["q_closed_form"] <= SN["0.50"][p]["q_hi"] for p, k in lv]
    add11("enclosure", r"$q_p$ of $T_{1/2}$, four levels", r"certified brackets, $J=16384$",
          r"closed form \citep{AP97}", "all inside", "E2_sn_critical_values.json; E0_anchors.json", all(ins))
    dW = E0d["d"]["cvm_J4096"]
    add11("enclosure", r"Cram\'er--von Mises law, three levels", r"certified bridge engine, $J=4096$",
          "closed-form quantiles", "all inside", "E0_anchors.json (d)", all(v["contains"] for v in dW.values()))
    bH, bE = E0d["b"]["HI"], E0d["b"]["EWC"]
    kH = sorted({r["k"] for r in bH["rows"]}); kE = sorted({r["k"] for r in bE["rows"]})
    add11("enclosure", rf"fixed smoothing at $H=\frac12$, $m\le{max(kH)}$, $B\le{max(kE)}$", "certified Ruben series",
          r"Student $t$ and $F$ laws", "all inside", "E0_anchors.json (b)", bH["all_certified_contain"] and bE["all_certified_contain"])
    if "E5" in globals():
        t12 = E5[E5.target.str.startswith(("T1", "T2")) & E5.impl.isin(["imhof", "cqfimhof", "gx2imhof", "gx2auto", "ruben"])]
        dv = E5[E5.target.str.startswith(("T1", "T2")) & (E5.impl == "davies")]
        cw = E5[E5.target.str.startswith(("T1", "T2")) & (E5.impl == "certruben")]
        add11("enclosure", r"T1, T2: fixed smoothing, $H=0.8$", "certified Ruben series",
              "Imhof (three codes), Ruben (double)", rf"all inside; Davies {_s(float(dv.err_lo.max()), 1)} outside",
              "E5_methods.csv", bool((t12.err_lo == 0).all()))
        g4 = E5[(E5.target == "T4 T_H H=0.75 via Galerkin J=512") & (E5.impl == "imhof")].iloc[0]
        c4 = E5[E5.target == "T4 T_H H=0.75 (limit)"].iloc[0]
        add11("enclosure", r"T4: $F_{0.75}$ at its $95\,\%$ point", r"certified, $J=16384$", r"Imhof, $J=512$, remainder by its mean",
              rf"inside; error $\le$ {_s(float(g4.err_hi), 1)}", "E5_methods.csv; E2r/H0.75.json", float(g4.err_lo) == 0.0)
    # --- independent implementation
    a_ = E0d["a"]; c_ = E0d["c"]
    add11("implementation", r"closed forms at $H=\frac12$ ($T_{1/2}$; Cram\'er--von Mises)",
          "characteristic functions (mpmath)", r"Imhof over the explicit eigenvalues",
          rf"$\ge{min(v['sig_digits'] for v in a_.values()):.1f}$; $\ge{min(v['cvm_digits'] for v in c_.values()):.1f}$ digits",
          "E0_anchors.json (a, c)")
    if "E5" in globals():
        t3 = E5[E5.target.str.startswith("T3") & E5.impl.isin(["imhof", "cqfimhof", "davies", "gx2imhof"])]
        t3i = E5[E5.target.str.startswith("T3") & (E5.impl == "imhof")]
        add11("implementation", r"T3: exact law of $T_n$, $n=1000$", r"one eigendecomposition (\cref{prop:exactn}b)",
              "Imhof, Davies, gx2 on the ratio form", rf"$\le$ {_s(float(t3.err_hi.max()), 1)}", "E5_methods.csv")
    X10 = pd.read_csv(os.path.join(RES, "E10_matched_example.csv"))
    xn = X10[X10.cutoff_label == "Brownian cutoff"]
    add11("implementation", r"naive size of $T_n$, fGn, $n\le128$", "series route", r"$30$-digit Imhof; separate double-precision code",
          rf"{_s(float(X10.A_minus_B_pp.abs().max()), 1)}; {_s(float(xn.routeA_minus_user_pp.abs().max()), 1)}\,pp",
          "E10_matched_example.csv")
    if "pw-nct-maxdev" in NUM:
        add11("implementation", r"local power at $H=\frac12$, fixed smoothing", r"noncentral Imhof (\cref{prop:power})",
              r"noncentral Student $t$", rf"$\le$ {NUM['pw-nct-maxdev']}, {NUM['pw-nct-ncells']} cells", "E7_power_H0.50.csv; E7c_nct_check.json")
    # --- asymptotic consistency
    xq = X10[X10.cutoff_label == "H = 0.99 limit cutoff (certified point)"].iloc[0]
    add11("asymptotic", r"$q_{0.95}$ at $H=0.99$", "certified limit", r"exact law, fGn, $n=128$",
          rf"size {_m(xq.size_routeA_pct, 6)}\,\%", "E2_sn_critical_values.json; E10_matched_example.csv")
    def naive_lim(h):
        f2 = os.path.join(RES, "E2r", "J4096", f"H{h}.json")
        if os.path.exists(f2):
            r_ = json.load(open(f2)).get("E6")
            if r_:
                return r_["size_lo"], r_["size_hi"], "E2r/J4096"
        for f in glob.glob(os.path.join(RES, "E6_naive_sn_*.json")):
            d_ = json.load(open(f))
            if h in d_:
                return d_[h]["0.05"]["size_lo"], d_[h]["0.05"]["size_hi"], os.path.basename(f)
    nl = {h: naive_lim(h) for h in ("0.90", "0.99")}
    xs = {h: float(X10[(np.isclose(X10.H, float(h))) & (X10.n == 128) & (X10.cutoff_label == "Brownian cutoff")].size_routeA_pct.iloc[0]) for h in nl}
    dev = {h: abs(xs[h] - 50 * (nl[h][0] + nl[h][1])) for h in nl}
    add11("asymptotic", r"naive size at $H=0.9$, $0.99$", "certified limits", r"exact law, fGn, $n=128$",
          rf"{_s(dev['0.90'], 1)}; {_s(dev['0.99'], 1)}\,pp", f"{nl['0.90'][2]}; E10_matched_example.csv")
    E12d = json.load(open(os.path.join(RES, "E12_edge_limit.json")))
    ex, Cp = E12d["extrap_from_certified_0.98_0.99"]["0.950"], E12d["C_p"]["0.950"]
    add11("asymptotic", r"$(1-H)\,q_{0.95}(H)$, $H\to1$", r"certified, $H=0.98$, $0.99$, extrapolated",
          r"$C_{0.95}$ of \cref{prop:edge}", rf"{_m(100 * abs(ex / Cp - 1), 2)}\,\% relative", "E12_edge_limit.json")
    # --- probabilistic checks
    Vf2 = sorted(glob.glob(os.path.join(RES, "E3b_validate_*.csv")))
    V2 = pd.concat([pd.read_csv(f) for f in Vf2], ignore_index=True)
    add11("probabilistic", r"finite-sample rates (\cref{sec:finite})", "exact laws",
          rf"simulation, $R={grouped(int(V2.R.max()))}$, {len(V2)} cells",
          rf"{int(V2.exact_inside_ci99.sum())} inside $99\,\%$ intervals", "E3b_validate_*.csv")
    if "or-n-cells" in NUM:
        add11("probabilistic", r"oracle rates, estimated-$H$ study", r"exact laws, true $H$",
              rf"simulation, {NUM['or-n-cells']} cells", rf"{NUM['or-inside-pct']}\,\% inside $99\,\%$ intervals",
              "E3b_plugin grid; E3_rates_*.csv")
    if "hi17-t2-maxz" in NUM:
        add11("probabilistic", r"\citet[Table~2]{HI17}", r"exact finite-$T$ laws", "their simulation",
              rf"max $|z|={NUM['hi17-t2-maxz']}$, {NUM['hi17-t2-n']} cells", "E3c_hi2017_table2.csv")
    # --- convergence checks
    if "pw-jl-512-1024" in NUM:
        add11("convergence", r"Bartlett/SN local power", r"$J=512$", r"$J=256$ and $1024$",
              rf"$\le$ {NUM['pw-jl-256-512']}; $\le$ {NUM['pw-jl-512-1024']}", "E7b_sn_power_Jladder.json")
    if "e1-slope-maxdev" in NUM:
        add11("convergence", r"enclosure width of $F_H$", r"$J=512,\dots,8192$", r"rate $J^{-1/2-2H}$ (\cref{prop:rate})",
              rf"slopes within {_m(float(NUM['e1-slope-maxdev']), 2)}", "E1_benchmark_H*.json")
    # --- reproducibility
    same, nrow = [], 0
    for tb in ("3", "4", "5"):
        f = os.path.join(RES, f"E3b_hi2017_table{tb}.csv")
        if os.path.exists(f + ".bak16nov"):
            A_, B_ = pd.read_csv(f + ".bak16nov"), pd.read_csv(f); same.append(A_.equals(B_)); nrow += len(B_)
    f2 = os.path.join(RES, "E3c_hi2017_table2.csv")
    if need(len(same) == 3 and os.path.exists(f2 + ".bak16nov"), "E11 reruns"):
        A_, B_ = pd.read_csv(f2 + ".bak16nov"), pd.read_csv(f2)
        add11("reproducibility", r"\citet{HI17} Tables 2--5", "stored runs", "re-runs, same seeds",
              rf"Tables 3--5 bit-identical ({nrow} rows); Table 2 within {_s(float((A_.exact_rate - B_.exact_rate).abs().max()), 1)}",
              "E3b_hi2017_table{3,4,5}.csv, E3c_hi2017_table2.csv (and .bak16nov)", all(same))
    E2rf = sorted(glob.glob(os.path.join(RES, "E2r", "H*.json")))
    if need(len(E2rf) == len(SN), "E2r complete"):
        okb, nb, dmax = 0, 0, 0.0
        for f in E2rf:
            d_ = json.load(open(f))
            for p_, v_ in d_["sn"].items():
                nb += 1; okb += int(v_["bracket_ok"])
                for a, b in zip(v_["F_at_q_lo"] + v_["F_at_q_hi"], v_["old_F_at_q_lo"] + v_["old_F_at_q_hi"]):
                    dmax = max(dmax, abs(a - b))
        put("e2r-n-brackets", str(nb)); put("e2r-n-ok", str(okb)); put("e2r-maxchange", sci(dmax, 0))
        add11("reproducibility", rf"certified $q_p(H)$, {nb} values", "first implementation",
              r"final implementation (\cref{app:certificate})", rf"all confirmed; shifts $\le$ {sci(dmax, 0)}", "E2r/H*.json", okb == nb)
        CHECKS.append((f"E2r: all {nb} certified brackets confirmed by the corrected certificate", okb == nb))
    CHECKS.append(("E11: every check row passes its own criterion", all(r["ok"] for r in E11rows)))
    # --- write
    L = [r"\renewcommand{\arraystretch}{1.12}\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.245\textwidth}>{\raggedright\arraybackslash}p{0.205\textwidth}"
         r">{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.24\textwidth}@{}}\toprule",
         r"Quantity & Primary computation & Independent check & Agreement\\\midrule"]
    first = True
    for key, name, defin in CAT:
        rows_ = [r for r in E11rows if r["category"] == key]
        if not rows_:
            continue
        if not first:
            L.append(r"\addlinespace[4pt]")
        first = False
        L.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{name}}}: {defin}}}\\[1pt]")
        for r in rows_:
            L.append(" & ".join(r[c] for c in ("quantity", "primary", "independent", "agreement")) + r"\\")
    L.append(r"\bottomrule\end{tabular}")
    open(os.path.join(TAB, "tab_reproduction.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    pd.DataFrame(E11rows).to_csv(os.path.join(RES, "E11_reproduction.csv"), index=False)
    put("e11-n-rows", str(len(E11rows))); put("e11-n-cat", str(len({r["category"] for r in E11rows})))

# ------------------------------------------------------------------ E3c: numerical safeguards of the exact laws (17 Nov)
CLf = os.path.join(RES, "E3c_clip_check.json")
if need(os.path.exists(CLf), "E3c clip"):
    cl = json.load(open(CLf))
    put("clip-ncells", str(cl["summary"]["cells"])); put("clip-nwith", str(cl["summary"]["cells_with_clipping"]))
    put("clip-max", {1: "one", 2: "two", 3: "three"}.get(cl["summary"]["max_clipped"], str(cl["summary"]["max_clipped"])))
    put("clip-rel", sci(abs(cl["summary"]["most_negative_rel"]), 0)); put("arfima-trunc", sci(cl["arfima1_trunc_rel_bound"], 0))

# ------------------------------------------------------------------ write numbers
with open(os.path.join(TAB, "p3_numbers.tex"), "w", encoding="utf-8") as fh:
    fh.write("% generated by code/manuscript/make_numbers.py -- do not edit\n")
    for k in sorted(NUM):
        fh.write(f"\\pthreedef{{{k}}}{{{NUM[k]}}}\n")
used = set()
for f in glob.glob(os.path.join(WORK, "manuscript", "sections", "*.tex")):
    used |= set(re.findall(r"\\pthree\{([^}]+)\}", open(f, encoding="utf-8").read()))
undef = sorted(used - set(NUM))
print(f"wrote {len(NUM)} numbers; used in text: {len(used)}; undefined: {len(undef)}")
for k in undef:
    print("  UNDEFINED", k)
for k in MISSING:
    print("  missing input:", k)
for c, ok in CHECKS:
    print("  CHECK", "ok " if ok else "FAIL", c)
