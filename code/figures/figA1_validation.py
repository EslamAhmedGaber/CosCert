"""Appendix figure: validation of the exact finite-sample laws.
(a) simulated minus exact rejection rate (pp) for the 192 AT4 cells, bars = 99 % Wilson intervals, colour = procedure;
(b) standardised differences against N(0,1) quantiles;
(c) Hualde-Iacone (2017) Table 2 published rates minus our exact rates, bars = +-2 s.e. of their 10,000 replications.
"""
import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = os.path.join(F.WORK, "results")


def pcol(proc, k):
    if proc == "SN":
        return F.COL["SN"]
    base = F.COL["HI"] if proc == "HI" else F.COL["EWC"]
    return {1: F.tint(base, 0.45), 4: base, 16: F.shade(base, 0.5), 8: base, 32: F.shade(base, 0.5)}.get(int(k), base)


def main():
    vf = sorted(glob.glob(os.path.join(RES, "E3b_validate_*.csv"))); V = pd.concat([pd.read_csv(f) for f in vf], ignore_index=True)
    V["k"] = pd.to_numeric(V.param, errors="coerce").fillna(0).astype(int)
    t2f = os.path.join(RES, "E3c_hi2017_table2.csv"); T2 = pd.read_csv(t2f).dropna(subset=["published_rate"])
    fig = plt.figure(figsize=(F.TEXTWIDTH, 2.55))
    gs = fig.add_gridspec(1, 3, left=0.075, right=0.985, bottom=0.17, top=0.80, wspace=0.42, width_ratios=[1.3, 0.85, 1.15])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    V = V.sort_values(["exact_rate"])
    for _, r in V.iterrows():
        col = pcol(r.procedure, r.k); x = 100 * r.exact_rate
        a.plot([x, x], [100 * (r.ci99_lo - r.exact_rate), 100 * (r.ci99_hi - r.exact_rate)], color=col, lw=0.5, alpha=0.6, zorder=2)
        a.plot(x, 100 * (r.mc_rate - r.exact_rate), "o", ms=2.4, color=col, zorder=3)
    a.axhline(0, color=F.COL["REF"], lw=0.6); a.set_xscale("log")
    a.set_xticks([2, 5, 10, 20, 50]); a.set_xticklabels(["2", "5", "10", "20", "50"]); a.minorticks_off()
    a.set_xlabel(r"exact rejection rate (\%)"); a.set_ylabel(r"simulated $-$ exact (pp)")
    inside = int(V.exact_inside_ci99.sum())
    a.set_title(rf"{inside} of {len(V)} inside the $99\,\%$ intervals")

    # second, larger check: oracle rule of the estimated-H grid (true-H critical value) against the exact E3 rates
    pf2 = sorted(glob.glob(os.path.join(RES, "E3b_plugin_v2_*.csv"))); P2 = pd.concat([pd.read_csv(f) for f in pf2], ignore_index=True)
    O = P2[P2.rule == "oracle_exact"].copy(); O["k"] = pd.to_numeric(O.param, errors="coerce").fillna(0).astype(int)
    E3 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "E3_rates_*.csv"))], ignore_index=True)
    E3 = E3[E3.cv_label == "own_limit"].copy(); E3["k"] = pd.to_numeric(E3.param, errors="coerce").fillna(0).astype(int)
    O = O.merge(E3[["model", "H", "n", "procedure", "k", "alpha", "rejection_rate"]], on=["model", "H", "n", "procedure", "k", "alpha"])
    zo = np.sort((O.mc_rate - O.rejection_rate) / np.sqrt(O.rejection_rate * (1 - O.rejection_rate) / O.R))
    qo = stats.norm.ppf((np.arange(1, zo.size + 1) - 0.5) / zo.size)
    print(f"figA1(b): oracle cells {zo.size}, z mean {zo.mean():.3f}, sd {zo.std():.3f}")
    z = np.sort(V.z.values); qn = stats.norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
    b.plot([-4, 4], [-4, 4], color=F.COL["REF"], lw=0.6)
    b.plot(qo, zo, "o", ms=1.6, color="#B8B8B8", mec="none", zorder=2, label=rf"{zo.size} oracle")
    b.plot(qn, z, "o", ms=2.0, color=F.COL["INK"], zorder=3, label=rf"{z.size} validation")
    b.legend(loc="lower right", fontsize=F.SIZES[2], frameon=False, handletextpad=0.15, borderaxespad=0.2, markerscale=1.5)
    b.set_xlim(-4.1, 4.1); b.set_ylim(-4.1, 4.1); b.set_aspect("equal", adjustable="box")
    b.set_xticks([-4, -2, 0, 2, 4]); b.set_yticks([-4, -2, 0, 2, 4])
    b.set_xlabel(r"$N(0,1)$ quantile"); b.set_ylabel("standardised difference")
    b.set_title(r"Standardised: $N(0,1)$")

    T2 = T2.sort_values(["T", "critical_value", "m"])
    for (T, cvl), g in T2.groupby(["T", "critical_value"]):
        col = F.COL["HI"] if cvl == "t2m" else F.COL["REF"]; mk = "o" if T == 64 else "s"
        se = np.sqrt(g.exact_rate * (1 - g.exact_rate) / g.N)
        c.errorbar(g.m + (0.12 if T == 256 else -0.12), 100 * (g.published_rate - g.exact_rate), yerr=200 * se, fmt=mk, ms=2.8, color=col,
                   mfc=col if T == 64 else "white", elinewidth=0.5, capsize=0)
    c.axhline(0, color=F.COL["REF"], lw=0.6)
    c.set_xlabel(r"number of frequencies $m$"); c.set_ylabel(r"published $-$ exact (pp)")
    c.set_title(r"HI Table 2: max $|z|=" + f"{T2.z.abs().max():.2f}" + r"$")
    hand = [Line2D([], [], ls="none", marker="o", ms=2.8, color=F.COL["HI"], label=r"$t_{2m}$, 64"),
            Line2D([], [], ls="none", marker="s", ms=2.8, mfc="white", color=F.COL["HI"], label=r"$t_{2m}$, 256"),
            Line2D([], [], ls="none", marker="o", ms=2.8, color=F.COL["REF"], label=r"$N(0,1)$, 64"),
            Line2D([], [], ls="none", marker="s", ms=2.8, mfc="white", color=F.COL["REF"], label=r"$N(0,1)$, 256")]
    c.legend(handles=hand, loc="lower left", fontsize=F.SIZES[2], ncol=2, columnspacing=0.6, handletextpad=0.15, frameon=False,
             borderaxespad=0.1, title=r"cut-off, $T$", title_fontsize=F.SIZES[2])
    c.set_ylim(-1.75, 1.1)
    for i, axx in enumerate((a, b, c)):
        F.panel_tag(axx, "abc"[i])
    hp = [Line2D([], [], ls="none", marker="o", ms=2.6, color=pcol(p_, k_), label=l_) for p_, k_, l_ in
          (("SN", 0, "SN"), ("HI", 1, r"HI $m{=}1$"), ("HI", 4, r"HI $m{=}4$"), ("HI", 16, r"HI $m{=}16$"), ("EWC", 8, r"EWC $B{=}8$"), ("EWC", 32, r"EWC $B{=}32$"))]
    fig.legend(handles=hp, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=F.SIZES[1], handletextpad=0.2,
               columnspacing=1.3)
    return fig, vf + [t2f] + pf2


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "figA1_validation", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
