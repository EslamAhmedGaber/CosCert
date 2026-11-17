"""Figure 2 (version B, 16 Nov): what size a nominal 5 % test actually has at sample size n.

Row 1 -- H known: exact finite-sample size (E3, finite-dimensional exact laws), each test at its own limit critical value at
         the true H.  Lines: H = 0.7; bands: range over H in {0.5, ..., 0.9}; open circles: EWC (B = 2m) at H = 0.7.
Row 2 -- H estimated: simulated size (E3b plug-in, R = 40 000 per cell, Wilson 99 % intervals smaller than the markers)
         with the local Whittle estimate (bandwidth floor(n^0.65)) clipped to [1/2, 0.99] for every procedure, plugged into
         the same limit critical values.  Markers + lines: H = 0.7; bands: range over H.
Columns = data models.  Shared log y-axis: over- and under-rejection by the same factor are equally far from 5 %.
One message: memory itself costs little once the critical value is right, short-run dynamics cost a lot, and estimating
H costs more than either at practical n.
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter  # noqa: E402

RES = os.path.join(F.WORK, "results")
MODELS = [("fgn", "fGn"), ("arfima0", r"ARFIMA$(0,d,0)$"), ("arfima1p", r"ARFIMA$(1,d,0)$, $\phi{=}0.5$"), ("arfima1m", r"$\phi{=}-0.5$")]
H_LINE = 0.7
HI_SHOW, EWC_SHOW = (1, 4, 16), (8, 32)


def fmt_axes(ax, row, col):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: {50: "50", 100: "100", 200: "200", 500: "500", 1000: "1000", 2000: "2000"}.get(int(round(v)), "")))
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(42, 2400); ax.set_ylim(1.6, 26)
    ax.yaxis.set_major_locator(FixedLocator([2, 3, 4, 5, 6, 8, 10, 15, 20, 25]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.axhline(5, color=F.COL["REF"], lw=0.7, ls=(0, (4, 2)), zorder=0)
    if col:
        ax.tick_params(labelleft=False)
    if row == 0:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel(r"sample size $n$")


def main():
    files = sorted(glob.glob(os.path.join(RES, "E3_rates_*.csv")))
    E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    E = E[(E.cv_label == "own_limit") & (np.isclose(E.alpha, 0.05))].copy()
    E["k"] = pd.to_numeric(E.param, errors="coerce").fillna(0).astype(int); E["size"] = 100 * E.rejection_rate
    pfiles = sorted(glob.glob(os.path.join(RES, "E3b_plugin_v2_*.csv")))
    P = pd.concat([pd.read_csv(f) for f in pfiles], ignore_index=True)
    P = P[np.isclose(P.alpha, 0.05)].copy(); P["k"] = pd.to_numeric(P.param, errors="coerce").fillna(0).astype(int); P["size"] = 100 * P.mc_rate
    # one rule for all procedures: H-hat clipped to [1/2, 0.99] (SN: plugin_exact already clipped; HI/EWC: plugin_exact_trunc)
    P = P[((P.procedure == "SN") & (P.rule == "plugin_exact")) | ((P.procedure != "SN") & (P.rule == "plugin_exact_trunc"))]
    hi_col = dict(zip(HI_SHOW, [F.tint(F.COL["HI"], 0.42), F.COL["HI"], F.shade(F.COL["HI"], 0.5)]))
    ewc_col = dict(zip(EWC_SHOW, [F.COL["EWC"], F.shade(F.COL["EWC"], 0.45)]))
    series = [("SN", 0, F.COL["SN"], 1.8)] + [("HI", m, hi_col[m], 1.15) for m in HI_SHOW]

    fig = plt.figure(figsize=(F.TEXTWIDTH, 4.55))
    gs = fig.add_gridspec(2, 4, left=0.083, right=0.99, bottom=0.095, top=0.885, wspace=0.08, hspace=0.12)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
    for c, (mod, title) in enumerate(MODELS):
        for r, (D, marker) in enumerate(((E, None), (P, "o"))):
            ax = axes[r][c]; S = D[D.model == mod]
            for proc, k, col, lw in series:
                s = S[(S.procedure == proc) & (S.k == k)]
                band = s.groupby("n")["size"].agg(["min", "max"])
                ax.fill_between(band.index, band["min"], band["max"], color=col, alpha=0.16, lw=0, zorder=1)
                line = s[np.isclose(s.H, H_LINE)].sort_values("n")
                ax.plot(line.n, line["size"], color=col, lw=lw, zorder=4 if proc == "SN" else 3,
                        marker=marker, ms=2.6 if marker else 0, mfc=col, mec=col)
            for B in EWC_SHOW:
                s = S[(S.procedure == "EWC") & (S.k == B) & np.isclose(S.H, H_LINE)].sort_values("n")
                ax.plot(s.n, s["size"], ls="none", marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.85, zorder=5)
            fmt_axes(ax, r, c)
            if r == 0:
                ax.set_title(title, fontsize=F.SIZES[0])
            ax.text(0.04, 0.965, r"\textbf{(" + "abcdefgh"[4 * r + c] + ")}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=F.SIZES[0], color=F.COL["INK"])
    axes[0][0].text(0.96, 0.05, r"bands: range over $H$", transform=axes[0][0].transAxes, ha="right", va="bottom",
                    fontsize=F.SIZES[2], color=F.COL["INK"])   # bands are variation over H, not uncertainty (referee, 16 Nov)
    axes[0][0].set_ylabel(r"$H$ known: exact size (\%)")
    axes[1][0].set_ylabel(r"$H$ estimated: simulated size (\%)")
    handles = [Line2D([], [], color=F.COL["SN"], lw=1.8, label=r"SN (Bartlett, $b{=}1$)")]
    handles += [Line2D([], [], color=hi_col[m], lw=1.15, label=rf"HI $m{{=}}{m}$") for m in HI_SHOW]
    handles += [Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.85, label=rf"EWC $B{{=}}{B}$") for B in EWC_SHOW]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=6, frameon=False, handlelength=1.6,
               columnspacing=1.2, handletextpad=0.4, fontsize=F.SIZES[1])
    # data checks printed for the record (used in the caption / text)
    pure = P[P.model.isin(["fgn", "arfima0"]) & (P.n == 100) & np.isclose(P.H, 0.9)]
    print("fig2: plug-in size at n=100, H=0.9 (fGn/ARFIMA0):", pure.groupby(["model", "procedure", "k"])["size"].first().round(2).to_dict())
    return fig, files + pfiles


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig2_finite_n", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
