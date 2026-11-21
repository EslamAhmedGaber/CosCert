"""Figure 6 (manuscript numbering; file fig5_methods; version C, 17 Nov 2026): computational methods on identical targets (E5).

Left: error of the distribution function at the 95 % quantile in test-size percentage points; right: wall-clock seconds
including each method's own spectral step.  Rows = methods grouped by family; marker shape = target:
  T1 (circle)   fixed-m HI limit, m = 8, H = 0.8 (low rank)     [T2, EWC B = 16, gives the same picture: caption]
  T3 (square)   exact finite-n SN law, n = 1000, ARFIMA(0, 0.2, 0)
  T4 (triangle) limit law of T_H, H = 0.75 (double-precision methods on the J = 512 cosine coordinates; certified at 16384)
Errors (audit B4): against a certified enclosure [L, U] the true error of a value v lies in [dist(v,[L,U]), max|v - L|, |v - U|];
  filled marker = resolved (dist > 0, drawn at the interval's upper end), open marker + line to the floor = upper bound only
  (v inside the enclosure); T3 has a computed reference (|v - ref|, filled).  Certified rows: grey bar from the floor to the
  guaranteed half-width.  Values at or below 1e-12 pp are drawn in the shaded floor band.  Crosses: no value (overflow, memory).
Cost (audit B1): exact finite-n law, filled = set-up + first cutoff, open = each further cutoff.
Rows outside the documented domain (LTZ) and outer-quadrature diagnostics (Gauss-Hermite) are not drawn.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter  # noqa: E402

RES = os.path.join(F.WORK, "results")
FLOOR = 1e-12
XFAIL = 150.0
GROUPS = [
    ("This paper", "#1A1A1A", [(r"CosCert, certified", ["certified"]),
                              (r"exact finite-$n$ law", ["setup + first query"]),
                              (r"tilted Bromwich, double", ["Bromwich tilted midpoint"])]),
    ("Exact inversion", "#4D4D4D", [(r"Imhof, SciPy", ["Imhof (scipy"]), (r"Imhof, CompQuadForm (R)", ["CompQuadForm::imhof"]),
                                    (r"Davies, CompQuadForm (R)", ["CompQuadForm::davies"]),
                                    (r"gx2 imhof and auto", ["gx2 imhof"]), (r"Ruben series, double", ["Ruben-Beta"]),
                                    (r"gx2 ifft", ["gx2 ifft"])]),
    ("Sampling", "#666666", [(r"gx2 ray (MC ray-tracing)", ["gx2 ray"]), (r"Monte Carlo", ["Monte Carlo"])]),
    ("Approximation", "#7F7F7F", [(r"leading 20 + Satterthwaite", ["leading-20"]), (r"LPB4, momentchi2 (R)", ["lpb4 | adaptive"]),
                                  (r"saddlepoint, survey (R)", ["pchisqsum saddlepoint | adaptive"]),
                                  (r"Lugannani--Rice saddlepoint", ["Lugannani-Rice"]), (r"HBE, momentchi2 (R)", ["hbe | adaptive"]),
                                  (r"Satterthwaite, survey (R)", ["pchisqsum satterthwaite | adaptive"]), (r"gx2 pearson", ["gx2 pearson"])]),
]
TARGETS = [("T1", "T1 HI m=8 H=0.80", None, "o"), ("T3", "T3 SN n=1000 ARFIMA(0,0.2,0)", None, "s"),
           ("T4", "T4 T_H H=0.75 via Galerkin J=512", "T4 T_H H=0.75 (limit)", "^")]
DY = {"T1": -0.24, "T3": 0.0, "T4": 0.24}
MS = 4.8


def find(M, target, pats):
    s = M[(M.target == target) & M.method.apply(lambda m: any(p in m for p in pats))]
    return None if not len(s) else s.iloc[0]


def main():
    path = os.path.join(RES, "E5_methods.csv"); M = pd.read_csv(path)
    rows, heads, y = [], [], 0.0
    for g, col, items in GROUPS:
        heads.append((y, g)); y += 1.0
        for lab, pats in items:
            rows.append((y, lab, g, col, pats)); y += 1.0
        y += 0.35
    ymax = y
    fig = plt.figure(figsize=(F.TEXTWIDTH, 5.15))
    gs = fig.add_gridspec(1, 2, left=0.285, right=0.985, bottom=0.095, top=0.875, wspace=0.08, width_ratios=[1.45, 1.0])
    ae, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for ax in (ae, at):
        ax.set_ylim(ymax - 0.2, -0.8); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        for (yy, *_r) in rows:
            ax.axhline(yy, color="#F0F0F0", lw=0.6, zorder=0)
    ae.axvspan(1e-15, FLOOR, color="#EDEDED", lw=0, zorder=0)
    ae.axvline(1e-2, color=F.COL["REF"], lw=0.7, ls=(0, (4, 2)), zorder=1)
    ae.text(1.25e-2, -0.5, r"$0.01$\,pp", fontsize=F.SIZES[1], color=F.COL["REF"], ha="left", va="center")
    n_plot, n_fail, n_bound = 0, 0, 0
    for (yy, lab, g, col, pats) in rows:
        for tname, tkey, tcert, mk in TARGETS:
            key = tcert if (tcert and "certified" in pats) else tkey
            r = find(M, key, pats)
            if r is None:
                continue
            yv = yy + DY[tname]
            if not np.isfinite(r.value):
                ae.plot([XFAIL], [yv], marker="x", ms=MS, color=col, ls="none", zorder=3, mew=1.0)
                ae.text(XFAIL * 1.35, yv, tname, fontsize=5.5, color=col, ha="left", va="center", zorder=3)
                n_fail += 1
                continue
            lo, hi = 100 * float(r.err_lo), 100 * float(r.err_hi)
            if "certified" in pats:
                b = max(hi, FLOOR)
                ae.plot([FLOOR, b], [yv, yv], color="#9A9A9A", lw=1.6, solid_capstyle="butt", zorder=2)
                ae.plot([b], [yv], marker=mk, ms=MS, color=col, ls="none", zorder=3)
            elif r.ref_kind == "enclosure" and lo <= 0.0:
                b = max(hi, FLOOR)
                ae.plot([FLOOR, b], [yv, yv], color=col, lw=0.6, zorder=2)
                ae.plot([b], [yv], marker=mk, ms=MS, mfc="white", mec=col, mew=0.9, ls="none", zorder=3); n_bound += 1
            elif "setup + first query" in pats:          # the T3 reference itself
                ae.plot([FLOOR * 0.6], [yv], marker=mk, ms=MS, color=col, ls="none", zorder=3)
                ae.text(FLOOR * 2.2, yv, "reference", fontsize=F.SIZES[2], color=col, ha="left", va="center")
            else:
                ae.plot([max(hi, FLOOR * 0.6)], [yv], marker=mk, ms=MS, color=col, ls="none", zorder=3)
            sec = float(r.seconds_total)
            at.plot([sec], [yv], marker=mk, ms=MS, color=col, ls="none", zorder=3); n_plot += 1
            if "setup + first query" in pats:          # further cutoffs of the exact finite-n law
                rr = M[(M.target == tkey) & M.method.str.contains("repeated query", regex=False)]
                if len(rr):
                    at.plot([float(rr.seconds_method.iloc[0])], [yv], marker=mk, ms=MS, mfc="white", mec=col, mew=0.9, ls="none", zorder=3)
    ae.set_xscale("log"); ae.set_xlim(2e-13, 600)
    ae.xaxis.set_major_locator(FixedLocator([1e-12, 1e-9, 1e-6, 1e-3, 1, 100]))
    ae.xaxis.set_major_formatter(FuncFormatter(lambda v, p: {1e-12: r"$10^{-12}$", 1e-9: r"$10^{-9}$", 1e-6: r"$10^{-6}$",
                                                             1e-3: r"$10^{-3}$", 1: r"$1$", 100: r"$100$"}.get(v, "")))
    ae.xaxis.set_minor_formatter(NullFormatter()); ae.xaxis.set_minor_locator(FixedLocator([]))
    at.set_xscale("log"); at.set_xlim(1e-4, 1e4)
    at.xaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 100, 1e4]))
    at.xaxis.set_major_formatter(FuncFormatter(lambda v, p: {1e-4: r"$10^{-4}$", 1e-2: r"$10^{-2}$", 1: r"$1$", 100: r"$100$",
                                                             1e4: r"$10^{4}$"}.get(v, "")))
    at.xaxis.set_minor_formatter(NullFormatter()); at.xaxis.set_minor_locator(FixedLocator([]))
    for ax in (ae, at):
        ax.tick_params(axis="x", labelsize=F.SIZES[1])
    ae.set_xlabel(r"size error (pp) at the 95\,\% quantile", fontsize=F.SIZES[0]); at.set_xlabel(r"time (s)", fontsize=F.SIZES[0])
    ae.set_yticks([r[0] for r in rows]); ae.set_yticklabels([r[1] for r in rows], fontsize=F.SIZES[0])
    for lab_, (yy, _l, g, col, _p) in zip(ae.get_yticklabels(), rows):
        lab_.set_color(col)
    at.set_yticks([])
    for (yh, g) in heads:
        ae.text(-0.02, yh, g, transform=ae.get_yaxis_transform(), ha="right", va="center", fontsize=F.SIZES[0], fontweight="bold",
                color=F.COL["INK"])
    ae.set_title("Accuracy", fontsize=F.SIZES[0]); at.set_title("Cost", fontsize=F.SIZES[0])
    F.panel_tag(ae, "a"); F.panel_tag(at, "b")
    hd = [Line2D([], [], ls="none", marker=mk, ms=MS, color=F.COL["INK"], label=lab) for lab, mk in
          ((r"T1 fixed-$m$ limit", "o"), (r"T3 exact law, $n{=}1000$", "s"), (r"T4 limit of $T_H$", "^"))]
    hd += [Line2D([], [], ls="none", marker="o", ms=MS, mfc="white", mec=F.COL["INK"], label="open (a): inside enclosure"),
           Line2D([], [], ls="none", marker="s", ms=MS, mfc="white", mec=F.COL["INK"], label="open (b): repeat cutoff"),
           Line2D([], [], ls="none", marker="x", ms=MS, color=F.COL["INK"], label="no value")]
    fig.legend(handles=hd[:3], loc="upper center", bbox_to_anchor=(0.61, 1.0), ncol=3, frameon=False, fontsize=F.SIZES[1],
               handletextpad=0.25, columnspacing=1.3)
    fig.legend(handles=hd[3:], loc="upper center", bbox_to_anchor=(0.61, 0.965), ncol=3, frameon=False, fontsize=F.SIZES[1],
               handletextpad=0.25, columnspacing=1.0)
    print(f"fig5: points plotted {n_plot}, upper bounds {n_bound}, failures {n_fail}")
    return fig, [path]


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig5_methods", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
