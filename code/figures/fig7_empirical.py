"""Figure 4 (file fig7_empirical): the Nile annual minima, AD 622-1284, and nominal 95 % intervals for their mean (E13).

(a) The series, its sample mean (solid), the illustrative level mu0 = 1100 cm (dashed) and the nominal 95 % intervals of
    the Bartlett/self-normalised test with the short-memory cutoff (grey band) and the CosCert cutoff at H_LW (blue band).
(b) Nominal 95 % intervals for the mean from every procedure of Section 6: short-memory cutoffs (thin grey, end caps),
    CosCert cutoffs at H_LW (thick, coloured by family), ADG-MAC and rate-free self-normalised subsampling.
"""
import csv
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RES = os.path.join(F.WORK, "results")
DATA = os.path.join(F.WORK, "data")


def main():
    e13_path = os.path.join(RES, "E13_empirical.json")
    rec = json.load(open(e13_path, encoding="utf-8"))["cases"][0]
    data_path = os.path.join(DATA, "nile_minima_622_1284.csv")
    with open(data_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    yr = np.array([float(r["year_ad"]) for r in rows]); y = np.array([float(r["min_level_cm"]) for r in rows])
    mu0, xbar, T = float(rec["mu0"]), float(rec["xbar"]), rec["tests"]

    fig = plt.figure(figsize=(F.TEXTWIDTH, 2.62))
    gs = fig.add_gridspec(1, 2, left=0.083, right=0.968, bottom=0.16, top=0.825, wspace=0.30, width_ratios=[1.12, 1.0])

    # (a) series with the SN intervals as bands
    ax = fig.add_subplot(gs[0, 0])
    lo_n, hi_n = T["SN"]["ci95"]["naive"]; lo_c, hi_c = T["SN"]["ci95"]["lw"]
    ax.axhspan(lo_c, hi_c, color=F.tint(F.COL["SN"], 0.20), lw=0, zorder=0)
    ax.axhspan(lo_n, hi_n, color="#C9C9C9", lw=0, zorder=1)
    ax.plot(yr, y, color="#4A4A4A", lw=0.5, zorder=2)
    ax.axhline(xbar, color=F.COL["SN"], lw=1.0, zorder=3)
    ax.axhline(mu0, color=F.COL["INK"], lw=0.85, ls=(0, (4, 2)), zorder=3)
    ax.set_xlim(yr[0], yr[-1]); ax.set_ylim(880, 1480)
    ax.set_xticks([700, 800, 900, 1000, 1100, 1200]); ax.set_yticks([900, 1000, 1100, 1200, 1300, 1400])
    ax.set_xlabel("year AD"); ax.set_ylabel("annual minimum (cm)")
    ax.set_title(rf"Nile minima, $n={rec['n']}$, $\hat H_{{\mathrm{{LW}}}}={rec['H_lw']:.2f}$", fontsize=F.SIZES[1])
    F.panel_tag(ax, "a")

    # (b) forest plot of nominal 95 % intervals
    bx = fig.add_subplot(gs[0, 1])
    order = [("SN", "SN", F.COL["SN"]), ("HI4", r"HI, $m{=}4$", F.COL["HI"]), ("HI16", r"HI, $m{=}16$", F.shade(F.COL["HI"], 0.40)),
             ("EWC8", r"EWC, $B{=}8$", F.COL["EWC"]), ("EWC32", r"EWC, $B{=}32$", F.shade(F.COL["EWC"], 0.40)),
             ("ADG_MAC", "ADG--MAC", F.COL["MP"]), ("Subsampling_SN", "subsampling", F.COL["REF"])]
    ypos = np.arange(len(order))[::-1].astype(float)
    bx.axvline(mu0, color=F.COL["INK"], lw=0.85, ls=(0, (4, 2)), zorder=1)
    bx.axvline(xbar, color="#D0D0D0", lw=0.7, zorder=0)
    for yp, (k, lab, col) in zip(ypos, order):
        t = T[k]
        if k in ("ADG_MAC", "Subsampling_SN"):
            lo, hi = t["ci95"]
            bx.plot([lo, hi], [yp, yp], color=col, lw=1.6, solid_capstyle="butt", zorder=3)
            bx.plot([xbar], [yp], marker="D" if k == "ADG_MAC" else "s", ms=3.0, color=col, zorder=4)
            continue
        lo_n, hi_n = t["ci95"]["naive"]; lo_c, hi_c = t["ci95"]["lw"]
        bx.plot([lo_c, hi_c], [yp - 0.14, yp - 0.14], color=col, lw=2.2, solid_capstyle="butt", zorder=3)
        bx.plot([xbar], [yp - 0.14], marker="o", ms=2.8, color=col, zorder=4)
        bx.plot([lo_n, hi_n], [yp + 0.17, yp + 0.17], color="#8A8A8A", lw=1.0, zorder=2)
        for e in (lo_n, hi_n):
            bx.plot([e, e], [yp + 0.07, yp + 0.27], color="#8A8A8A", lw=0.8, zorder=2)
    bx.set_yticks(ypos); bx.set_yticklabels([o[1] for o in order], fontsize=F.SIZES[2])
    bx.set_ylim(-0.6, len(order) - 0.4); bx.tick_params(axis="y", length=0); bx.spines["left"].set_visible(False)
    bx.set_xlim(900, 1400); bx.set_xticks([900, 1000, 1100, 1200, 1300, 1400])
    bx.set_xlabel(r"mean level $\mu$ (cm)")
    bx.set_title(r"Nominal $95\,\%$ intervals for $\mu$", fontsize=F.SIZES[1])
    F.panel_tag(bx, "b")

    handles = [Line2D([], [], color=F.COL["SN"], lw=1.0, label="sample mean"),
               Line2D([], [], color=F.COL["INK"], lw=0.85, ls=(0, (4, 2)), label=rf"$\mu_0={mu0:.0f}$\,cm"),
               Patch(facecolor="#C9C9C9", edgecolor="#8A8A8A", lw=0.6, label=r"short-memory cutoff ($H=\frac12$)"),
               Patch(facecolor=F.tint(F.COL["SN"], 0.20), edgecolor=F.COL["SN"], lw=0.6, label=r"CosCert cutoff at $\hat H_{\mathrm{LW}}$")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.525, 1.005), ncol=4, frameon=False,
               fontsize=F.SIZES[1], handlelength=1.7, columnspacing=1.3, handletextpad=0.45)
    return fig, [e13_path, data_path]


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig7_empirical", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
