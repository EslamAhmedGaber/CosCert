"""Figure 7: Empirical applications across three long-memory regimes (Section 6, E13).

Three rows = three contiguous real-world series spanning moderate to strong long memory:
  Row 1: Australian Electricity Demand (NSW daily YoY % growth, 2012-2015, n = 1095, H_hat ~ 0.64), H0: mu = 0 %.
  Row 2: London Urban PM2.5 (GR9 daily log ug/m^3, 2017-2018, n = 454, H_hat ~ 0.78), H0: mu = log(10).
  Row 3: Nile River Annual Minima (Roda Gauge, 622-1284 AD, n = 663, H_hat ~ 0.91), H0: mu = 1100 cm.

Left column ((a), (c), (e)): observed series, sample mean (solid) and null value mu0 (dashed).
Right column ((b), (d), (f)): 95 % confidence intervals for mu under naive short-memory critical values (thin grey)
versus CosCert critical values at H_hat_LW (thick coloured bars) and ADG-MAC (diamond), highlighting decision reversals.
"""
from datetime import date
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

RES = os.path.join(F.WORK, "results")
DATA = os.path.join(F.WORK, "data")


def load_xy(csv_name, x_col, y_col, is_date=False):
    xs, ys = [], []
    with open(os.path.join(DATA, csv_name), newline="", encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if is_date:
                day = date.fromisoformat(r[x_col])
                start = date(day.year, 1, 1)
                end = date(day.year + 1, 1, 1)
                xs.append(day.year + (day - start).days / (end - start).days)
            else:
                xs.append(float(r[x_col]))
            ys.append(float(r[y_col]))
    return np.asarray(xs), np.asarray(ys)


def main():
    e13_path = os.path.join(RES, "E13_empirical.json")
    e13 = json.load(open(e13_path, encoding="utf-8"))
    by_id = {c["id"]: c for c in e13["cases"]}

    specs = [
        (
            "aus_elec_nsw", "aus_elec_nsw_yoy_2012_2015.csv", "date", "yoy_log_growth_pct", True,
            r"YoY demand (\%)", r"year", r"$\mu_0=0\,\%$",
            r"AusElec-NSW ($n{=}1095$, $\hat H{=}0.64$)", (-42.0, 44.0),
        ),
        (
            "london_pm25_who05", "london_gr9_pm25_2017_2018.csv", "date", "log_pm25", True,
            r"$\log(\mathrm{PM}_{2.5})$", r"year", r"$\mu_0=\log 10$",
            r"London GR9 $\mathrm{PM}_{2.5}$ ($n{=}454$, $\hat H{=}0.78$)", (-1.3, 5.4),
        ),
        (
            "nile_minima", "nile_minima_622_1284.csv", "year_ad", "min_level_cm", False,
            r"min.\ level (cm)", r"year AD", r"$\mu_0=1100$\,cm",
            r"Nile annual minima ($n{=}663$, $\hat H{=}0.91$)", (900.0, 1580.0),
        ),
    ]

    test_order = [
        ("SN", "SN", F.COL["SN"]),
        ("HI4", r"HI, $m{=}4$", F.COL["HI"]),
        ("HI16", r"HI, $m{=}16$", F.shade(F.COL["HI"], 0.45)),
        ("EWC8", r"EWC, $B{=}8$", F.COL["EWC"]),
        ("EWC32", r"EWC, $B{=}32$", F.shade(F.COL["EWC"], 0.45)),
        ("ADG_MAC", r"ADG--MAC", F.COL["MP"]),
    ]

    fig = plt.figure(figsize=(F.TEXTWIDTH, 5.15))
    gs = fig.add_gridspec(3, 2, left=0.095, right=0.985, bottom=0.075, top=0.90, wspace=0.28, hspace=0.44,
                          width_ratios=[1.15, 1.0])

    inputs = [e13_path]
    for row_i, (cid, fname, xcol, ycol, is_dt, ylab, xlab, mu_str, title_str, ylims) in enumerate(specs):
        inputs.append(os.path.join(DATA, fname))
        rec = by_id[cid]
        xs, ys = load_xy(fname, xcol, ycol, is_date=is_dt)
        mu0 = float(rec["mu0"])
        xbar = float(rec["xbar"])

        # Left panel: time series
        axL = fig.add_subplot(gs[row_i, 0])
        axL.plot(xs, ys, color="#4A4A4A", lw=0.55, alpha=0.85, zorder=2)
        axL.axhline(mu0, color=F.COL["REF"], lw=0.85, ls=(0, (4, 2)), zorder=3)
        axL.axhline(xbar, color=F.COL["SN"], lw=1.0, zorder=4)
        axL.set_ylabel(ylab, fontsize=F.SIZES[1])
        axL.set_title(title_str, fontsize=F.SIZES[1])
        if row_i == 2:
            axL.set_xlabel(xlab, fontsize=F.SIZES[1])
        axL.set_xlim(xs[0], xs[-1])
        axL.set_ylim(*ylims)
        if row_i == 0:
            axL.set_xticks([2012.5, 2013.0, 2013.5, 2014.0, 2014.5, 2015.0])
            axL.set_xticklabels(["2012.5", "2013", "2013.5", "2014", "2014.5", "2015"])
            axL.set_yticks([-40, -20, 0, 20, 40])
        elif row_i == 1:
            axL.set_xticks([2017.0, 2017.5, 2018.0])
            axL.set_xticklabels(["2017", "2017.5", "2018"])
            axL.set_yticks([0, 2, 4])
        else:
            axL.set_xticks([650, 800, 950, 1100, 1250])
            axL.set_yticks([1000, 1200, 1400])
        axL.text(0.98, 0.95, rf"$\bar x={xbar:.2f}$, {mu_str}", transform=axL.transAxes,
                 ha="right", va="top", fontsize=F.SIZES[2], color=F.COL["INK"], zorder=6,
                 bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.92))
        F.panel_tag(axL, "ace"[row_i])

        # Right panel: 95% confidence intervals (naive vs CosCert)
        axR = fig.add_subplot(gs[row_i, 1])
        axR.axvline(mu0, color=F.COL["REF"], lw=0.85, ls=(0, (4, 2)), zorder=1)
        axR.axvline(xbar, color="#D8D8D8", lw=0.6, zorder=0)
        y_pos = np.arange(len(test_order))[::-1]
        for yp, (tkey, tlab, col) in zip(y_pos, test_order):
            td = rec["tests"][tkey]
            if tkey == "ADG_MAC":
                lo, hi = td["ci95"]
                axR.plot([lo, hi], [yp, yp], color=col, lw=1.5, ls=(0, (2, 1.2)), zorder=3)
                axR.plot([xbar], [yp], marker="D", ms=3.0, color=col, zorder=4)
            else:
                lo_n, hi_n = td["ci95_naive"]
                lo_c, hi_c = td["ci95_coscert_lw"]
                # CosCert CI: bold coloured bar slightly shifted down
                axR.plot([lo_c, hi_c], [yp - 0.13, yp - 0.13], color=col, lw=2.0, solid_capstyle="round", zorder=3)
                axR.plot([xbar], [yp - 0.13], marker="o", ms=2.6, color=col, zorder=4)
                # Naive short-memory CI: thin grey bar slightly shifted up with vertical end caps
                axR.plot([lo_n, hi_n], [yp + 0.16, yp + 0.16], color="#888888", lw=1.0, zorder=2)
                axR.plot([lo_n, lo_n], [yp + 0.06, yp + 0.26], color="#888888", lw=0.8, zorder=2)
                axR.plot([hi_n, hi_n], [yp + 0.06, yp + 0.26], color="#888888", lw=0.8, zorder=2)
                if td["reversal_lw"]:
                    axR.plot([hi_c], [yp - 0.13], marker="|", ms=4.5, color=col, mew=1.0, zorder=4)
                    axR.plot([lo_c], [yp - 0.13], marker="|", ms=4.5, color=col, mew=1.0, zorder=4)
        axR.set_yticks(y_pos)
        axR.set_yticklabels([t[1] for t in test_order], fontsize=F.SIZES[2])
        axR.set_ylim(-0.65, len(test_order) - 0.35)
        axR.tick_params(axis="y", length=0)
        axR.spines["left"].set_visible(False)
        axR.set_title(r"Nominal $95\,\%$ intervals for $\mu$", fontsize=F.SIZES[1])
        if row_i == 2:
            axR.set_xlabel(r"mean $\mu$ (series units)", fontsize=F.SIZES[1])
        xmin, xmax = axR.get_xlim()
        axR.set_xticks([tick for tick in axR.get_xticks() if xmin <= tick <= xmax])
        F.panel_tag(axR, "bdf"[row_i])

    handles = [
        Line2D([], [], color="#888888", lw=1.0, label=r"naive $H{=}\frac12$ $95\,\%$ CI"),
        Line2D([], [], color=F.COL["SN"], lw=2.0, marker="o", ms=3.0, label=r"CosCert ($\hat H_{\mathrm{LW}}$) $95\,\%$ CI"),
        Line2D([], [], color=F.COL["MP"], lw=1.5, ls=(0, (2, 1.2)), marker="D", ms=3.0, label=r"ADG--MAC $95\,\%$ CI"),
        Line2D([], [], color=F.COL["REF"], lw=0.85, ls=(0, (4, 2)), label=r"null $\mu_0$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=4,
               frameon=False, fontsize=F.SIZES[1], handlelength=1.7, columnspacing=1.1, handletextpad=0.4)
    return fig, inputs


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig7_empirical", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
