"""Figure 4: audit of published critical values in test-size units (E4).  One message: published tables are accurate
only to their own Monte Carlo precision (a few tenths of a point in size), except where the simulated design is far from
the limit (Hualde-Iacone, delta <= -0.4: correct for the T = 10,000 ARFIMA design, not for the limit) or the fit fails
(McElroy-Politis Bartlett b = 1: systematic excess for H >= 0.6; row beta = -0.4 inconsistent).
Version B (16 Nov): 2 x 2 layout; (c) is new (limit vs design law at delta <= -0.4).

(a) H = 1/2 sources: size deviation (pp) of each published value; grey band +-2 MC s.e. of the source's own design.
(b) Hualde-Iacone Table 1 (176 values, delta = H - 1/2 in [-0.49, 0.49]): deviation against H, colour = m; band +-2 s.e.
(c) McElroy-Politis (2009) Bartlett b = 1 fit: deviation against H for each level, bars +-2 s.e.; open markers: beta < 0,
    compared with the exact law of their N = 1000 discretised design (limit not certified for H < 1/2).
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = os.path.join(F.WORK, "results")


def main():
    files = {p: os.path.join(RES, f"E4_audit_{p}.csv") for p in ("half", "hi", "hi_neg", "mp", "mp_neg")}
    A = {p: pd.read_csv(f) for p, f in files.items()}
    fig = plt.figure(figsize=(F.TEXTWIDTH, 4.6))
    gs = fig.add_gridspec(2, 2, left=0.175, right=0.985, bottom=0.085, top=0.95, wspace=0.30, hspace=0.50, width_ratios=[1.0, 1.25])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # (a) H = 1/2 -----------------------------------------------------------------------------------------------------
    a = ax[0]; h = A["half"][A["half"].source != "KPSS1990wp"].copy()   # the 1990 working paper prints the article's values
    def grp(r):
        lid = str(r.ledger_id).lower()
        if r.source == "Lobato2001":
            return r"Lobato"
        if r.source == "AP1997":
            return r"Abadir--Paruolo"
        if r.source == "KV2002wp":
            return r"Kiefer--Vogelsang"
        return r"KPSS, $\eta_\tau$" if "tau" in lid else r"KPSS, $\eta_\mu$"
    h["grp"] = h.apply(grp, axis=1)
    order = [r"Lobato", r"Abadir--Paruolo", r"Kiefer--Vogelsang", r"KPSS, $\eta_\mu$", r"KPSS, $\eta_\tau$"]
    col = {g: (F.COL["H05"] if "KPSS" not in g else F.COL["KPSS"]) for g in order}
    for i, g in enumerate(order[::-1]):
        s = h[h.grp == g]
        if not len(s):
            continue
        se2 = 3 * float(s.mc_se_pp.max())
        a.fill_betweenx([i - 0.32, i + 0.32], -se2, se2, color="#E6E6E6", lw=0, zorder=0)
        jit = np.linspace(-0.18, 0.18, len(s)) if len(s) > 1 else [0.0]
        a.scatter(s.dev_pp, i + np.asarray(jit), s=12, color=col[g], edgecolor="white", linewidth=0.4, zorder=3)
    a.axvline(0, color=F.COL["REF"], lw=0.6, zorder=1)
    a.set_yticks(range(len(order))); a.set_yticklabels(order[::-1], fontsize=F.SIZES[1]); a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    out_a = int((h.dev_pp.abs() > 3 * h.mc_se_pp).sum())
    a.set_xlabel("size deviation (pp)"); a.set_title(rf"$H=\frac12$: {out_a} of {len(h)} outside $\pm3$ s.e.")
    xm = max(0.3, 1.15 * float(h.dev_pp.abs().max())); a.set_xlim(-xm, xm)

    # (b) Hualde-Iacone ------------------------------------------------------------------------------------------------
    b = ax[1]; hi_all = pd.concat([A["hi_neg"], A["hi"]], ignore_index=True); hi = hi_all[hi_all.H >= 0.2 - 1e-9]
    ms = sorted(hi.m.unique()); cols = F.ramp(F.COL["HI"], len(ms))
    se2 = 3 * float(hi.mc_se_pp.max())
    b.axhspan(-se2, se2, color="#E6E6E6", lw=0, zorder=0); b.axhline(0, color=F.COL["REF"], lw=0.6, zorder=1)
    for m, c in zip(ms, cols):
        s = hi[hi.m == m]; off = (m - 8.5) * 0.0016
        b.scatter(s.H + off, s.dev_pp, s=7, color=c, edgecolor="none", zorder=3)
    b.axvline(0.5, color=F.COL["REF"], lw=0.5, ls=(0, (2, 2)), zorder=1)
    b.set_xlim(0.17, 1.0); b.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); b.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
    b.set_xlabel(r"Hurst index $H=\delta+\frac12$"); b.set_ylabel("size deviation (pp)")
    out_b = int((hi.dev_pp.abs() > 3 * hi.mc_se_pp).sum())
    b.set_ylim(-0.78, 0.78)
    b.set_title(rf"$\delta\ge-0.3$: {out_b} of {len(hi)} beyond $\pm3$ s.e.")
    b.text(0.02, 0.96, r"Hualde--Iacone (2017), Table 1", transform=b.transAxes, fontsize=F.SIZES[2], color=F.COL["INK"], ha="left", va="top")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cm = ListedColormap(cols); nm = BoundaryNorm(np.arange(0.5, len(ms) + 1.5), len(ms))
    cax = b.inset_axes([0.62, 0.90, 0.34, 0.045])
    cb = fig.colorbar(ScalarMappable(norm=nm, cmap=cm), cax=cax, orientation="horizontal", ticks=[1, 8, 16])
    cb.ax.tick_params(labelsize=F.SIZES[2], length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cax.set_title(r"$m$", fontsize=F.SIZES[2], pad=1.5, loc="center")

    # (c) Hualde-Iacone at delta <= -0.4: limit law vs the simulated finite design ----------------------------------------
    d3 = ax[2]; hn = A["hi_neg"]
    se2n = 3 * float(hn.mc_se_pp.max())
    d3.axhspan(-se2n, se2n, color="#E6E6E6", lw=0, zorder=0); d3.axhline(0, color=F.COL["REF"], lw=0.6, zorder=1)
    for Hv, colv, lab in ((0.01, F.shade(F.COL["HI"], 0.45), r"$\delta=-0.49$"), (0.10, F.tint(F.COL["HI"], 0.75), r"$\delta=-0.4$")):
        s = hn[np.isclose(hn.H, Hv)].sort_values("m")
        d3.plot(s.m, s.dev_pp, color=colv, lw=1.1, marker="o", ms=3.0, zorder=3)
        d3.plot(s.m, s.design_dev_pp, color=colv, lw=0.9, ls=(0, (3, 1.6)), marker="o", ms=3.0, mfc="white", mec=colv, mew=0.8, zorder=4)
        d3.text(16.4, float(s.dev_pp.iloc[-1]), lab, color=colv, fontsize=F.SIZES[2], ha="left", va="center")
    d3.set_xlim(0.5, 19.2); d3.set_xticks([1, 4, 8, 12, 16]); d3.set_xlabel(r"number of frequencies $m$"); d3.set_ylabel("size deviation (pp)")
    big = hn[hn.H <= 0.1 + 1e-9]; ok_design = bool((big.design_z.abs() <= 3).all()); bad_limit = bool((big.limit_z.abs() > 3).any())
    d3.set_title(r"$\delta\le-0.4$: fits inferred design, not limit" if (ok_design and bad_limit) else r"Hualde--Iacone, $\delta\le-0.4$")
    hd = [Line2D([], [], color=F.COL["INK"], lw=1.1, marker="o", ms=3.0, label="limit law"),
          Line2D([], [], color=F.COL["INK"], lw=0.9, ls=(0, (3, 1.6)), marker="o", ms=3.0, mfc="white", label=r"inferred design")]
    d3.legend(handles=hd, loc="upper left", fontsize=F.SIZES[2], frameon=False, handlelength=1.6, handletextpad=0.4,
              borderaxespad=0.25, labelspacing=0.25)

    # (d) McElroy-Politis -----------------------------------------------------------------------------------------------
    c = ax[3]; mp = A["mp"]; mn = A["mp_neg"]
    alphas = sorted(mp.alpha.unique(), reverse=True); acol = dict(zip(alphas, F.ramp(F.COL["MP"], len(alphas), lo=0.45)))
    offs = dict(zip(alphas, np.linspace(-0.015, 0.015, len(alphas))))
    c.axhline(0, color=F.COL["REF"], lw=0.6, zorder=1)
    for al in alphas:
        for D, filled in ((mp, True), (mn, False)):
            s = D[np.isclose(D.alpha, al)].sort_values("H")
            if not len(s):
                continue
            c.errorbar(s.H + offs[al], s.dev_pp, yerr=3 * s.mc_se_pp, fmt="o", ms=3.2, mfc=acol[al] if filled else "white", mec=acol[al],
                       ecolor=acol[al], elinewidth=0.6, capsize=0, mew=0.8, zorder=3)
    c.axvline(0.5, color=F.COL["REF"], lw=0.5, ls=(0, (2, 2)), zorder=1)
    t3 = mn[np.isclose(mn.beta, -0.4)]
    if len(t3) and float(t3.dev_pp.min()) < -1.0:
        c.text(0.335, float(t3.dev_pp.min()) + 0.45, "row\n" + r"$\beta=-0.4$", fontsize=F.SIZES[2], color=F.COL["INK"], ha="left", va="center", linespacing=1.1)
    c.set_xlabel(r"Hurst index $H=(\beta+1)/2$"); c.set_ylabel("size deviation (pp)")
    allpos = bool((mp[mp.H >= 0.6 - 1e-9].dev_pp > 0).all())
    c.set_title(r"McElroy--Politis fit: liberal for $H\ge0.6$" if allpos else r"McElroy--Politis Bartlett fit")
    hand = [Line2D([], [], ls="none", marker="o", ms=3.2, color=acol[al], label=rf"$\alpha={100 * al:g}\,\%$") for al in alphas]
    hand += [Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=F.COL["REF"], label=r"$\beta<0$ (design law)")]
    c.legend(handles=hand, loc="lower right", fontsize=F.SIZES[2], handletextpad=0.2, borderaxespad=0.3, labelspacing=0.2, frameon=False)
    for i, axx in enumerate(ax):
        F.panel_tag(axx, "abcd"[i])
    return fig, list(files.values())


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig4_audit", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
