"""Figure 3: which test to use -- local power (E7) and the size-power trade-off (E7 x E3).

(a) limit power at H = 0.75 against the local alternative c (mu - mu0 = c sigma n^(H-1)), every test at its own correct
    5 % critical value; grey dashed = infeasible z-test that knows the scale sigma of the partial sums.
(b) power at c = 2.8 (where the infeasible z-test has 80 % power) as a function of H.
(c) trade-off at H = 0.7: limit power at c = 2.8 against the exact finite-sample size at n = 100 under ARFIMA(1,d,0) with
    phi = 0.5 (E3, own limit critical value).  Each extra smoothing ordinate buys power and costs finite-sample size.
"""
import os, sys, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = os.path.join(F.WORK, "results")
C_STAR = 2.8


def main():
    pf = sorted(glob.glob(os.path.join(RES, "E7_power_H*.csv"))); P = pd.concat([pd.read_csv(f) for f in pf], ignore_index=True)
    P["k"] = pd.to_numeric(P.param, errors="coerce").fillna(0).astype(int)
    ef = sorted(glob.glob(os.path.join(RES, "E3_rates_arfima1p_*.csv"))); E = pd.concat([pd.read_csv(f) for f in ef], ignore_index=True)
    E = E[(E.cv_label == "own_limit") & np.isclose(E.alpha, 0.05)].copy(); E["k"] = pd.to_numeric(E.param, errors="coerce").fillna(0).astype(int)

    HI_SHOW, EWC_SHOW = (1, 4, 16), (2, 8, 32)
    hi_col = dict(zip(HI_SHOW, [F.tint(F.COL["HI"], 0.42), F.COL["HI"], F.shade(F.COL["HI"], 0.5)]))
    ewc_col = dict(zip(EWC_SHOW, [F.tint(F.COL["EWC"], 0.50), F.COL["EWC"], F.shade(F.COL["EWC"], 0.45)]))
    fig = plt.figure(figsize=(F.TEXTWIDTH, 2.62))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.975, bottom=0.165, top=0.80, wspace=0.42)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # (a) power curves at H = 0.75
    a = ax[0]; S = P[np.isclose(P.H, 0.75)]
    z = S[S.procedure.str.startswith("oracle")].sort_values("c")
    a.plot(z.c, z.power, color=F.COL["REF"], lw=1.0, ls=(0, (4, 2)), zorder=2)
    s = S[S.procedure == "SN"].sort_values("c"); a.plot(s.c, s.power, color=F.COL["SN"], lw=1.9, zorder=5)
    for m in HI_SHOW:
        s = S[(S.procedure == "HI") & (S.k == m)].sort_values("c"); a.plot(s.c, s.power, color=hi_col[m], lw=1.2, zorder=3)
    for B in EWC_SHOW:
        s = S[(S.procedure == "EWC") & (S.k == B)].sort_values("c"); s = s[np.isclose((s.c * 5) % 5, 0)]   # markers at integer c
        a.plot(s.c, s.power, ls="none", marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.85, zorder=6)
    a.set_xlim(0, 8); a.set_ylim(0, 1.0); a.set_yticks([0.05, 0.25, 0.5, 0.75, 1.0])
    a.set_yticklabels(["0.05", "0.25", "0.5", "0.75", "1"])
    a.set_xlabel(r"local alternative $c$"); a.set_ylabel(r"power, $H = 0.75$")
    a.set_title("More ordinates, more power")
    a.plot([4.3, 5.2], [0.12, 0.12], color=F.COL["REF"], lw=1.0, ls=(0, (4, 2)))
    a.text(5.4, 0.12, r"$z$-test," + "\n" + r"known $\sigma$", fontsize=F.SIZES[2], color=F.COL["REF"], ha="left", va="center")

    # (b) power at c* across H
    b = ax[1]; Q = P[np.isclose(P.c, C_STAR)]
    s = Q[Q.procedure == "SN"].sort_values("H"); b.plot(s.H, s.power, color=F.COL["SN"], lw=1.9, marker="s", ms=2.8, zorder=5)
    for m in HI_SHOW:
        s = Q[(Q.procedure == "HI") & (Q.k == m)].sort_values("H"); b.plot(s.H, s.power, color=hi_col[m], lw=1.2, marker="s", ms=2.4)
    for B in EWC_SHOW:
        s = Q[(Q.procedure == "EWC") & (Q.k == B)].sort_values("H")
        b.plot(s.H, s.power, ls="none", marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.85, zorder=6)
    zz = Q[Q.procedure.str.startswith("oracle")].power.iloc[0]
    b.axhline(zz, color=F.COL["REF"], lw=1.0, ls=(0, (4, 2)))
    b.set_xlim(0.48, 0.92); b.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); b.set_ylim(0.3, 0.85)
    b.set_xlabel(r"Hurst index $H$"); b.set_ylabel(rf"power at $c = {C_STAR}$")
    sn_b = Q[Q.procedure == "SN"].sort_values("H").power.values; hi_b = Q[(Q.procedure == "HI") & (Q.k == 16)].sort_values("H").power.values
    drop_sn, drop_hi = sn_b[0] - sn_b[-1], hi_b[0] - hi_b[-1]
    print(f"fig3(b): power drop H .5->.9: SN {drop_sn:.3f}, HI16 {drop_hi:.3f}")
    b.set_title("Memory erodes SN's power" if drop_sn > 3 * max(drop_hi, 1e-9) else rf"Power at $c={C_STAR}$")

    # (c) trade-off at H = 0.7, n = 100, phi = 0.5
    c = ax[2]; Hh, n = 0.7, 100
    Qh = P[np.isclose(P.H, Hh) & np.isclose(P.c, C_STAR)]; Eh = E[np.isclose(E.H, Hh) & (E.n == n)]

    def pts(proc, ks):
        xs, ys, kk = [], [], []
        for k in ks:
            p = Qh[(Qh.procedure == proc) & (Qh.k == k)]; e = Eh[(Eh.procedure == proc) & (Eh.k == k)]
            if len(p) and len(e):
                xs.append(100 * float(e.rejection_rate.iloc[0])); ys.append(float(p.power.iloc[0])); kk.append(k)
        return np.array(xs), np.array(ys), kk
    xh, yh, kh = pts("HI", (1, 2, 4, 8, 16)); c.plot(xh, yh, color=F.COL["HI"], lw=1.0, marker="o", ms=3.4, zorder=4)
    for xx, yy, k in zip(xh, yh, kh):
        left = k <= 2
        c.annotate(rf"$m{{=}}{k}$", xy=(xx, yy), xytext=((-5, 0) if left else (5, -3)), textcoords="offset points", fontsize=F.SIZES[2],
                   color=F.COL["HI"], ha=("right" if left else "left"), va=("center" if left else "top"))
    xe, ye, ke = pts("EWC", (4, 8, 16, 32)); c.plot(xe, ye, color=F.COL["EWC"], lw=1.0, ls="none", marker="o", ms=3.6, mfc="white", mec=F.COL["EWC"], zorder=5)

    xs_, ys_, _ = pts("SN", (0,)); c.plot(xs_, ys_, ls="none", marker="s", ms=4.6, color=F.COL["SN"], zorder=6)
    c.annotate("SN", xy=(xs_[0], ys_[0]), xytext=(0, -6), textcoords="offset points", fontsize=F.SIZES[1], color=F.COL["SN"], va="top", ha="center")
    mono = bool(np.all(np.diff(xh) > 0) and np.all(np.diff(yh) > 0)); print(f"fig3(c): size and power increase with m: {mono}")
    c.text(0.97, 0.22, r"ARFIMA$(1,0.2,0)$" + "\n" + r"$\phi=0.5$, $H=0.7$", transform=c.transAxes, fontsize=F.SIZES[2], color=F.COL["INK"],
           ha="right", va="bottom", linespacing=1.3)
    c.axvline(5, color=F.COL["REF"], lw=0.7, ls=(0, (4, 2)))
    c.set_xlabel(r"exact size (\%) at $n=100$"); c.set_ylabel(rf"limit power at $c={C_STAR}$")
    c.set_title(r"Size and power at $n=100$")
    c.margins(x=0.12, y=0.12); x0_, x1_ = c.get_xlim(); c.set_xlim(x0_ - 0.25, x1_ + 0.9)
    for i, axx in enumerate(ax):
        F.panel_tag(axx, "abc"[i])
    handles = [Line2D([], [], color=F.COL["SN"], lw=1.9, label=r"SN (Bartlett, $b{=}1$)")]
    handles += [Line2D([], [], color=hi_col[m], lw=1.2, marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.85,
                       label=rf"HI $m{{=}}{m}$, EWC $B{{=}}{B}$") for m, B in zip(HI_SHOW, EWC_SHOW)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=4, frameon=False, handlelength=1.7,
               columnspacing=1.2, handletextpad=0.45, fontsize=F.SIZES[1])
    return fig, pf + ef


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig3_power", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
