"""Figure 1 (headline).  One message: under long memory the correct critical values of every self-normalised test of the
mean grow steeply with H, the short-memory (H = 1/2) values over-reject badly, and a closed-form first-moment rescaling
repairs the fixed-smoothing tests (HI, EWC) but not the Bartlett b = 1 / SN test.

(a) c_0.05(H) / c_0.05(1/2) on the t-scale (normalisation-free): SN sqrt(q_H / q_1/2); HI, EWC two-sided t-type values.
(b) exact asymptotic size of the nominal 5 % test that uses the H = 1/2 value (E6; SN certified at 5 %, HI/EWC certified).
(c) exact asymptotic size of the rescaled rule c_1/2 / sqrt(rho(H)) (E6b).
HI m in {1, 4, 16}: lines; EWC B = 2m: open markers on the matching HI line (HI(m) and EWC(2m) nearly coincide).
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = os.path.join(F.WORK, "results")


def merged(pattern):
    out, files = {}, sorted(glob.glob(os.path.join(RES, pattern)))
    for f in files:
        out.update(json.load(open(f)))
    return out, files


def main():
    sn_path = os.path.join(RES, "E2_sn_critical_values.json")
    sn = json.load(open(sn_path))["cells"]
    fx, f_fx = merged("E2b_fixed_*.json"); fx = {h: v["cells"] for h, v in fx.items()}
    nsn, f_nsn = merged("E6_naive_sn_*.json"); nfx, f_nfx = merged("E6_naive_fixed_*.json")
    rfx, f_rfx = merged("E6b_rescaled_fixed_*.json"); rsn, f_rsn = merged("E6b_rescaled_sn_*.json")

    H_sn = np.array(sorted(float(h) for h in sn)); H_fx = np.array(sorted(float(h) for h in fx))
    key = lambda h: f"{h:.2f}"
    HI_SHOW, EWC_SHOW = (1, 4, 16), (2, 8, 32)
    hi_col = dict(zip(HI_SHOW, [F.tint(F.COL["HI"], 0.42), F.COL["HI"], F.shade(F.COL["HI"], 0.5)]))
    ewc_col = dict(zip(EWC_SHOW, [F.tint(F.COL["EWC"], 0.50), F.COL["EWC"], F.shade(F.COL["EWC"], 0.45)]))
    sn_col = F.COL["SN"]
    mk_H = np.round(np.arange(0.55, 0.991, 0.05), 2)          # EWC marker positions

    fig = plt.figure(figsize=(F.TEXTWIDTH, 2.62))
    gs = fig.add_gridspec(1, 3, left=0.072, right=0.985, bottom=0.165, top=0.80, wspace=0.42)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # (a) inflation of the correct critical value -------------------------------------------------------------------
    a = ax[0]
    q05 = sn[key(0.5)]["0.950"]["q"]
    a.plot(H_sn, np.sqrt([sn[key(h)]["0.950"]["q"] / q05 for h in H_sn]), color=sn_col, lw=1.9, zorder=5)
    for m in HI_SHOW:
        x0 = fx[key(0.5)][f"HI|{m}|0.05"]["x"]
        a.plot(H_fx, [fx[key(h)][f"HI|{m}|0.05"]["x"] / x0 for h in H_fx], color=hi_col[m], lw=1.2)
    for B in EWC_SHOW:
        x0 = fx[key(0.5)][f"EWC|{B}|0.05"]["x"]
        a.plot(mk_H, [fx[key(h)][f"EWC|{B}|0.05"]["x"] / x0 for h in mk_H], ls="none", marker="o", ms=3.6, mfc="white",
               mec=ewc_col[B], mew=0.9, zorder=6)
    a.set_yscale("log"); F.log_ticks(a, "y", [1, 2, 5, 10, 20])
    a.set_ylim(0.93, 16)
    a.set_ylabel(r"$c_{0.05}(H)\,/\,c_{0.05}(1/2)$, $t$-scale")
    a.set_title("Correct values grow with $H$")

    # (b) size with the short-memory critical value ------------------------------------------------------------------
    b = ax[1]
    Hn = np.array(sorted(float(h) for h in nsn))
    b.plot(Hn, [100 * nsn[key(h)]["0.05"]["size_computed"] for h in Hn], color=sn_col, lw=1.9, zorder=5)
    Hf = np.array(sorted(float(h) for h in nfx))
    for m in HI_SHOW:
        b.plot(Hf, [100 * nfx[key(h)][f"HI|{m}|0.05"]["size_computed"] for h in Hf], color=hi_col[m], lw=1.2)
    for B in EWC_SHOW:
        b.plot(mk_H, [100 * nfx[key(h)][f"EWC|{B}|0.05"]["size_computed"] for h in mk_H], ls="none", marker="o", ms=3.6,
               mfc="white", mec=ewc_col[B], mew=0.9, zorder=6)
    b.axhline(5, color=F.COL["REF"], lw=0.7, ls=(0, (4, 2)), zorder=1)
    b.set_yscale("log"); F.log_ticks(b, "y", [5, 10, 20, 50, 100]); b.set_ylim(4.2, 100)   # log: factors of over-rejection (referee, 16 Nov)
    b.set_ylabel(r"actual size (\%), nominal 5\,\%")
    b.set_title(r"The $H=\frac12$ value over-rejects")
    h9 = 0.90
    if key(h9) in nsn:
        v = 100 * nsn[key(h9)]["0.05"]["size_computed"]
        b.plot([h9], [v], ls="none", marker="o", ms=3.4, color=sn_col, zorder=7)
        b.text(0.515, 88, rf"SN at $H={h9:.1f}$: {v:.1f}\,\%", color=sn_col, fontsize=F.SIZES[1], ha="left", va="center")

    # (c) size of the closed-form rescaled rule ------------------------------------------------------------------------
    c = ax[2]
    Hr = np.array(sorted(float(h) for h in rfx))
    for m in HI_SHOW:
        c.plot(Hr, [100 * 0.5 * (rfx[key(h)][f"HI|{m}|0.05"]["size_lo"] + rfx[key(h)][f"HI|{m}|0.05"]["size_hi"]) for h in Hr],
               color=hi_col[m], lw=1.2)
    for B in EWC_SHOW:
        c.plot(mk_H, [100 * 0.5 * (rfx[key(h)][f"EWC|{B}|0.05"]["size_lo"] + rfx[key(h)][f"EWC|{B}|0.05"]["size_hi"]) for h in mk_H],
               ls="none", marker="o", ms=3.6, mfc="white", mec=ewc_col[B], mew=0.9, zorder=6)
    if rsn:
        Hs = np.array(sorted(float(h) for h in rsn))
        c.plot(Hs, [100 * rsn[key(h)]["0.05"]["size_computed"] for h in Hs], color=sn_col, lw=1.9, marker="s", ms=2.6, zorder=5)
    c.axhline(5, color=F.COL["REF"], lw=0.7, ls=(0, (4, 2)), zorder=1)
    c.set_ylabel(r"actual size (\%), nominal 5\,\%")
    dev_fx = max(abs(100 * 0.5 * (rfx[key(h)][f"{kd}|{k}|0.05"]["size_lo"] + rfx[key(h)][f"{kd}|{k}|0.05"]["size_hi"]) - 5)
                 for h in Hr for kd, ks in (("HI", HI_SHOW), ("EWC", EWC_SHOW)) for k in ks)
    dev_sn = max(abs(100 * rsn[h]["0.05"]["size_computed"] - 5) for h in rsn) if rsn else 0.0
    claim = dev_fx <= 1.2 and dev_sn >= 3.0
    print(f"fig1(c): max |dev| HI/EWC = {dev_fx:.2f} pp, SN = {dev_sn:.2f} pp -> claim {'holds' if claim else 'FAILS'}")
    c.set_title(r"Rescaling fixes HI, EWC" if claim else r"Size after rescaling by $\rho(H)$")
    c.text(0.80, 8.55, "SN", color=sn_col, fontsize=F.SIZES[1], ha="right", va="bottom")
    c.text(0.985, 6.25, "HI, EWC", color=F.COL["HI"], fontsize=F.SIZES[1], ha="right", va="bottom")

    for i, axx in enumerate(ax):
        axx.set_xlim(0.49, 1.0); axx.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        axx.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        axx.set_xlabel(r"Hurst index $H$")
        F.panel_tag(axx, "abc"[i])

    handles = [Line2D([], [], color=sn_col, lw=1.9, label=r"SN (Bartlett, $b{=}1$)")]
    handles += [Line2D([], [], color=hi_col[m], lw=1.2, marker="o", ms=3.4, mfc="white", mec=ewc_col[B], mew=0.9,
                       label=rf"HI $m{{=}}{m}$, EWC $B{{=}}{B}$") for m, B in zip(HI_SHOW, EWC_SHOW)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=4, frameon=False, handlelength=1.7,
               columnspacing=1.2, handletextpad=0.45, fontsize=F.SIZES[1])
    return fig, [sn_path] + f_fx + f_nsn + f_nfx + f_rfx + f_rsn


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig1_headline", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
