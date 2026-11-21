"""Figure 6: why the certificate works.
(a) Normalised eigenvalues nu_k = lambda_k (k pi)^(1+2H) / c_H of the fBm bridge covariance operator (E8, Galerkin J = 2048,
    k <= J/4) inside the band [1/2, 1] of the two-sided bound (Theorem eig).
(b) Width of the certified enclosure of F_H at the 90 % quantile against J (E1), with the predicted slope -(1/2 + 2H).
The bridge spectrum is that of the Bartlett/self-normalised denominator, so the SN hue is used; lightness = H.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import p3fig as F  # noqa: E402

F.setup()
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(F.WORK, "results")


def main():
    sp_path = os.path.join(RES, "E8_spectrum.json"); sp = json.load(open(sp_path))
    e1f = sorted(glob.glob(os.path.join(RES, "E1_benchmark_H*.json"))); E1 = {float(json.load(open(f))["H"]): json.load(open(f)) for f in e1f}
    fig = plt.figure(figsize=(F.TEXTWIDTH, 2.6))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.985, bottom=0.17, top=0.86, wspace=0.28)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    Hs = sorted(float(h) for h in sp["H"]); cols = dict(zip(Hs, F.ramp(F.COL["SN"], len(Hs), lo=0.35)))
    a.axhspan(0.5, 1.0, color="#EDEDED", lw=0, zorder=0)
    a.axhline(1.0, color=F.COL["REF"], lw=0.7, zorder=1); a.axhline(0.5, color=F.COL["REF"], lw=0.7, zorder=1)
    kmax = sp["J"] // 4
    for H in Hs:
        nu = np.array(sp["H"][f"{H:.2f}"]["nu"])[:kmax]; k = np.arange(1, nu.size + 1)
        a.plot(k, nu, color=cols[H], lw=1.3, label=rf"$H={H:g}$")
    allnu = np.concatenate([np.array(sp["H"][f"{H:.2f}"]["nu"]) for H in Hs]); inband = bool(allnu.min() >= 0.5 - 1e-12 and allnu.max() <= 1 + 1e-12)
    print(f"fig6(a): nu range [{allnu.min():.4f}, {allnu.max():.6f}] over all k <= J -> in band: {inband}")
    a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=F.SIZES[2], frameon=False, handlelength=1.6, labelspacing=0.3)
    a.set_xscale("log"); a.set_xlim(1, kmax * 1.15); a.set_ylim(0.45, 1.05)
    a.set_xlabel(r"eigenvalue index $k$"); a.set_ylabel(r"$\lambda_k\,(k\pi)^{1+2H}/c^\star_H$")
    a.set_title(r"Every eigenvalue lies in the proven band" if inband else r"Normalised bridge eigenvalues")
    a.text(1.3, 0.53, "lower bound", fontsize=F.SIZES[2], color=F.COL["REF"], va="bottom")

    He = [h for h in sorted(E1) if abs(h - 0.51) > 1e-9]; ce = dict(zip(He, F.ramp(F.COL["SN"], len(He), lo=0.48))); ends = []
    for H in He:
        rows = sorted([r for r in E1[H]["rows"] if abs(r["p"] - 0.9) < 1e-9], key=lambda r: r["J"])
        J = np.array([r["J"] for r in rows], float); w = np.array([r["width_F"] for r in rows])
        b.plot(J, w, color=ce[H], lw=1.3, marker="o", ms=3.0)
        slope = -(0.5 + 2 * H); ref = 0.35 * w[0] * (J / J[0]) ** slope          # reference rate, offset below the data
        b.plot(J, ref, color=ce[H], lw=0.8, ls=(0, (3, 2)))
        ends.append((J[-1], w[-1], rf"$H={H:g}$", ce[H]))
    b.set_xscale("log", base=2); b.set_yscale("log")
    b.set_xlim(400, 16384 * 3.1)
    F.direct_labels(b, ends[0][0] * 1.15, [e[1] for e in ends], [e[2] for e in ends], [e[3] for e in ends])
    b.set_xlim(400, 16384 * 3.1)
    b.set_xticks([512, 2048, 8192, 16384]); b.set_xticklabels(["512", "2048", "8192", "16384"])
    b.set_xlabel(r"truncation $J$"); b.set_ylabel(r"width of the enclosure of $F_H(q_{0.9})$")
    b.set_title(r"Width falls like $J^{-1/2-2H}$ (dashed)")
    F.panel_tag(a, "a"); F.panel_tag(b, "b")
    return fig, [sp_path] + e1f


if __name__ == "__main__":
    fig, inputs = main()
    F.layout_check(fig)
    print(F.save(fig, "fig6_certificate", inputs=inputs, script=os.path.relpath(__file__, F.WORK)))
