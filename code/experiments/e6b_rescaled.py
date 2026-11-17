"""E6b: exact asymptotic size of the closed-form 'first-moment' rescaled critical values.

rho_P(H) = E_H[D_P] / E_{1/2}[D_P] for the self-normaliser D_P of procedure P (Type I fBm, Var B_H(1) = 1):
    SN :  E int_0^1 (B_H(r) - r B_H(1))^2 dr = 1/((2H+1)(H+1)) - 1/6
    HI :  m^-1 sum_{j<=m} E[C_j^2 + S_j^2]      (cosine/sine Gram diagonals, Paper 1)
    EWC:  B^-1 sum_{j<=B} E[Lambda_j^2] = B^-1 sum_j G_jj
Rule: reject when |t| > c_{1/2} / sqrt(rho_P(H))  (t-scale; for SN T_H > q_{1/2} / rho_SN(H)), c_{1/2} = t_{2m}, t_B, q_p(1/2).
HI / EWC: certified enclosures of the size (cdf2_certified).  SN: computed (tilted Bromwich, J = 4096), certified at 5 %.
usage: python e6b_rescaled.py fixed <H...> | sn <H...>
"""
import os, sys, json, math, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import certify as cf, lowrank as lr, gram as gr  # noqa: E402

Q_HALF = {"0.10": 28.3309213235089, "0.05": 45.5260857524482, "0.01": 100.3456394280872}
HI_M = list(range(1, 17)); EWC_B = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def exp_den(H, kind, k):
    if kind == "SN":
        return 1.0 / ((2 * H + 1) * (H + 1)) - 1.0 / 6.0
    Sg = gr.hi_fixed_m_cov(H, k) if kind == "HI" else gr.ewc_cov(H, k)
    return float(np.trace(np.asarray(Sg)[1:, 1:])) / k


def part_fixed(Hs):
    out = {}; path = os.path.join(WORK, "results", f"E6b_rescaled_fixed_{Hs[0]}_{Hs[-1]}.json")
    for h in Hs:
        H = float(h); t0 = time.perf_counter(); ga = lr.GramArb(h, 33); rec = {}
        for kind, ks in (("HI", HI_M), ("EWC", EWC_B)):
            for k in ks:
                rho = exp_den(H, kind, k) / exp_den(0.5, kind, k)
                law = lr.FixedSmoothingLaw(h, kind, k, ga=ga); dof = 2 * k if kind == "HI" else k
                for a in (0.10, 0.05, 0.01):
                    t = float(stats.t.ppf(1 - a / 2, dof)); c2 = t * t / rho
                    lo, hi = law.cdf2_certified(c2)
                    rec[f"{kind}|{k}|{a:.2f}"] = {"rho": rho, "cv": math.sqrt(c2), "size_lo": 1 - float(hi), "size_hi": 1 - float(lo)}
        out[h] = rec
        print(f"fixed H={h}: HI m=4 5% [{rec['HI|4|0.05']['size_lo']:.6f}, {rec['HI|4|0.05']['size_hi']:.6f}]  {time.perf_counter() - t0:.0f}s", flush=True)
        json.dump(out, open(path, "w"), indent=1)


def part_sn(Hs, J=4096):
    out = {}; path = os.path.join(WORK, "results", f"E6b_rescaled_sn_{Hs[0]}_{Hs[-1]}.json")
    for h in Hs:
        H = float(h); t0 = time.perf_counter(); rho = exp_den(H, "SN", 0) / exp_den(0.5, "SN", 0)
        sysm = cf.BridgeSystem(h, J); rec = {"rho": rho, "J": J}
        for a, q in Q_HALF.items():
            c = q / rho; r = sysm.cdf_T_computed_tilt(c); cell = {"cv": c, "size_computed": 1.0 - float(r["point"])}
            if a == "0.05":
                rc = sysm.cdf_T_certified_tilt(c); cell.update({"size_lo": 1.0 - float(rc["hi"]), "size_hi": 1.0 - float(rc["lo"])})
            rec[a] = cell
        out[h] = rec
        print(f"SN H={h}: rho={rho:.5f} sizes " + " ".join(f"{a}:{rec[a]['size_computed']:.5f}" for a in Q_HALF) + f"  {time.perf_counter() - t0:.0f}s", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    (part_fixed if sys.argv[1] == "fixed" else part_sn)(sys.argv[2:])
