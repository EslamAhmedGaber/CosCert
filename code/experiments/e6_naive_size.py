"""E6: asymptotic size of nominal tests that use the short-memory (H = 1/2) critical value when H > 1/2.

SN / Bartlett b = 1 (T_H):  size = P(T_H > q_p(1/2)),  q_p(1/2) the exact H = 1/2 quantiles (E0, closed form).
                            Computed (tilted Bromwich, J = 4096) at alpha = .10/.05/.01; certified bracket at alpha = .05.
HI fixed-m:                 size = P(|tau| > t_{2m, 1 - alpha/2})   (HI2017 eq. 2 statistic, Student-t value valid at H = 1/2).
EWC cosine series (LLSW):   size = P(|t| > t_{B, 1 - alpha/2}).
Fixed-smoothing laws are exact finite-dimensional ratio laws (Paper 1 Gram); all cells certified.

usage:  python e6_naive_size.py sn <J> <H1> <H2> ...     |     python e6_naive_size.py fixed <H1> <H2> ...
"""
import os, sys, json, math, time
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import certify as cf, lowrank as lr  # noqa: E402

Q_HALF = {"0.10": 28.3309213235089, "0.05": 45.5260857524482, "0.01": 100.3456394280872}   # E0 closed-form T_1/2 quantiles
HI_M = list(range(1, 17)); EWC_B = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def sn_part(J, Hs):
    out = {}
    path = os.path.join(WORK, "results", f"E6_naive_sn_{Hs[0]}_{Hs[-1]}.json")
    for h in Hs:
        t0 = time.perf_counter(); sysm = cf.BridgeSystem(h, J); tb = time.perf_counter() - t0
        rec = {"J": J, "build_s": tb}
        for a, c in Q_HALF.items():
            r = sysm.cdf_T_computed_tilt(c)
            cell = {"cv_half": c, "size_computed": 1.0 - float(r["point"])}
            if a == "0.05":
                t1 = time.perf_counter(); rc = sysm.cdf_T_certified_tilt(c)
                lo, hi = (float(rc["lo"]), float(rc["hi"])) if "lo" in rc else (float(rc["F_lo"]), float(rc["F_hi"]))
                cell.update({"size_lo": 1.0 - hi, "size_hi": 1.0 - lo, "certified": True, "cert_s": time.perf_counter() - t1})
            rec[a] = cell
        out[h] = rec
        print(f"SN H={h}: " + "  ".join(f"a={a}: {rec[a]['size_computed']:.6f}" for a in Q_HALF)
              + f"  [5% cert {rec['0.05'].get('size_lo', float('nan')):.8f}, {rec['0.05'].get('size_hi', float('nan')):.8f}]"
              + f"  {time.perf_counter() - t0:.0f}s", flush=True)
        json.dump(out, open(path, "w"), indent=1)


def fixed_part(Hs):
    out = {}
    path = os.path.join(WORK, "results", f"E6_naive_fixed_{Hs[0]}_{Hs[-1]}.json")
    for h in Hs:
        t0 = time.perf_counter(); ga = lr.GramArb(h, 33); rec = {}
        for kind, ks in (("HI", HI_M), ("EWC", EWC_B)):
            for k in ks:
                law = lr.FixedSmoothingLaw(h, kind, k, ga=ga); dof = 2 * k if kind == "HI" else k
                for a in (0.10, 0.05, 0.01):
                    cv = float(stats.t.ppf(1 - a / 2, dof)); c2 = cv * cv
                    lo, hi = law.cdf2_certified(c2)
                    rec[f"{kind}|{k}|{a:.2f}"] = {"cv_half": cv, "dof": dof, "size_computed": 1.0 - float(law.cdf2(c2)),
                                                  "size_lo": 1.0 - float(hi), "size_hi": 1.0 - float(lo)}
        out[h] = rec
        print(f"fixed H={h}: HI m=4 5% {rec['HI|4|0.05']['size_computed']:.5f}  EWC B=16 5% {rec['EWC|16|0.05']['size_computed']:.5f}"
              f"  {time.perf_counter() - t0:.0f}s", flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "sn":
        sn_part(int(sys.argv[2]), sys.argv[3:])
    else:
        fixed_part(sys.argv[2:])
