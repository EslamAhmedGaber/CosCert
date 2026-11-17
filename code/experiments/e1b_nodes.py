"""E1b: number of inversion nodes needed by the untilted Gil-Pelaez midpoint sum and by the tilted Bromwich sum
(theta = 1/4) for the same alias and truncation tolerance (1e-14), at the certified quantiles of T_H, J = 4096.
Output: results/E1b_nodes.json
"""
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import certify as cf  # noqa: E402

EPS = 1e-14
tab = json.load(open(os.path.join(WORK, "results", "E2_sn_critical_values.json")))["cells"]
out = []
for h in ("0.50", "0.75", "0.95"):
    sysm = cf.BridgeSystem(h, 4096); rem = sysm.remainder(25.0)
    for p in ("0.900", "0.950", "0.990"):
        c = tab[h][p]["q"]; xs = [c * rem["s_lo"], c * rem["s_hi"]]
        t0 = time.perf_counter()
        L0 = sysm.choose_L(c, xs, EPS); K0 = sysm.choose_K(c, 2 * 3.141592653589793 / L0, EPS, kmax=4000000)
        L1 = sysm.choose_L_tilt(xs, 0.25, EPS); K1 = sysm.choose_K_tilt(c, 2 * 3.141592653589793 / L1, 0.25, min(xs), EPS)
        out.append({"H": h, "p": p, "c": c, "L_untilted": L0, "K_untilted": K0, "L_tilted": L1, "K_tilted": K1, "ratio": K0 / K1,
                    "seconds": time.perf_counter() - t0})
        print(out[-1], flush=True)
json.dump({"eps": EPS, "J": 4096, "rows": out}, open(os.path.join(WORK, "results", "E1b_nodes.json"), "w"), indent=1)
print("ratio range:", min(r["ratio"] for r in out), max(r["ratio"] for r in out))
