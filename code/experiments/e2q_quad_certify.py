"""E2q: certification of every tabulated Bartlett critical value by adaptive complex ball quadrature (Theorem 10).

For each H of the E2 grid and each level p, the certified distribution function of T_H is evaluated with
BridgeSystem.cdf_T_certified_quad (J = 16384, t = 25, tilt 1/4) at both ends of the tabulated bracket [q_lo, q_hi].
The bracket is certified when the upper bound at q_lo lies below p and the lower bound at q_hi lies above p.
The tilted midpoint route of e2r_revalidate.py (results/E2r/) is the independent confirmation of the same brackets.

Resumable: one JSON per H under results/E2q/; results/E2q_summary.json is written once all H are present.
Usage: python e2q_quad_certify.py [H1,H2,...]   (parallel workers take disjoint H subsets)
"""
import gc, glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import certify as cf  # noqa: E402

RES = os.path.join(WORK, "results"); OUT = os.path.join(RES, "E2q"); os.makedirs(OUT, exist_ok=True)
E2 = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
P_ALL = ["0.800", "0.900", "0.950", "0.975", "0.980", "0.990"]
HSEL = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None
J, T_TAIL, TILT = 16384, 25.0, 0.25


def ev(sysm, c):
    t0 = time.perf_counter()
    P = sysm.cdf_T_certified_quad(float(c), t=T_TAIL, theta=TILT)
    return {"c": float(c), "lo": float(P["lo"]), "hi": float(P["hi"]), "width": float(P["width"]), "s_max": P.get("s_max"),
            "n_panels": P.get("n_panels"), "evals": P.get("evals"), "seconds": time.perf_counter() - t0}


Hs = sorted(E2, key=float)
first = [h for h in ("0.50", "0.75", "0.95", "0.99") if h in E2]
for h in first + [h for h in Hs if h not in first]:
    fo = os.path.join(OUT, f"H{h}.json")
    if os.path.exists(fo) or (HSEL is not None and h not in HSEL):
        continue
    t0 = time.perf_counter(); sysm = cf.BridgeSystem(h, J); tb = time.perf_counter() - t0
    rec = {"H": h, "J": J, "t": T_TAIL, "tilt": TILT, "route": "cdf_T_certified_quad", "build_s": tb, "sn": {}}
    for p in P_ALL:
        c = E2[h][p]; pp = float(p)
        A = ev(sysm, c["q_lo"]); B = ev(sysm, c["q_hi"])
        rec["sn"][p] = {"q_lo": c["q_lo"], "q_hi": c["q_hi"], "F_at_q_lo": [A["lo"], A["hi"]], "F_at_q_hi": [B["lo"], B["hi"]],
                        "bracket_ok": bool(A["hi"] < pp < B["lo"]), "width": max(A["width"], B["width"]),
                        "evals": [A["evals"], B["evals"]], "seconds": A["seconds"] + B["seconds"]}
    rec["total_s"] = time.perf_counter() - t0
    json.dump(rec, open(fo + ".tmp", "w"), indent=1); os.replace(fo + ".tmp", fo)
    ok = all(v["bracket_ok"] for v in rec["sn"].values())
    print(f"H={h}: build {tb:.1f}s total {rec['total_s']:.1f}s brackets_ok={ok}", flush=True)
    del sysm; gc.collect()

done = sorted(glob.glob(os.path.join(OUT, "H*.json")))
if len(done) == len(Hs):
    recs = [json.load(open(f)) for f in done]
    cells = [(r["H"], p, v) for r in recs for p, v in r["sn"].items()]
    summ = {"route": "cdf_T_certified_quad", "J": J, "t": T_TAIL, "tilt": TILT, "n_H": len(recs), "n_brackets": len(cells),
            "n_certified": sum(v["bracket_ok"] for _, _, v in cells), "max_width": max(v["width"] for _, _, v in cells),
            "total_seconds": sum(r["total_s"] for r in recs), "build_seconds": sum(r["build_s"] for r in recs),
            "failed": [[h, p] for h, p, v in cells if not v["bracket_ok"]]}
    json.dump(summ, open(os.path.join(RES, "E2q_summary.json"), "w"), indent=1)
    print("summary:", {k: summ[k] for k in ("n_brackets", "n_certified", "max_width", "total_seconds")}, flush=True)
print("done", flush=True)
