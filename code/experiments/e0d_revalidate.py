"""E0d (17 Nov 2026): re-run part (d) of the H = 1/2 gate with the corrected certificate (Arb scalar assembly, Gershgorin
bounds; W route with outward evaluation points). Parts (a)-(c) of results/E0_anchors.json do not use certify.py and are
kept; the previous (d) is kept as d_16nov. Usage: python e0d_revalidate.py
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time  # noqa: E401,E402
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from flint import ctx  # noqa: E402
from p3 import certify as cf, laws as lw  # noqa: E402

ctx.prec = 128
f = os.path.join(WORK, "results", "E0_anchors.json")
out = json.load(open(f))
if "d_16nov" not in out:
    out["d_16nov"] = out["d"]
t0 = time.perf_counter()
s5 = cf.BridgeSystem("0.5", 4096)
dT = {}
for p, a in out["a"].items():
    q = a["q_closed_form"]; Fex = float(lw.cdf_T_half(q)); r = s5.cdf_T_certified_tilt(q)
    dT[p] = {"c": q, "F_exact": Fex, "cert": [r["lo"], r["hi"]], "contains": r["lo"] <= Fex <= r["hi"], "width": r["hi"] - r["lo"], "K": r["K"]}
    print(f"(d) T_1/2 p={p}: [{r['lo']:.10f}, {r['hi']:.10f}] contains {dT[p]['contains']}", flush=True)
dW = {}
for p in ("0.9", "0.95", "0.99"):
    q = out["c"][p]["cvm_quantile"]; Fex = float(lw.cdf_cvm_fast(q)); r = s5.cdf_Wfull(q, certified=True)
    dW[p] = {"x": q, "F_exact": Fex, "cert": [r["lo"], r["hi"]], "contains": r["lo"] <= Fex <= r["hi"]}
    print(f"(d) CvM p={p}: [{r['lo']:.10f}, {r['hi']:.10f}] contains {dW[p]['contains']}", flush=True)
out["d"] = {"T_half_J4096": dT, "cvm_J4096": dW, "fixed_smoothing": out["d_16nov"].get("fixed_smoothing"),
            "certificate": "corrected 17 Nov 2026", "seconds": time.perf_counter() - t0}
out["gate_G0_pass"] = bool(all(v["contains"] for v in dT.values()) and all(v["contains"] for v in dW.values())
                           and all((out["d"]["fixed_smoothing"] or {}).values()))
json.dump(out, open(f + ".tmp", "w"), indent=1, default=float); os.replace(f + ".tmp", f)
print("GATE G0 PASS:", out["gate_G0_pass"], f"{time.perf_counter() - t0:.0f}s", flush=True)
