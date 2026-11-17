"""E2r: revalidation of every certified SN value with the corrected certificate (17 Nov 2026; audit item A1, D1).
Corrected: Gershgorin-cluster eigenvalue bounds, ball-arithmetic alias bound with geometric tail, ball truncation bound,
exact lattice period 2 pi / Delta, outward evaluation points, e^{-t} rounded up.  Re-evaluates
  (a) E6 naive SN sizes (J = 4096, alpha = .05) and E6b SN rescaled sizes (J = 4096, alpha = .05);
  (b) every E2 bracket endpoint (J = 16384): F at q_lo and q_hi, bracket condition F_hi(q_lo) < p < F_lo(q_hi);
  (c) E9b: the printed response surface (decimal coefficients, exact H, ball arithmetic) at the odd hundredths;
  (d) the T4 reference of E5 (H = 0.75, q_0.95, J = 16384).
Resumable: one JSON per H under results/E2r/ and results/E2r/J4096/.
Usage: python e2r_revalidate.py [a|b|all] [H1,H2,...]   (parallel workers take disjoint H subsets)
"""
import gc, json, math, os, sys, time, ctypes
from fractions import Fraction
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from flint import arb, ctx  # noqa: E402
from p3 import certify as cf  # noqa: E402
RES = os.path.join(WORK, "results"); OUT = os.path.join(RES, "E2r"); os.makedirs(OUT, exist_ok=True)

def rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    except Exception:
        return -1.0

def ev(sysm, c):
    t0 = time.perf_counter(); P = sysm.cdf_T_certified_tilt(float(c))
    return {"c": float(c), "lo": float(P["lo"]), "hi": float(P["hi"]), "K": P["K"], "L": P["L"], "seconds": time.perf_counter() - t0,
            "budget": P["budget"]}

ctx.prec = 128
E2 = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
SF = json.load(open(os.path.join(RES, "E9_surface.json")))
P_ALL = ["0.800", "0.900", "0.950", "0.975", "0.980", "0.990"]

# (a) J = 4096 parts: one file per H under results/E2r/J4096/, merged into J4096_E6_E6b.json when complete ----------------
import glob
PART = sys.argv[1] if len(sys.argv) > 1 else "all"                 # "a", "b" or "all"
HSEL = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None    # optional subset of H (strings such as 0.75)
f4 = os.path.join(OUT, "J4096_E6_E6b.json"); D4 = os.path.join(OUT, "J4096"); os.makedirs(D4, exist_ok=True)
e6 = {}
for f in sorted(glob.glob(os.path.join(RES, "E6_naive_sn_*.json"))):
    for h, rec in json.load(open(f)).items():
        if isinstance(rec, dict) and "0.05" in rec and rec["0.05"].get("certified"):
            e6[h] = rec["0.05"]
e6b = {}
for f in sorted(glob.glob(os.path.join(RES, "E6b_rescaled_sn_*.json"))):
    for h, rec in json.load(open(f)).items():
        if isinstance(rec, dict) and "0.05" in rec and "size_lo" in rec["0.05"]:
            e6b[h] = rec["0.05"]
H4 = sorted(set(e6) | set(e6b), key=float)
if PART in ("a", "all") and not os.path.exists(f4):
    for h in H4:
        fo4 = os.path.join(D4, f"H{h}.json")
        if os.path.exists(fo4) or (HSEL is not None and h not in HSEL):
            continue
        sysm = cf.BridgeSystem(h, 4096); rec4 = {}
        if h in e6:
            r = ev(sysm, e6[h]["cv_half"]); r.update(size_lo=1 - r["hi"], size_hi=1 - r["lo"], old_size_lo=e6[h]["size_lo"], old_size_hi=e6[h]["size_hi"])
            rec4["E6"] = r
        if h in e6b:
            r = ev(sysm, e6b[h]["cv"]); r.update(size_lo=1 - r["hi"], size_hi=1 - r["lo"], old_size_lo=e6b[h]["size_lo"], old_size_hi=e6b[h]["size_hi"])
            rec4["E6b"] = r
        json.dump(rec4, open(fo4 + ".tmp", "w"), indent=1); os.replace(fo4 + ".tmp", fo4)
        del sysm; gc.collect()
        print(f"J4096 H={h} done", flush=True)
    if all(os.path.exists(os.path.join(D4, f"H{h}.json")) for h in H4):
        out4 = {"E6": {}, "E6b": {}}
        for h in H4:
            rec4 = json.load(open(os.path.join(D4, f"H{h}.json")))
            for k in ("E6", "E6b"):
                if k in rec4:
                    out4[k][h] = rec4[k]
        json.dump(out4, open(f4, "w"), indent=1)
        print("J4096 merged", flush=True)

# (b)-(d) J = 16384 per H -------------------------------------------------------------------------------------------------
Hs = sorted(E2, key=float)
first = [h for h in ("0.50", "0.75", "0.95", "0.99") if h in E2]
order = first + [h for h in Hs if h not in first]
e1 = json.load(open(os.path.join(RES, "E1_benchmark_H0.75.json")))
t4_row = [r for r in e1["rows"] if r["J"] == 16384 and abs(r["p"] - 0.95) < 1e-12][0]
for h in (order if PART in ("b", "all") else []):
    fo = os.path.join(OUT, f"H{h}.json")
    if os.path.exists(fo) or (HSEL is not None and h not in HSEL):
        continue
    t0 = time.perf_counter(); sysm = cf.BridgeSystem(h, 16384); tb = time.perf_counter() - t0
    rec = {"H": h, "J": 16384, "build_s": tb, "sn": {}, "e9b": {}}
    for p in P_ALL:
        c = E2[h][p]; pp = float(p)
        A = ev(sysm, c["q_lo"]); B = ev(sysm, c["q_hi"])
        rec["sn"][p] = {"q_lo": c["q_lo"], "q_hi": c["q_hi"], "F_at_q_lo": [A["lo"], A["hi"]], "F_at_q_hi": [B["lo"], B["hi"]],
                        "bracket_ok": bool(A["hi"] < pp < B["lo"]), "old_F_at_q_lo": c["F_at_q_lo"], "old_F_at_q_hi": c["F_at_q_hi"],
                        "K": [A["K"], B["K"]], "seconds": A["seconds"] + B["seconds"], "budget_lo": A["budget"], "budget_hi": B["budget"]}
    if round(100 * float(h)) % 2 == 1:
        Ha = cf.exact_H(h)[1]; x = Ha - arb(1) / 2
        for p in P_ALL:
            fit = SF["fits"][p]; coef = [arb(f"{a:.6e}") for a in fit["coef"][: int(fit["K"]) + 1]]
            poly = arb(0)
            for a in reversed(coef):
                poly = poly * x + a
            qb = poly.exp() / (1 - Ha)
            qa, qb_ = cf.flo(qb), cf.fup(qb)
            s = rec["sn"][p]; pp = float(p)
            if s["q_lo"] <= qa and qb_ <= s["q_hi"]:
                lower, upper, route = s["F_at_q_lo"][0], s["F_at_q_hi"][1], "bracket (monotone)"
            else:
                A = ev(sysm, qa); B = ev(sysm, qb_); lower, upper, route = A["lo"], B["hi"], "direct certificate"
            rec["e9b"][p] = {"q_hat_lo": qa, "q_hat_hi": qb_, "route": route, "F_lo": lower, "F_hi": upper,
                             "bound_pp": 100 * max(pp - lower, upper - pp, 0.0)}
    if h == "0.75":
        T = ev(sysm, t4_row["q"]); rec["t4"] = {"q": t4_row["q"], "F": [T["lo"], T["hi"]], "old": [t4_row["cert_lo"], t4_row["cert_hi"]], "seconds": T["seconds"]}
    rec["total_s"] = time.perf_counter() - t0; rec["peak_mb"] = rss_mb()
    json.dump(rec, open(fo + ".tmp", "w"), indent=1); os.replace(fo + ".tmp", fo)
    ok = all(v["bracket_ok"] for v in rec["sn"].values())
    print(f"H={h}: build {tb:.0f}s total {rec['total_s']:.0f}s brackets_ok={ok} e9b={len(rec['e9b'])} peak {rec['peak_mb']:.0f} MB", flush=True)
    del sysm; gc.collect()
print("done", flush=True)
