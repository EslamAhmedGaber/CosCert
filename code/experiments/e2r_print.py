"""E2r, part (e) (17 Nov 2026): certified size of every PRINTED critical value d (the decimal shown in the tables) with
the final certificate, J = 16384. A decimal is in general not a binary float, so
    P(T_H <= d) in [F_lo(d_-), F_hi(d_+)],  d_- = largest float <= d,  d_+ = smallest float >= d   (monotonicity),
and the printed size 1 - P(T_H <= d) is enclosed by [1 - F_hi(d_+), 1 - F_lo(d_-)].
Resumable; an earlier record that evaluated only the nearest float is completed by the other neighbour.
Usage: python e2r_print.py [H1,H2,...]   -> results/E2r/print/H<h>.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import gc, json, math, sys, time  # noqa: E401,E402
from fractions import Fraction  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from flint import ctx  # noqa: E402
from p3 import certify as cf  # noqa: E402

ctx.prec = 128
RES = os.path.join(WORK, "results"); OUT = os.path.join(RES, "E2r", "print"); os.makedirs(OUT, exist_ok=True)
E2 = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
HSEL = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else None


def neighbours(c_print):
    d = Fraction(repr(float(c_print)))            # the printed decimal (repr of the stored float = the printed string)
    f = float(d)
    if Fraction(f) == d:
        return f, f, d
    if Fraction(f) < d:
        return f, math.nextafter(f, math.inf), d
    return math.nextafter(f, -math.inf), f, d


for h in sorted(E2, key=float):
    fo = os.path.join(OUT, f"H{h}.json")
    old = json.load(open(fo)) if os.path.exists(fo) else None
    if old is not None and old.get("version") == 2:
        continue
    if HSEL is not None and h not in HSEL:
        continue
    t0 = time.perf_counter(); sysm = cf.BridgeSystem(h, 16384)
    rec = {"H": h, "J": 16384, "version": 2, "build_s": time.perf_counter() - t0, "p": {}}
    for p, c in E2[h].items():
        lo_pt, hi_pt, d = neighbours(c["c_print"])
        near = float(d)
        have = None
        if old is not None and p in old["p"]:                     # previous record: enclosure of F at the nearest float
            s = old["p"][p]["size_at_print"]; have = (1 - s[1], 1 - s[0])
        def F(x):
            if have is not None and x == near:
                return have
            P = sysm.cdf_T_certified_tilt(float(x)); return (float(P["lo"]), float(P["hi"]))
        Flo = F(lo_pt)
        Fhi = Flo if hi_pt == lo_pt else F(hi_pt)
        rec["p"][p] = {"c_print": c["c_print"], "digits": c["digits"], "decimal": str(d), "lo_pt": lo_pt, "hi_pt": hi_pt,
                       "F_lo_at_lo_pt": Flo[0], "F_hi_at_hi_pt": Fhi[1], "size_at_print": [1 - Fhi[1], 1 - Flo[0]],
                       "old_size_at_print": c["size_at_print"]}
    rec["total_s"] = time.perf_counter() - t0
    json.dump(rec, open(fo + ".tmp", "w"), indent=1); os.replace(fo + ".tmp", fo)
    print(f"H={h}: {rec['total_s']:.0f}s", flush=True)
    del sysm; gc.collect()
print("done", flush=True)
