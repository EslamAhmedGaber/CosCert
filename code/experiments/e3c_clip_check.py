"""E3c (17 Nov 2026, audit D4): negative eigenvalues of the positive semi-definite matrix K of finite_n.SNExact that are
set to zero (rounding). For every (model, H, n) of the E3 design: the number clipped, the most negative raw eigenvalue and
its size relative to the largest eigenvalue. Also the ARFIMA(1,d,0) autocovariance truncation bound.
usage: python e3c_clip_check.py  ->  results/E3c_clip_check.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, math, sys  # noqa: E401,E402
import numpy as np      # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import finite_n as fn  # noqa: E402

if __name__ == "__main__":
    out = {"cells": []}
    for model in ("fgn", "arfima0", "arfima1p", "arfima1m"):
        for H in (0.5, 0.6, 0.7, 0.8, 0.9):
            d = H - 0.5
            for n in (50, 100, 200, 500, 1000, 2000):
                if model == "fgn":
                    g = fn.acf_fgn(H, n)
                elif model == "arfima0":
                    g = fn.acf_arfima0(d, n)
                else:
                    g = fn.acf_arfima1(0.5 if model == "arfima1p" else -0.5, d, n)
                s = fn.SNExact(g)
                out["cells"].append({"model": model, "H": H, "n": n, "n_clipped": s.n_clipped, "min_raw": s.min_kappa_raw,
                                     "rel_to_max": s.min_kappa_raw / float(s.kappa[0])})
            print(model, H, flush=True)
    phi, tol = 0.5, 1e-18
    M = int(math.ceil(math.log(tol) / math.log(phi))) + 1
    out["arfima1_trunc_rel_bound"] = 2 * phi ** (M + 1) / ((1 - phi) * (1 - phi * phi))
    neg = [c for c in out["cells"] if c["n_clipped"] > 0]
    out["summary"] = {"cells": len(out["cells"]), "cells_with_clipping": len(neg),
                      "max_clipped": max((c["n_clipped"] for c in out["cells"]), default=0),
                      "most_negative_rel": min((c["rel_to_max"] for c in out["cells"]), default=0.0)}
    json.dump(out, open(os.path.join(WORK, "results", "E3c_clip_check.json"), "w"), indent=1)
    print(out["summary"], out["arfima1_trunc_rel_bound"])
