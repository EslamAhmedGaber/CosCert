"""E7b (17 Nov 2026, audit C4): convergence of the computed Bartlett/self-normalised local power in the Galerkin
truncation J. For each H of the E7 grid and every c of C_GRID, the power of e7_power.sn_power at J = 128, 256, 512, 1024;
reports max_c |power_J - power_1024| and max_c |power_J - power_512|.
usage: python e7b_power_jladder.py  ->  results/E7b_sn_power_Jladder.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time  # noqa: E401,E402
import numpy as np      # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e7_power as e7   # noqa: E402

if __name__ == "__main__":
    tab = json.load(open(os.path.join(e7.RES, "E2_sn_critical_values.json")))
    out = {"J": [128, 256, 512, 1024], "c_grid": [float(c) for c in e7.C_GRID], "H": {}}
    for H in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
        q = tab["cells"][f"{H:.2f}"]["0.950"]["q"]
        t0 = time.perf_counter()
        pw = {J: e7.sn_power(H, q, J=J) for J in out["J"]}
        ref, r512 = np.array(pw[1024]), np.array(pw[512])
        out["H"][f"{H:.2f}"] = {"q": q, "power": {str(J): pw[J] for J in out["J"]},
                                "maxdiff_vs_1024": {str(J): float(np.max(np.abs(np.array(pw[J]) - ref))) for J in out["J"]},
                                "maxdiff_vs_512": {str(J): float(np.max(np.abs(np.array(pw[J]) - r512))) for J in out["J"]},
                                "seconds": time.perf_counter() - t0}
        print(H, out["H"][f"{H:.2f}"]["maxdiff_vs_1024"], f"{time.perf_counter() - t0:.0f}s", flush=True)
        json.dump(out, open(os.path.join(e7.RES, "E7b_sn_power_Jladder.json"), "w"), indent=1)
