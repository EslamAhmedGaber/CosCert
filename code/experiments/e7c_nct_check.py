"""E7c (17 Nov 2026): check of the H = 1/2 fixed-smoothing local power against the noncentral Student-t power.
Reference: P(|T| > t) = E_V[ Phi(-(t sqrt(V/nu) - c)) + Phi(-t sqrt(V/nu) - c) ], V ~ chi2_nu, by adaptive quadrature
(scipy.stats.nct returns NaN for noncentrality above about 8.4 in SciPy 1.18, so it is not used).
usage: python e7c_nct_check.py  ->  results/E7c_nct_check.json
"""
import glob, json, math, os  # noqa: E401
import numpy as np
import pandas as pd
from scipy import integrate, stats

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")


def nct_power(t, nu, c):
    f = lambda v: (stats.norm.sf(t * math.sqrt(v / nu) - c) + stats.norm.cdf(-t * math.sqrt(v / nu) - c)) * stats.chi2.pdf(v, nu)
    hi = stats.chi2.isf(1e-18, nu)
    return integrate.quad(f, 0.0, hi, epsabs=1e-16, epsrel=1e-13, limit=500, points=[nu])[0]


if __name__ == "__main__":
    P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "E7_power_H*.csv")))], ignore_index=True)
    P["k"] = pd.to_numeric(P.param, errors="coerce").fillna(0).astype(int)
    h5 = P[np.isclose(P.H, 0.5) & P.procedure.isin(["HI", "EWC"])]
    mx, worst, n, scipy_nan = 0.0, None, 0, 0
    for (proc, k), g in h5.groupby(["procedure", "k"]):
        dof = 2 * k if proc == "HI" else k; t = stats.t.ppf(0.975, dof)
        sc = stats.nct.sf(t, dof, g.c.values) + stats.nct.cdf(-t, dof, g.c.values); scipy_nan += int(np.isnan(sc).sum())
        for c, p in zip(g.c.values, g.power.values):
            d = abs(p - nct_power(t, dof, c)); n += 1
            if d > mx:
                mx, worst = d, [proc, int(k), float(c)]
    json.dump({"cells": n, "max_abs_dev": mx, "worst": worst, "scipy_nct_nan_cells": scipy_nan,
               "reference": "quadrature over V ~ chi2_nu"}, open(os.path.join(RES, "E7c_nct_check.json"), "w"), indent=1)
    print(n, mx, worst, scipy_nan)
