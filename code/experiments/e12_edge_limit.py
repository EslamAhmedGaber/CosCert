"""E12: limit constants C_p = lim (1-H) q_p(H) as H -> 1 (Proposition 'edge'), computed (double precision, not certified).
k1(r,s) = -d/dH Cov(B_H(r)-rB_H(1), B_H(s)-sB_H(1)) at H=1 = -[phi(r)+phi(s)-phi(|r-s|) - s(phi(r)-phi(1-r)) - r(phi(s)-phi(1-s))],
phi(x) = x^2 log x.  Eigenvalues by Nystrom with Gauss-Legendre nodes; C_p solves P(Z^2 - C sum mu_k xi_k^2 <= 0) = p (Imhof).
Output: results/E12_edge_limit.json
"""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import laws as LW, gram as G_  # noqa: E402
def phi(x):
    x = np.asarray(x, float); out = np.zeros_like(x); m = x > 0; out[m] = x[m] ** 2 * np.log(x[m]); return out
def k1(r, s):
    R_, S_ = np.meshgrid(r, s, indexing="ij")
    return -(phi(R_) + phi(S_) - phi(np.abs(R_ - S_)) - S_ * (phi(R_) - phi(1 - R_)) - R_ * (phi(S_) - phi(1 - S_)))
def mu_eigs(N):
    x, w = np.polynomial.legendre.leggauss(N); r = (x + 1) / 2; w = w / 2
    ev = np.linalg.eigvalsh(np.sqrt(w)[:, None] * k1(r, r) * np.sqrt(w)[None, :])[::-1]; return ev[ev > 1e-15]
if __name__ == "__main__":
    RES = os.path.join(WORK, "results"); cells = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))["cells"]
    mu = mu_eigs(800); mu400 = mu_eigs(400)
    F = lambda c, m_: LW.imhof_cdf(np.concatenate([[1.0], -c * m_]), 0.0)[0]
    Cp = {p: brentq(lambda c: F(c, mu) - float(p), 1.0, 5000.0, xtol=1e-10) for p in ("0.800", "0.900", "0.950", "0.975", "0.980", "0.990")}
    c400 = brentq(lambda c: F(c, mu400) - 0.95, 1.0, 5000.0, xtol=1e-10)
    ext = {p: 2 * (0.01 * cells["0.99"][p]["q"]) - 0.02 * cells["0.98"][p]["q"] for p in Cp}
    rr = {**{f"HI|{m}": float(np.trace(np.asarray(G_.hi_fixed_m_cov(0.999, m))[1:, 1:]) / m / 0.001) for m in range(1, 17)},
          **{f"EWC|{B}": float(np.trace(np.asarray(G_.ewc_cov(0.999, B))[1:, 1:]) / B / 0.001) for B in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)}}
    out = {"trace_k1_numeric": float(mu.sum()), "trace_k1_exact": 7 / 36, "nystrom_N": 800, "mu_top": [float(x) for x in mu[:6]],
           "mu1_share": float(mu[0] / mu.sum()), "C_p": Cp, "C_0.95_N400": c400, "extrap_from_certified_0.98_0.99": ext,
           "extrap_rel_dev": {p: abs(ext[p] - Cp[p]) / Cp[p] for p in Cp}, "rho_over_1mH_at_0.999": rr, "rho_over_1mH_min": min(rr.values())}
    json.dump(out, open(os.path.join(RES, "E12_edge_limit.json"), "w"), indent=1); print(json.dumps({k: out[k] for k in ("C_p", "extrap_rel_dev", "rho_over_1mH_min")}, indent=1))
