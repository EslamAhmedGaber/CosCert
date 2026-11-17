"""E7: local power in the limit experiment (exact for fixed smoothing, computed for SN), H0: mu = mu0 against mu - mu0 = c * sigma * n^(H-1).

In the limit the numerator Z = B_H(1) becomes Z + c while every self-normaliser is unchanged, so the rejection event
(Z + c)^2 - q * D > 0 is a NONCENTRAL Gaussian quadratic form.  With xi = (Z, components) ~ N(mu = (c, 0, ..), Sigma),
A = diag(1, -q w, ..), L = chol(Sigma), L'AL = P Lambda P':  Q = sum_i lambda_i (w_i + delta_i)^2 exactly
(delta = Lambda^-1 P'L'A mu; the constant mu'A mu - sum b_i^2/lambda_i vanishes identically), so power = P(Q > 0)
is a noncentral weighted chi-square probability evaluated by Imhof's (1961) noncentral formula.
  HI fixed-m, EWC: exact finite-dimensional ('exact').
  SN (T_H): Galerkin J = 512 sine coordinates of the bridge + mean-remainder shift ('computed', null error ~2e-10, E5 T4).
Checks: c = 0 reproduces the nominal level; at H = 1/2 the HI/EWC power equals the noncentral-t power (scipy.stats.nct).
usage: python e7_power.py <H> [<H> ...]   ->  results/E7_power_H<H>.csv
"""
import csv, json, math, os, sys, time
import numpy as np
from scipy import integrate, stats

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import gram as gr  # noqa: E402

RES = os.path.join(WORK, "results")
C_GRID = np.round(np.arange(0.0, 12.0001, 0.2), 2)
HI_M = (1, 2, 4, 8, 16); EWC_B = (2, 4, 8, 16, 32)


def imhof_sf_nc(lam, delta, y=0.0, epsabs=1e-12, epsrel=1e-10, limit=2000):
    """P(sum lam_i (w_i + delta_i)^2 > y), w iid N(0,1)  (Imhof 1961, eq. 3.2 with h_i = 1)."""
    lam = np.asarray(lam, float); d2 = np.asarray(delta, float) ** 2

    def f(u):
        lu = lam * u; den = 1.0 + lu * lu
        th = 0.5 * np.sum(np.arctan(lu) + d2 * lu / den) - 0.5 * y * u
        lrho = 0.25 * np.sum(np.log(den)) + 0.5 * np.sum(d2 * lu * lu / den)
        return math.sin(th) * math.exp(-lrho) / u

    v, err = integrate.quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return 0.5 + v / math.pi


def nc_form(Sigma, wdiag, c):
    """lambda_i, delta_i of Q = (xi_0 + c)^2 - sum_k wdiag_k xi_k^2 with xi ~ N(0, Sigma)."""
    L = np.linalg.cholesky(Sigma); A = np.diag(np.concatenate([[1.0], -np.asarray(wdiag, float)]))
    M = L.T @ A @ L; M = 0.5 * (M + M.T); lam, P = np.linalg.eigh(M)
    mu = np.zeros(Sigma.shape[0]); mu[0] = c
    b = P.T @ (L.T @ (A @ mu)); keep = np.abs(lam) > 1e-15 * np.abs(lam).max()
    return lam[keep], b[keep] / lam[keep]


def fixed_power(H, kind, k, alpha=0.05):
    from p3 import lowrank as lr
    law = lr.FixedSmoothingLaw(f"{H:.2f}", kind, k); q = law.quantile2(1 - alpha)
    Sg = np.asarray(gr.hi_fixed_m_cov(H, k) if kind == "HI" else gr.ewc_cov(H, k), float)
    w = np.full(Sg.shape[0] - 1, q / k)          # denominator = k^-1 sum of squares of the k (EWC) or 2k (HI: C, S) components
    return q, [imhof_sf_nc(*nc_form(Sg, w, c)) for c in C_GRID]


def sn_power(H, q, J=512):
    G = gr.cosine_gram(H, J + 1); kk = np.arange(1, J + 1)
    Sig = np.empty((J + 1, J + 1)); Sig[0, 0] = 1.0; Sig[0, 1:] = Sig[1:, 0] = G[0, 1:J + 1] / (kk * np.pi)
    Sig[1:, 1:] = G[1:J + 1, 1:J + 1] / np.outer(kk * np.pi, kk * np.pi)
    tau = gr.trace_bridge(H) - float(np.sum(np.diag(Sig)[1:]))
    out = []
    for c in C_GRID:
        lam, dl = nc_form(Sig, np.full(J, q), c)
        out.append(imhof_sf_nc(lam, dl, y=q * tau))
    return out


def main(Hs):
    tab = json.load(open(os.path.join(RES, "E2_sn_critical_values.json")))
    for H in Hs:
        rows = []; t0 = time.perf_counter()
        q_sn = tab["cells"][f"{H:.2f}"]["0.950"]["q"]
        pw = sn_power(H, q_sn)
        rows += [[H, "SN", "", q_sn, c, p, "computed (Galerkin J=512 + mean remainder)"] for c, p in zip(C_GRID, pw)]
        for kind, ks in (("HI", HI_M), ("EWC", EWC_B)):
            for k in ks:
                q, pw = fixed_power(H, kind, k)
                rows += [[H, kind, k, math.sqrt(q), c, p, "exact"] for c, p in zip(C_GRID, pw)]
                if abs(H - 0.5) < 1e-12:      # independence at H = 1/2: power = noncentral t
                    dof = 2 * k if kind == "HI" else k; t = stats.t.ppf(0.975, dof)
                    ref = [stats.nct.sf(t, dof, c) + stats.nct.cdf(-t, dof, c) for c in C_GRID]
                    print(f"  check {kind} {k}: max |power - nct| = {max(abs(a - b) for a, b in zip(pw, ref)):.2e}", flush=True)
        zpow = [stats.norm.sf(1.959963984540054 - c) + stats.norm.cdf(-1.959963984540054 - c) for c in C_GRID]
        rows += [[H, "oracle z (known variance)", "", 1.959963984540054, c, p, "exact"] for c, p in zip(C_GRID, zpow)]
        sz = {(r[1], r[2]): r[5] for r in rows if r[4] == 0.0}
        print(f"H={H}: power at c=0 (should be 0.05): max dev {max(abs(v - 0.05) for v in sz.values()):.2e};  "
              f"c=4: SN {pw_at(rows, 'SN', '', 4.0):.3f} HI4 {pw_at(rows, 'HI', 4, 4.0):.3f} HI16 {pw_at(rows, 'HI', 16, 4.0):.3f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        with open(os.path.join(RES, f"E7_power_H{H:.2f}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["H", "procedure", "param", "critical_value", "c", "power", "status"]); w.writerows(rows)


def pw_at(rows, proc, k, c):
    return next(r[5] for r in rows if r[1] == proc and r[2] == k and abs(r[4] - c) < 1e-9)


if __name__ == "__main__":
    main([float(h) for h in sys.argv[1:]])
