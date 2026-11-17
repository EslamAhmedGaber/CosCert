"""E10: matched example -- same model (fGn), same n, only the cutoff changes.

Reproduces the author's independent double-precision check (SN statistic, squared form, two-sided 5 %):
  Brownian (H = 1/2) cutoff 45.52608575 applied at H = .5, .9, .99 and n = 64, 128;
  the H = .99 limit cutoff 6829.9 applied at n = 128.

Statistic on the finite grid (exact Gaussian finite-n law, no limit argument):
  T_n = n (sum_t x_t)^2 / sum_{k=1}^n S_k^2,   S_k = sum_{t<=k} (x_t - xbar),   x ~ fGn(H), Var x_t = 1.
P(T_n > c) = P(x' A_c x > 0),  A_c = 11' - (c/n) M'M,  M = L (I - 11'/n),  L = lower-triangular ones.

Route A: library SNExact (p3.finite_n; Ruben-type series on the (rank-structured) quadratic form).
Route B: independent code (no p3 import): eigenvalues of R' A_c R (Gamma = R R'), then Imhof's integral
         P(Q > 0) = 1/2 + (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du evaluated with mpmath at 30 digits.
Output: results/E10_matched_example.csv (+ .json with metadata).
"""
import json, math, os, sys, time
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
RES = os.path.join(WORK, "results")

CASES = [  # (H, n, cutoff, label)
    (0.50, 64, 45.52608575, "Brownian cutoff"), (0.50, 128, 45.52608575, "Brownian cutoff"),
    (0.90, 64, 45.52608575, "Brownian cutoff"), (0.90, 128, 45.52608575, "Brownian cutoff"),
    (0.99, 64, 45.52608575, "Brownian cutoff"), (0.99, 128, 45.52608575, "Brownian cutoff"),
    (0.99, 128, 6829.9, "H = 0.99 limit cutoff (rounded)"),
    (0.99, 128, 6829.9404006874, "H = 0.99 limit cutoff (certified point)"),
]
USER = {(0.50, 64, 45.52608575): 5.0167, (0.50, 128, 45.52608575): 5.0042, (0.90, 64, 45.52608575): 42.8085,
        (0.90, 128, 45.52608575): 42.8081, (0.99, 64, 45.52608575): 79.5227, (0.99, 128, 45.52608575): 79.5227,
        (0.99, 128, 6829.9): 5.00008}


def fgn_cov(H, n):
    h = np.arange(n, dtype=float)
    g = 0.5 * (np.abs(h + 1) ** (2 * H) - 2 * h ** (2 * H) + np.abs(h - 1) ** (2 * H))
    i = np.arange(n)
    return g[np.abs(i[:, None] - i[None, :])]


def imhof_weights(H, n, c):
    G = fgn_cov(H, n); R = np.linalg.cholesky(G)
    one = np.ones((n, 1)); L = np.tril(np.ones((n, n))); M = L @ (np.eye(n) - one @ one.T / n)
    A = one @ one.T - (c / n) * (M.T @ M)
    W = R.T @ A @ R
    return np.linalg.eigvalsh(0.5 * (W + W.T))


def imhof_upper0(lam, dps=30):
    mp.mp.dps = dps
    lam = [mp.mpf(float(x)) for x in lam if abs(x) > 1e-300]
    def f(u):
        if u == 0:
            return mp.mpf(0.5) * mp.fsum(lam)                     # limit of sin(theta)/(u rho) as u -> 0
        th = mp.fsum(mp.atan(l * u) for l in lam) / 2
        lr = mp.fsum(mp.log1p((l * u) ** 2) for l in lam) / 4
        return mp.sin(th) / (u * mp.exp(lr))
    lmax = max(abs(x) for x in lam)
    brk = [0] + [mp.mpf(10) ** k / lmax for k in range(-3, 9)] + [mp.inf]
    return mp.mpf(0.5) + mp.quad(f, brk) / mp.pi


def main():
    from p3 import finite_n as fn
    rows = []
    for H, n, c, lab in CASES:
        t0 = time.time(); a = 1.0 - fn.SNExact(fn.acf_fgn(H, n)).cdf(c); ta = time.time() - t0
        t0 = time.time(); lam = imhof_weights(H, n, c); b = float(imhof_upper0(lam)); tb = time.time() - t0
        u = USER.get((H, n, c))
        rows.append({"H": H, "n": n, "cutoff": c, "cutoff_label": lab, "size_routeA_pct": 100 * a, "size_routeB_pct": 100 * b,
                     "A_minus_B_pp": 100 * (a - b), "user_check_pct": u, "routeA_minus_user_pp": (100 * a - u) if u is not None else None,
                     "n_pos_weights": int((lam > 0).sum()), "seconds_A": ta, "seconds_B": tb,
                     "kind": "exact finite-grid law (fGn, Gaussian), not a limit"})
        print(f"H={H} n={n} c={c}: A={100*a:.6f}%  B={100*b:.6f}%  user={u}  ({ta:.2f}s, {tb:.1f}s)", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "E10_matched_example.csv"), index=False)
    json.dump({"description": __doc__, "max_abs_A_minus_B_pp": float(df.A_minus_B_pp.abs().max()),
               "max_abs_A_minus_user_pp": float(df.routeA_minus_user_pp.dropna().abs().max())},
              open(os.path.join(RES, "E10_matched_example.json"), "w"), indent=1)
    print("max |A-B| pp:", df.A_minus_B_pp.abs().max(), " max |A-user| pp:", df.routeA_minus_user_pp.dropna().abs().max())


if __name__ == "__main__":
    main()
