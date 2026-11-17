"""Exact finite-dimensional limit laws of the fixed-smoothing statistics, with certified evaluation.

  Hualde-Iacone fixed-m:  tau^2 => Z^2 / (m^-1 sum_{j<=m} (C_j^2 + S_j^2)),  (Z, C, S) ~ N(0, Sigma_HI(H, m))
  EWC (cosine series):    t^2  => Z^2 / (B^-1 sum_{j<=B} Lambda_j^2),     (Z, Lambda) ~ N(0, G_{0..B})
P(stat^2 <= c) = P(X_c <= 0),  X_c = Y_0^2 - c * scale * sum_{i>=1} Y_i^2  (one positive ratio-form weight, Sylvester).

Certificate: Gram entries in arb (Paper 1 closed cosine Gram; closed sine Gram of the vendored Round 17 code, re-derived
in the manuscript), the ratio-form weights = eigenvalues of M Sigma enclosed by an exact similarity V^-1 (M Sigma) V
computed in arb plus Gershgorin clusters (a connected union of k discs holds exactly k eigenvalues), then Ruben's
Beta mixture in arb with monotone coupling over the weight intervals (laws.ruben_beta_cdf0_arb).
"""
import math

import numpy as np
from flint import arb, acb, arb_mat, ctx

from .certify import exact_H, fup, flo
from . import laws as _laws


class GramArb:
    """Cosine Gram G_kl (k, l = 0..K) and sine Gram S_ij = <sin(i pi .), sin(j pi .)>_H (i, j = 1..K) in arb."""

    def __init__(self, H, K, prec=128):
        ctx.prec = prec
        self.prec = prec
        self.fr, self.Ha = exact_H(H)
        Ha = self.Ha
        b = 2 * Ha - 1
        self.b, self.C, self.pi = b, 2 * Ha, arb.pi()
        self.P, self.R, self.Q = [arb(1)], [arb(0)], [arb(1)]
        for k in range(1, K + 1):
            z = acb(0, k * self.pi)
            f = z.hypgeom_1f1(b, b + 1)
            g = z.hypgeom_1f1(b + 1, b + 2)
            self.P.append(f.real)
            self.R.append(f.imag)
            self.Q.append(g.real)
        self.K = K

    def G(self, k, l):
        if k > l:
            k, l = l, k
        if (k + l) % 2:
            return arb(0)
        if k == 0 and l == 0:
            return arb(1)
        if k == 0:
            return -arb(2).sqrt() * self.C * self.R[l] / (self.pi * l)
        if k == l:
            return self.C * (self.P[k] - (1 - self.b) * self.R[k] / (k * self.pi))
        return 2 * self.C / self.pi * (l * self.R[l] - k * self.R[k]) / (k * k - l * l)

    def S(self, i, j):
        if (i + j) % 2:
            return arb(0)
        Ha, b = self.Ha, self.b
        if i == j:
            return Ha * (self.P[i] + self.R[i] / (i * self.pi)) - Ha * b / (b + 1) * self.Q[i]
        return Ha / self.pi * ((self.R[i] + self.R[j]) / (i + j) - (self.R[i] - self.R[j]) / (i - j))


def sigma_hi_arb(ga, m):
    """Covariance of (Z, C_1..C_m, S_1..S_m) for the Hualde-Iacone fixed-m limit (arb_mat)."""
    n = 2 * m + 1
    Sg = arb_mat(n, n)
    Sg[0, 0] = ga.G(0, 0)
    sq2 = arb(2).sqrt()
    for j in range(1, m + 1):
        v = ga.G(0, 2 * j) / sq2
        Sg[0, j] = v
        Sg[j, 0] = v
        for k in range(1, m + 1):
            Sg[j, k] = ga.G(2 * j, 2 * k) / 2
            Sg[m + j, m + k] = ga.S(2 * j, 2 * k)
    return Sg


def sigma_ewc_arb(ga, B):
    n = B + 1
    Sg = arb_mat(n, n)
    for k in range(n):
        for l in range(n):
            Sg[k, l] = ga.G(k, l)
    return Sg


def arb_to_np(M):
    return np.array([[float(M[i, j].mid()) for j in range(M.ncols())] for i in range(M.nrows())])


def ratio_weights_certified(Sg, c, k):
    """Enclosures [lo_i, hi_i] (sorted descending) of the eigenvalues of M Sigma, M = diag(1, -c/k, ...), with the
    exact binary value c and the exact integer k.  Exact similarity in arb + Gershgorin clusters."""
    n = Sg.nrows()
    ca = arb(float(c)) / k
    Mx = arb_mat(n, n)
    Mx[0, 0] = arb(1)
    for i in range(1, n):
        Mx[i, i] = -ca
    A = Mx * Sg
    Ad = arb_to_np(A)
    w, V = np.linalg.eig(Ad)
    V = np.real_if_close(V, tol=1e6)
    if np.iscomplexobj(V):
        raise ValueError("complex eigenvectors")
    Va = arb_mat(n, n, V.ravel().tolist())
    T = Va.inv() * A * Va
    cen = [T[i, i] for i in range(n)]
    rad = []
    for i in range(n):
        r = arb(0)
        for j in range(n):
            if j != i:
                r += T[i, j].abs_upper()
        rad.append(r)
    lo = np.array([flo(cen[i] - rad[i]) for i in range(n)])
    hi = np.array([fup(cen[i] + rad[i]) for i in range(n)])
    # clusters of overlapping intervals: every member gets the hull
    order = np.argsort(lo)
    cl_lo, cl_hi = lo.copy(), hi.copy()
    groups, cur = [], [order[0]]
    for a, b_ in zip(order[:-1], order[1:]):
        if lo[b_] <= max(hi[i] for i in cur):
            cur.append(b_)
        else:
            groups.append(cur)
            cur = [b_]
    groups.append(cur)
    for g in groups:
        L, Hh = min(lo[i] for i in g), max(hi[i] for i in g)
        for i in g:
            cl_lo[i], cl_hi[i] = L, Hh
    lo_s = np.sort(cl_lo)[::-1]
    hi_s = np.sort(cl_hi)[::-1]
    if not (lo_s[0] > 0 and hi_s[1] < 0):
        raise ValueError("inertia not certified (need one positive and n-1 negative weights)")
    return lo_s, hi_s


class FixedSmoothingLaw:
    """Exact limit law of a fixed-smoothing statistic; kind = 'HI' (parameter m) or 'EWC' (parameter B)."""

    def __init__(self, H, kind, k, prec=128, ga=None):
        self.H, self.kind, self.k = H, kind, k
        need = 2 * k + 1 if kind == "HI" else k + 1
        self.ga = ga if ga is not None and ga.K >= need and ga.fr == exact_H(H)[0] else GramArb(H, need, prec)
        if kind == "HI":
            self.Sg = sigma_hi_arb(self.ga, k)
            self.scale = 1.0 / k
        elif kind == "EWC":
            self.Sg = sigma_ewc_arb(self.ga, k)
            self.scale = 1.0 / k
        else:
            raise ValueError(kind)
        self.Sd = arb_to_np(self.Sg)
        self.Sd = 0.5 * (self.Sd + self.Sd.T)

    # computed (double): P(stat^2 <= c)
    def cdf2(self, c):
        L = np.linalg.cholesky(self.Sd)
        d = np.full(self.Sd.shape[0], -c * self.scale)
        d[0] = 1.0
        th = np.sort(np.linalg.eigvalsh(L.T @ (d[:, None] * L)))[::-1]
        return _laws.ruben_beta_cdf0(th)[0]

    # certified: enclosure of P(stat^2 <= c) for the exact binary value c
    def cdf2_certified(self, c):
        lo_w, hi_w = ratio_weights_certified(self.Sg, c, self.k)
        lo, hi, K = _laws.ruben_beta_cdf0_arb_fast(lo_w, hi_w)
        return lo, hi

    def quantile2(self, p, lo=1e-6, hi=None):
        from scipy.optimize import brentq
        hi = hi or 10.0
        while self.cdf2(hi) < p:
            hi *= 2
        return brentq(lambda c: self.cdf2(c) - p, lo, hi, xtol=1e-14, rtol=1e-15)

    def two_sided_cv_certified(self, alpha, rel=1e-9):
        """Two-sided level-alpha critical value x (P(|stat| > x) = alpha): certified bracket [x_lo, x_hi]
        with P(stat^2 <= x_lo^2) < 1 - alpha < P(stat^2 <= x_hi^2)."""
        p = 1.0 - alpha
        q = self.quantile2(p)
        for r in (rel, 10 * rel, 100 * rel, 1e-6, 1e-5):
            c_lo, c_hi = q * (1 - r), q * (1 + r)
            Flo = self.cdf2_certified(c_lo)
            Fhi = self.cdf2_certified(c_hi)
            if Flo[1] < p < Fhi[0]:
                return {"x": math.sqrt(q), "x_lo": flo(arb(c_lo).sqrt()), "x_hi": fup(arb(c_hi).sqrt()),
                        "c2": q, "F_at_c2_lo": Flo, "F_at_c2_hi": Fhi, "rel_bracket": r}
        raise RuntimeError("quantile not certified")
