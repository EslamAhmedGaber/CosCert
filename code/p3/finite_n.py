"""Exact finite-sample laws of studentised-mean statistics under Gaussian stationary models.

Data x_t = mu + u_t, t = 1..n, u Gaussian with autocovariance gamma (fGn, ARFIMA(0,d,0), ARFIMA(1,d,0), AR(1)).
Under H0 every statistic below is a ratio of quadratic forms in u, so P(stat^2 <= c) = P(X_c <= 0) with
X_c = Y_0^2 - c * (PSD form): an exact finite-dimensional law.

  SN / Lobato (2001) / Bartlett b = 1 (KVB):  T_n = n xbar^2 / C_n,  C_n = n^-2 sum_t S_t^2,  S_t = sum_{s<=t} (x_s - xbar);
      u = L xi (Gamma = L L^T), Z_n = w^T xi with w = L^T 1 / |L^T 1|,  n^-2 sum S_t^2 = xi^T K xi,  K = G^T G / n^2,
      G = D (I - 11^T/n) L (D = cumulative sum).  One eigendecomposition K = Q diag(kappa) Q^T per (model, n);
      phi(s) = det(I + z K)^(-1/2) (1 - 2 i s u(z))^(-1/2),  u(z) = sum_i wt_i^2 / (1 + z kappa_i),  z = 2 i s c n / (1^T Gamma 1).
  Hualde-Iacone (2017) Daniell tau:  tau^2 <= c  <=>  (sum u)^2 <= (c/m) sum_j |sum_t u_t e^{i lambda_j t}|^2,  lambda_j = 2 pi j / n.
  EWC (Lazarus et al. 2018, eq. 10):  t^2 <= c  <=>  (sum u)^2 / n <= (c/B) sum_j Lambda_j^2,  Lambda_j = sqrt(2/n) sum_t cos(pi j (t - 1/2)/n) u_t.
The low-rank laws use the (2m+1)- or (B+1)-dimensional covariance C Gamma C^T and Ruben's Beta mixture; the SN law uses the
tilted Bromwich midpoint rule of certify.py in double precision with the head/power-sum split of the spectrum of K.
"""
import math

import numpy as np
from scipy.linalg import cholesky, eigh, toeplitz, matmul_toeplitz
from scipy.optimize import brentq
from scipy.special import gammaln

from . import laws as _laws


# ============================================================================ autocovariances (unit innovation variance)
def acf_fgn(H, n):
    k = np.arange(n, dtype=float)
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def acf_arfima0(d, n):
    g = np.empty(n)
    g[0] = math.exp(gammaln(1 - 2 * d) - 2 * gammaln(1 - d))
    for k in range(1, n):
        g[k] = g[k - 1] * (k - 1 + d) / (k - d)
    return g


def acf_arfima1(phi, d, n, tol=1e-18):
    """(1 - phi L) y = x,  x ~ ARFIMA(0,d,0):  gamma_y(k) = (1 - phi^2)^-1 sum_m phi^|m| gamma_x(k + m)  (exact series).
    Truncation at |m| <= M with |phi|^M <= tol; since |gamma_x(j)| <= gamma_x(0), the omitted terms are bounded by
    2 gamma_x(0) |phi|^(M+1) / ((1 - |phi|)(1 - phi^2))  (about 1e-18 gamma_x(0) at |phi| = 0.5, tol = 1e-18)."""
    if phi == 0:
        return acf_arfima0(d, n)
    M = int(math.ceil(math.log(tol) / math.log(abs(phi)))) + 1
    gx = acf_arfima0(d, n + M + 1)
    k = np.arange(n)
    gy = gx[k].copy()
    for m in range(1, M + 1):
        gy += phi ** m * (gx[np.abs(k - m)] + gx[k + m])
    return gy / (1 - phi * phi)


def acf_ar1(phi, n):
    return phi ** np.arange(n, dtype=float) / (1 - phi * phi)


def make_acf(model, n):
    """model = ('fgn', H) | ('arfima0', d) | ('arfima1', phi, d) | ('ar1', phi) | ('wn',)."""
    kind = model[0]
    if kind == "fgn":
        return acf_fgn(model[1], n)
    if kind == "arfima0":
        return acf_arfima0(model[1], n)
    if kind == "arfima1":
        return acf_arfima1(model[1], model[2], n)
    if kind == "ar1":
        return acf_ar1(model[1], n)
    if kind == "wn":
        g = np.zeros(n)
        g[0] = 1.0
        return g
    raise ValueError(model)


# ============================================================================ SN / Bartlett b = 1: exact law via one eigendecomposition
class SNExact:
    def __init__(self, gamma):
        n = gamma.size
        self.n = n
        L = cholesky(toeplitz(gamma), lower=True, overwrite_a=True, check_finite=False)
        s = L.sum(axis=0)                      # L^T 1
        v = float(s @ s)                       # 1^T Gamma 1
        w = s / math.sqrt(v)
        L -= L.mean(axis=0, keepdims=True)     # (I - 11^T/n) L
        np.cumsum(L, axis=0, out=L)            # D (I - 11^T/n) L
        K = L.T @ L
        del L
        K /= float(n) ** 2
        kappa, Q = eigh(K, overwrite_a=True, check_finite=False)
        self.n_clipped = int(np.sum(kappa < 0))            # K is PSD; negative eigenvalues are rounding (recorded, D4)
        self.min_kappa_raw = float(kappa.min())
        kappa = np.clip(kappa, 0.0, None)
        wt2 = (Q.T @ w) ** 2
        del Q, K
        order = np.argsort(kappa)[::-1]
        self.kappa = kappa[order]
        self.w2 = wt2[order]
        self.w2 /= self.w2.sum()
        self.scale = n / v                     # c' = c * n / (1^T Gamma 1)

    # u(z) and log det(I + zK) with a head / power-sum split (tail |z| kappa <= 0.05; 8 terms, remainder < 1e-12 relative)
    def _prep(self, zmax):
        head = self.kappa * zmax > 0.05
        nh = int(head.sum())
        kt, wt = self.kappa[nh:], self.w2[nh:]
        P = [np.sum(kt ** j) for j in range(1, 9)]
        Wp = [np.sum(wt * kt ** j) for j in range(0, 9)]
        return nh, P, Wp

    def _logdet_u(self, z, prep):
        nh, P, Wp = prep
        kh, wh = self.kappa[:nh], self.w2[:nh]
        d = 1.0 + z[:, None] * kh[None, :]
        ld = np.log(d).sum(axis=1)
        u = (wh[None, :] / d).sum(axis=1)
        zp = np.ones_like(z)
        for j in range(1, 9):
            zp = zp * z
            ld = ld + ((-1) ** (j + 1)) * zp * P[j - 1] / j
        zp = np.ones_like(z)
        for j in range(0, 9):
            u = u + ((-1) ** j) * zp * Wp[j]
            zp = zp * z
        return ld, u

    def theta_plus(self, cp):
        prep = self._prep(cp * 1e3)

        def h(th):
            z = np.array([cp / th + 0j])
            _, u = self._logdet_u(z, prep)
            return float(u[0].real) - th
        if h(1.0) >= 0:
            return 1.0
        return brentq(h, 1e-10, 1.0, xtol=1e-15, rtol=1e-15)

    def cdf(self, c, theta=0.25, eps=1e-13):
        """P(T_n <= c) (computed; numerical error < 1e-10)."""
        cp = c * self.scale
        L = 20.0
        while _alias_tilt(0.0, L, theta) > eps:
            L *= 1.05
        Delta = 2 * math.pi / L
        nu = self.kappa[1:]
        K = 32
        while _trunc(cp, (K - 0.5) * Delta, theta, nu) > eps:
            K = int(K * 1.1) + 1
        sg = (np.arange(K) + 0.5) * Delta
        s = sg - 1j * theta
        z = 2j * s * cp
        prep = self._prep(float(np.max(np.abs(z))))
        ld, u = self._logdet_u(z, prep)
        tp = self.theta_plus(cp)
        tis = 2j * s
        p1 = 1.0 - tis * tp
        r = (1.0 - tis * u) / p1
        M = np.exp(-0.5 * (ld + np.log(p1) + np.log(r)))
        w = theta + 1j * sg
        P_gt = Delta / math.pi * math.fsum(np.real(M / w).tolist())
        return 1.0 - P_gt

    def reject(self, cv):
        return 1.0 - self.cdf(cv)


def _alias_tilt(x, L, theta):
    a = math.exp(-theta * L)
    e = a / (1 - a)
    m = 1
    while m < 200:
        y = x + m * L
        t = math.exp(theta * m * L) * math.erfc(math.sqrt(max(y, 0.0) / 2))
        e += t
        if t < 1e-30:
            break
        m += 1
    return e


def _trunc(cp, sK, theta, nu):
    a = 2 * sK * cp * nu
    sel = a[a > 1.0]
    m = sel.size
    if m == 0:
        return math.inf
    return 1.0 / math.sqrt(1 - 2 * theta) * 2.0 / (math.pi * m) * math.exp(-0.5 * np.sum(np.log(sel)))


# ============================================================================ low-rank statistics (HI fixed-m, EWC)
def _toeplitz_apply(gamma, X):
    """Gamma @ X for symmetric Toeplitz Gamma (first column gamma), X of shape (n, k)."""
    return matmul_toeplitz((gamma, gamma), X, check_finite=False)


def hi_design(n, m):
    t = np.arange(1, n + 1, dtype=float)
    C = np.empty((n, 2 * m + 1))
    C[:, 0] = 1.0
    for j in range(1, m + 1):
        lam = 2 * math.pi * j / n
        C[:, j] = np.cos(lam * t)
        C[:, m + j] = np.sin(lam * t)
    return C


def ewc_design(n, B):
    t = np.arange(1, n + 1, dtype=float)
    C = np.empty((n, B + 1))
    C[:, 0] = 1.0 / math.sqrt(n)
    for j in range(1, B + 1):
        C[:, j] = math.sqrt(2.0 / n) * np.cos(math.pi * j * (t - 0.5) / n)
    return C


class LowRankExact:
    """kind 'HI' (k = m) or 'EWC' (k = B): exact finite-n law of stat^2 via the (k')-dimensional covariance C^T Gamma C."""

    def __init__(self, gamma, kind, k):
        n = gamma.size
        C = hi_design(n, k) if kind == "HI" else ewc_design(n, k)
        S = C.T @ _toeplitz_apply(gamma, C)
        self.S = 0.5 * (S + S.T)
        self.kind, self.k, self.n = kind, k, n

    def weights(self, c):
        L = np.linalg.cholesky(self.S)
        d = np.full(self.S.shape[0], -c / self.k)
        d[0] = 1.0
        return np.sort(np.linalg.eigvalsh(L.T @ (d[:, None] * L)))[::-1]

    def cdf2(self, c):
        return _laws.ruben_beta_cdf0(self.weights(c))[0]

    def reject_two_sided(self, cv):
        """P(|stat| > cv)."""
        return 1.0 - self.cdf2(cv * cv)


# ============================================================================ simulation (circulant embedding) and local Whittle
class CirculantSampler:
    def __init__(self, gamma):
        n = gamma.size
        row = np.concatenate([gamma, gamma[-2:0:-1]])
        lam = np.fft.rfft(row).real if False else np.fft.fft(row).real
        if lam.min() < -1e-10 * lam.max():
            raise ValueError(f"circulant embedding not nonnegative (min {lam.min():.3e})")
        self.sq = np.sqrt(np.clip(lam, 0, None) / row.size)
        self.n, self.N2 = n, row.size

    def sample(self, nrep, rng):
        Z = rng.standard_normal((nrep, self.N2)) + 1j * rng.standard_normal((nrep, self.N2))
        Y = np.fft.fft(Z * self.sq[None, :], axis=1)
        return np.concatenate([Y.real[:, :self.n], Y.imag[:, :self.n]], axis=0)


def periodogram_fourier(X, m):
    """2 pi I(lambda_j), j = 1..m, I = |(2 pi n)^-1/2 sum_t x_t e^{i lambda_j t}|^2 (rows of X are series)."""
    n = X.shape[1]
    F = np.fft.fft(X, axis=1)[:, 1:m + 1]
    return (np.abs(F) ** 2) / n


def local_whittle(X, m, grid=np.linspace(-0.49, 0.49, 197)):
    """Robinson (1995) local Whittle estimate of d for each row (grid search + parabolic refinement)."""
    n = X.shape[1]
    I = periodogram_fourier(X, m)
    lam = 2 * math.pi * np.arange(1, m + 1) / n
    llam = np.log(lam)
    mean_llam = llam.mean()
    R = np.empty((X.shape[0], grid.size))
    for i, d in enumerate(grid):
        R[:, i] = np.log(np.mean(I * lam[None, :] ** (2 * d), axis=1)) - 2 * d * mean_llam
    j = np.clip(np.argmin(R, axis=1), 1, grid.size - 2)
    r0, r1, r2 = R[np.arange(R.shape[0]), j - 1], R[np.arange(R.shape[0]), j], R[np.arange(R.shape[0]), j + 1]
    h = grid[1] - grid[0]
    den = r0 - 2 * r1 + r2
    off = np.where(den > 0, 0.5 * (r0 - r2) / np.where(den > 0, den, 1.0), 0.0)
    return np.clip(grid[j] + off * h, -0.49, 0.49)


def stats_from_samples(X, m_list=(), B_list=()):
    """SN T_n, HI tau (per m) and EWC t (per B) for each row of X (mu = 0 under H0)."""
    n = X.shape[1]
    out = {}
    xbar = X.mean(axis=1)
    S = np.cumsum(X - xbar[:, None], axis=1)
    Cn = (S * S).sum(axis=1) / n ** 2
    out["SN"] = n * xbar ** 2 / Cn
    if m_list:
        I2 = periodogram_fourier(X, max(m_list))
        for m in m_list:
            out[("HI", m)] = math.sqrt(n) * xbar / np.sqrt(I2[:, :m].mean(axis=1))
    if B_list:
        t = np.arange(1, n + 1)
        for B in B_list:
            Cm = np.sqrt(2.0 / n) * np.cos(np.pi * np.outer(np.arange(1, B + 1), t - 0.5) / n)
            Lam = X @ Cm.T
            out[("EWC", B)] = math.sqrt(n) * xbar / np.sqrt((Lam ** 2).mean(axis=1))
    return out
