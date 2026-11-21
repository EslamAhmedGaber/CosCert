"""Exact finite-sample laws of studentised-mean statistics under Gaussian stationary models,
plus simulation, semiparametric memory estimators (LW, ELW), MAC (ADG 2009) and subsampling baselines.

Data x_t = mu + u_t, t = 1..n, u Gaussian with autocovariance gamma (fGn, ARFIMA(0,d,0), ARFIMA(1,d,0), AR(1)).
Under H0 every statistic below is a ratio of quadratic forms in u, so P(stat^2 <= c) = P(X_c <= 0) with
X_c = Y_0^2 - c * (PSD form): an exact finite-dimensional law (Proposition 6).

  SN / Lobato (2001) / Bartlett b = 1 (KVB):  T_n = n xbar^2 / C_n,  C_n = n^-2 sum_t S_t^2,  S_t = sum_{s<=t} (x_s - xbar);
      A = D (I - 11^T/n) (last row identically zero); K_tilde = n^-2 A Gamma A^T is (n-1)x(n-1) PSD, formed in O(n^2)
      without a Cholesky factorisation.  By the push-through identity (Proposition 6(b)), with v = 1^T Gamma 1 and
      b = n^-1 A Gamma 1, one eigendecomposition K_tilde = Q diag(kappa) Q^T serves every cutoff c:
          phi(s) = det(I + z K_tilde)^(-1/2) (1 - 2 i s u(z))^(-1/2),
          u(z) = 1 - z sum_k b2_k / (1 + z kappa_k),  b2_k = (q_k^T b)^2 / v,  z = 2 i s c n / v.
      Evaluated by adaptive Gauss-Kronrod quadrature (Imhof's integral) with a two-band head/power-sum split,
      and cross-checked by the tilted Bromwich midpoint rule (cdf_bromwich).
  Hualde-Iacone (2017) Daniell tau:  tau^2 <= c  <=>  (sum u)^2 <= (c/m) sum_j |sum_t u_t e^{i lambda_j t}|^2,  lambda_j = 2 pi j / n.
  EWC (Lazarus et al. 2018, eq. 10):  t^2 <= c  <=>  (sum u)^2 / n <= (c/B) sum_j Lambda_j^2,  Lambda_j = sqrt(2/n) sum_t cos(pi j (t - 1/2)/n) u_t.
"""
import math

import numpy as np
from scipy import integrate
from scipy.linalg import eigh, toeplitz, matmul_toeplitz
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


# ============================================================================ SN / Bartlett b = 1: Cholesky-free one-eigh law
class SNExact:
    """Exact finite-sample law of T_n via one eigendecomposition of K_tilde = n^-2 A Gamma A^T (no Cholesky)."""

    def __init__(self, gamma):
        n = gamma.size
        self.n = n
        G = toeplitz(gamma)
        s1 = G.sum(axis=1)                             # Gamma 1
        v = float(s1.sum())                            # 1^T Gamma 1
        b = np.cumsum(s1 - s1.mean())[:-1] / float(n)  # n^-1 A Gamma 1 (last entry is identically 0)
        # Apply A = D (I - 11^T/n) on the left and A^T on the right in O(n^2)
        G -= G.mean(axis=0, keepdims=True)
        np.cumsum(G, axis=0, out=G)
        AG = G[:-1, :]
        AG -= AG.mean(axis=1, keepdims=True)
        np.cumsum(AG, axis=1, out=AG)
        Kt = np.ascontiguousarray(AG[:, :-1])
        del G, AG
        Kt /= float(n) ** 2
        Kt = 0.5 * (Kt + Kt.T)
        kappa, Q = eigh(Kt, overwrite_a=True, check_finite=False, driver="evd")
        self.n_clipped = int(np.sum(kappa < 0))
        self.min_kappa_raw = float(kappa.min())
        kappa = np.clip(kappa, 0.0, None)
        b2 = (Q.T @ b) ** 2 / v                        # (q_k^T b)^2 / v
        del Q, Kt
        order = np.argsort(kappa)[::-1]
        self.kappa = kappa[order]
        self.b2 = b2[order]
        # Equivalent weights w2_k on nonzero modes for Bromwich path: w2_k = b2_k / kappa_k where kappa_k > 0
        # Note u(z) = 1 - z sum_k b2_k / (1 + z kappa_k) directly avoids dividing by small kappa_k!
        self.scale = n / v                             # c' = c * n / (1^T Gamma 1)

    def _band_prep(self, zmax):
        nh = int(np.sum(self.kappa * zmax > 0.05))
        kt, bt = self.kappa[nh:], self.b2[nh:]
        P = [float(np.sum(kt ** j)) for j in range(1, 9)]
        Bp = [float(np.sum(bt * (kt ** j))) for j in range(0, 9)]
        return nh, self.kappa[:nh], self.b2[:nh], P, Bp

    def _eval_band(self, z, band):
        nh, kh, bh, P, Bp = band
        if nh > 0:
            d = 1.0 + z * kh
            ld = complex(np.sum(np.log(d)))
            su = complex(np.sum(bh / d))
        else:
            ld = 0j
            su = 0j
        zp = z
        ld += zp * P[0]
        su += Bp[0] - zp * Bp[1]
        for j in range(2, 9):
            zp *= z
            if j & 1:
                ld += zp * (P[j - 1] / j)
                su -= zp * Bp[j]
            else:
                ld -= zp * (P[j - 1] / j)
                su += zp * Bp[j]
        return ld, 1.0 - z * su

    def cdf(self, c, epsabs=1e-13, epsrel=1e-12, limit=2000):
        """P(T_n <= c) via adaptive Gauss-Kronrod quadrature on the exact one-eigh characteristic function."""
        cp = c * self.scale
        kap = self.kappa
        b2 = self.b2
        # Two-band head/power-sum split (s <= 5 and s <= 80) accelerates small-s evaluations dramatically
        b1 = self._band_prep(10.0 * cp)
        b2p = self._band_prep(160.0 * cp)
        use_b1 = b1[0] < kap.size // 2
        use_b2 = b2p[0] < kap.size // 2

        def integrand(s):
            if s == 0.0:
                return 0.0
            z = 2j * s * cp
            if use_b1 and s <= 5.0:
                ld, u = self._eval_band(z, b1)
            elif use_b2 and s <= 80.0:
                ld, u = self._eval_band(z, b2p)
            else:
                d = 1.0 + z * kap
                ld = complex(np.sum(np.log(d)))
                u = 1.0 - z * complex(np.sum(b2 / d))
            r = 1.0 - 2j * s * u
            lphi = -0.5 * (ld + np.log(r))
            return math.sin(lphi.imag) * math.exp(lphi.real) / s

        val, _ = integrate.quad(integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return 0.5 - val / math.pi

    def _logdet_u_vec(self, z, band):
        nh, kh, bh, P, Bp = band
        if nh > 0:
            d = 1.0 + z[:, None] * kh[None, :]
            ld = np.log(d).sum(axis=1)
            su = (bh[None, :] / d).sum(axis=1)
        else:
            ld = np.zeros_like(z)
            su = np.zeros_like(z)
        zp = np.ones_like(z)
        for j in range(1, 9):
            zp = zp * z
            ld = ld + ((-1) ** (j + 1)) * zp * (P[j - 1] / j)
        zp = np.ones_like(z)
        for j in range(0, 9):
            su = su + ((-1) ** j) * zp * Bp[j]
            zp = zp * z
        return ld, 1.0 - z * su

    def theta_plus(self, cp):
        kap = self.kappa
        b2 = self.b2

        def h(th):
            z = cp / th
            u = 1.0 - z * float(np.sum(b2 / (1.0 + z * kap)))
            return u - th

        if h(1.0) >= 0:
            return 1.0
        return brentq(h, 1e-10, 1.0, xtol=1e-15, rtol=1e-15)

    def cdf_bromwich(self, c, theta=0.25, eps=1e-13):
        """P(T_n <= c) via tilted Bromwich midpoint sum (independent cross-check of cdf)."""
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
        band = self._band_prep(float(np.max(np.abs(z))))
        ld, u = self._logdet_u_vec(z, band)
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


# ============================================================================ simulation, memory estimators & competitors
class CirculantSampler:
    def __init__(self, gamma):
        n = gamma.size
        row = np.concatenate([gamma, gamma[-2:0:-1]])
        lam = np.fft.fft(row).real
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


def local_whittle(X, m, grid=np.linspace(-0.49, 0.49, 197), return_G=False):
    """Robinson (1995) local Whittle estimate of d for each row (grid search + interior parabolic refinement).
    Boundary minima (j = 0 or j = G - 1) are kept at the boundary rather than snapped inward."""
    n = X.shape[1]
    I = periodogram_fourier(X, m)
    lam = 2 * math.pi * np.arange(1, m + 1) / n
    llam = np.log(lam)
    mean_llam = llam.mean()
    R = np.empty((X.shape[0], grid.size))
    for i, d in enumerate(grid):
        R[:, i] = np.log(np.mean(I * lam[None, :] ** (2 * d), axis=1)) - 2 * d * mean_llam
    j_raw = np.argmin(R, axis=1)
    interior = (j_raw > 0) & (j_raw < grid.size - 1)
    jc = np.clip(j_raw, 1, grid.size - 2)
    idx = np.arange(X.shape[0])
    r0, r1, r2 = R[idx, jc - 1], R[idx, jc], R[idx, jc + 1]
    h = grid[1] - grid[0]
    den = r0 - 2 * r1 + r2
    off = np.where(interior & (den > 0), 0.5 * (r0 - r2) / np.where(den > 0, den, 1.0), 0.0)
    dhat = np.clip(grid[j_raw] + off * h, grid[0], grid[-1])
    if not return_G:
        return dhat
    # 2 pi G_hat = mean_j (lam_j^(2 dhat) * 2 pi I(lam_j))
    twopi_G = np.mean(I * (lam[None, :] ** (2.0 * dhat[:, None])), axis=1)
    return dhat, twopi_G


def exact_local_whittle(X, m, grid=np.linspace(-0.49, 0.49, 99)):
    """Demeaned (feasible) Exact Local Whittle estimator (Shimotsu & Phillips 2005; Shimotsu 2010) for each row of X.
    Evaluates (1 - L)^d (x_t - xbar) via FFT convolution on the grid, then applies interior parabolic refinement."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=1, keepdims=True)
    B, n = Xc.shape
    lam = 2 * math.pi * np.arange(1, m + 1) / n
    mean_llam = float(np.mean(np.log(lam)))
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    FX = np.fft.rfft(Xc, n=nfft, axis=1)
    # Precompute fractional filter FFTs for all grid points once
    R = np.empty((B, grid.size))
    idx_k = np.arange(1, n, dtype=float)
    for i, d in enumerate(grid):
        pi = np.empty(n)
        pi[0] = 1.0
        # pi_k = prod_{j=1}^k (j - 1 - d) / j
        pi[1:] = np.cumprod((idx_k - 1.0 - d) / idx_k)
        fpi = np.fft.rfft(pi, n=nfft)
        dX = np.fft.irfft(FX * fpi[None, :], n=nfft, axis=1)[:, :n]
        I = (np.abs(np.fft.rfft(dX, axis=1)[:, 1:m + 1]) ** 2) / n
        R[:, i] = np.log(np.mean(I, axis=1)) - 2 * d * mean_llam
    j_raw = np.argmin(R, axis=1)
    interior = (j_raw > 0) & (j_raw < grid.size - 1)
    jc = np.clip(j_raw, 1, grid.size - 2)
    rows = np.arange(B)
    r0, r1, r2 = R[rows, jc - 1], R[rows, jc], R[rows, jc + 1]
    h = grid[1] - grid[0]
    den = r0 - 2 * r1 + r2
    off = np.where(interior & (den > 0), 0.5 * (r0 - r2) / np.where(den > 0, den, 1.0), 0.0)
    return np.clip(grid[j_raw] + off * h, grid[0], grid[-1])


def adg_mac_stat(X, m, dhat=None, twopi_G=None, clip_d=(-0.49, 0.49)):
    """Memory- and autocorrelation-consistent (MAC) studentised mean statistic of Abadir, Distaso & Giraitis (2009):
        t_MAC = n^(1/2 - dhat) * xbar / sqrt(V_MAC),
        V_MAC = (2 pi G_hat) * Gamma(1 - 2 dhat) / ((2 dhat + 1) Gamma(1 + dhat) Gamma(1 - dhat)),
    which has an asymptotic N(0, 1) distribution for d in (-1/2, 1/2)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    if dhat is None or twopi_G is None:
        dhat, twopi_G = local_whittle(X, m, return_G=True)
    d = np.clip(np.asarray(dhat, dtype=float), clip_d[0], clip_d[1])
    log_cf = gammaln(1.0 - 2.0 * d) - np.log(2.0 * d + 1.0) - gammaln(1.0 + d) - gammaln(1.0 - d)
    V_mac = np.maximum(twopi_G * np.exp(log_cf), 1e-300)
    xbar = X.mean(axis=1)
    return (n ** (0.5 - d)) * xbar / np.sqrt(V_mac)


def subsampling_sn_pvalue(X, b=None):
    """Rate-free self-normalised subsampling (Politis, Romano & Wolf 1999; Shao 2010):
    computes the empirical upper-tail p-value of T_n against the (n - b + 1) overlapping block statistics T_b^{(i)}
    centred at the full-sample mean xbar, so no memory parameter H or convergence rate needs to be estimated."""
    X = np.asarray(X, dtype=float)
    B, n = X.shape
    if b is None:
        b = max(12, int(round(n ** 0.5)))
    xbar = X.mean(axis=1)
    S_full = np.cumsum(X - xbar[:, None], axis=1)
    Tn = n * (xbar ** 2) / ((S_full * S_full).sum(axis=1) / (n ** 2))
    # Overlapping blocks of length b, centred at xbar
    nblk = n - b + 1
    Xc = X - xbar[:, None]
    # Strided view (B, nblk, b)
    s0, s1 = Xc.strides
    W = np.lib.stride_tricks.as_strided(Xc, shape=(B, nblk, b), strides=(s0, s1, s1))
    wbar = W.mean(axis=2)                                       # (B, nblk)
    Sw = np.cumsum(W - wbar[:, :, None], axis=2)
    Cb = (Sw * Sw).sum(axis=2) / (b ** 2)
    Tb = b * (wbar ** 2) / np.maximum(Cb, 1e-300)
    return np.mean(Tb >= Tn[:, None], axis=1)


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
