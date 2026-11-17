"""Closed-form covariances of fBm functionals used by Paper 3.

Conventions (Type I fBm, Var B_H(1) = 1, 1/2 <= H < 1):
  h_0 = 1, h_k = sqrt(2) cos(k pi t);  G_kl = E[I_H(h_k) I_H(h_l)]  (Paper 1, Proposition 1; vendored, verified code).
  S_ij = <sin(i pi .), sin(j pi .)>_H, i, j >= 1 (unnormalised; closed form in the vendored Round 17 code).
  Pathwise bridge Bt(r) = B_H(r) - r B_H(1); sine basis s_k = sqrt(2) sin(k pi t):
      V_k := <Bt, s_k> = I_H(h_k)/(k pi)              (Theorem A'),  Cov(V_k, V_l) = G_kl/(k pi l pi),
      Cov(V_k, B_H(1)) = G_0k/(k pi),                  zero for odd k (parity),  G_kl = 0 for k + l odd.
  Hualde-Iacone fixed-m: C_j = int cos(2 pi j r) dB_H = I_H(h_2j)/sqrt2,  S_j = int sin(2 pi j r) dB_H;
      Cov(C_j, C_k) = G_{2j,2k}/2,  Cov(Z, C_j) = G_{0,2j}/sqrt2,  Cov(S_j, S_k) = S_{2j,2k},  Cov(C, S) = Cov(Z, S) = 0.
  EWC (Lazarus-Lewis-Stock-Watson 2018, eq. 10): Lambda_j = I_H(h_j), j = 1..B;  Cov(Z, Lambda) = G[0, 1..B].
"""
import math
import os
import sys

import numpy as np
from scipy.special import beta as _B

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(os.path.dirname(_HERE), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)
from covfourier.gram import CosineGram          # noqa: E402  verified closed cosine Gram
from covfourier.kummer import M as kummer_M      # noqa: E402  M_a(x) = 1F1(a; a+1; i x), double precision
import app4_kl_spectrum as _app4                 # noqa: E402  verified sine Gram and fBm Galerkin matrix

SQ2 = math.sqrt(2.0)


def c_H(H):
    return math.gamma(2.0 * H + 1.0) * math.sin(math.pi * H)


# ---------------------------------------------------------------- Gram matrices (double precision)
def cosine_gram(H, N):
    """G_kl, k, l = 0..N-1."""
    G = CosineGram(H, 1.0).dense(N)
    return 0.5 * (G + G.T)


def sine_gram(H, J):
    """S_ij = <sin(i pi .), sin(j pi .)>_H, i, j = 1..J (returned 0-based: S[i-1, j-1])."""
    return _app4.sine_gram(H, J)


def fbm_galerkin(H, J):
    """Verified cosine Galerkin matrix of the fBm covariance operator on L2(0,1) (J x J)."""
    A = _app4.galerkin_matrix(H, J)
    return 0.5 * (A + A.T)


# ---------------------------------------------------------------- exact traces and Hilbert-Schmidt norms
def trace_fbm(H):
    return 1.0 / (2.0 * H + 1.0)


def trace_bridge(H):
    """tr K_br = int_0^1 Var(B_H(t) - t B_H(1)) dt = 1/((2H+1)(H+1)) - 1/6."""
    return 1.0 / ((2.0 * H + 1.0) * (H + 1.0)) - 1.0 / 6.0


def trace_demeaned(H):
    """tr of the demeaned path B_H - int B_H: 1/((2H+1)(2H+2))."""
    return 1.0 / ((2.0 * H + 1.0) * (2.0 * H + 2.0))


def hs2_fbm(H):
    """||K||_HS^2 = [(4H+3)/(4H+1) - 2 B(2H+1, 2H+1)] / (4(2H+1))."""
    return ((4 * H + 3) / (4 * H + 1) - 2.0 * _B(2 * H + 1, 2 * H + 1)) / (4.0 * (2 * H + 1))


# ---------------------------------------------------------------- the bridge in the sine basis (Theorem A')
def bridge_blocks(H, J, G=None):
    """Parity blocks of Y = (Z, V_1..V_J), Z = B_H(1), V_k = <Bt, s_k>.

    Returns dict with
      A_even (Cov of V_k, k even), g (Cov(V_even, Z)), A_odd (Cov of V_k, k odd; independent of Z and of V_even),
      delta = tr K_br - sum_k Var V_k (exact remainder mean), cbar = c_H ((J+1) pi)^(-1-2H) (rigorous remainder scale).
    """
    if G is None:
        G = cosine_gram(H, J + 1)
    k = np.arange(1, J + 1)
    kp = k * np.pi
    ev = k[k % 2 == 0]
    od = k[k % 2 == 1]
    A_even = G[np.ix_(ev, ev)] / np.outer(ev * np.pi, ev * np.pi)
    A_odd = G[np.ix_(od, od)] / np.outer(od * np.pi, od * np.pi)
    g = G[0, ev] / (ev * np.pi)
    var_sum = float(np.sum(np.diag(G)[1:J + 1] / kp ** 2))
    delta = trace_bridge(H) - var_sum
    cbar = c_H(H) * ((J + 1) * np.pi) ** (-1.0 - 2.0 * H)
    return {"H": H, "J": J, "A_even": 0.5 * (A_even + A_even.T), "A_odd": 0.5 * (A_odd + A_odd.T), "g": g,
            "delta": delta, "cbar": cbar, "var_sum": var_sum, "G00": float(G[0, 0])}


# ---------------------------------------------------------------- limit covariances of the fixed-smoothing statistics
def hi_fixed_m_cov(H, m, G=None, S=None):
    """Covariance of (Z, C_1..C_m, S_1..S_m) for the Hualde-Iacone fixed-m limit (exact, finite-dimensional)."""
    if G is None:
        G = cosine_gram(H, 2 * m + 1)
    if S is None:
        S = sine_gram(H, 2 * m)
    idx = 2 * np.arange(1, m + 1)                      # cosine indices 2j
    Sig = np.zeros((2 * m + 1, 2 * m + 1))
    Sig[0, 0] = G[0, 0]
    Sig[0, 1:m + 1] = Sig[1:m + 1, 0] = G[0, idx] / SQ2
    Sig[1:m + 1, 1:m + 1] = G[np.ix_(idx, idx)] / 2.0
    Sig[m + 1:, m + 1:] = S[np.ix_(idx - 1, idx - 1)]
    return 0.5 * (Sig + Sig.T)


def ewc_cov(H, B, G=None):
    """Covariance of (Z, Lambda_1..Lambda_B) for the EWC (Type II DCT) limit: Paper 1's cosine Gram G_{0..B}."""
    if G is None:
        G = cosine_gram(H, B + 1)
    Sig = G[:B + 1, :B + 1].copy()
    return 0.5 * (Sig + Sig.T)


def ratio_form_eigs(Sig, c, scale):
    """Eigenvalues theta of the indefinite form X_c = Y_0^2 - c * scale * sum_{i>=1} Y_i^2, Y ~ N(0, Sig).

    P(stat^2 <= c) = P(X_c <= 0) where stat^2 = Y_0^2 / (scale * sum Y_i^2)^{-1}... i.e. stat^2 = Y_0^2 / (scale * sum Y_i^2).
    Returned sorted descending; exactly one is positive (Sylvester inertia)."""
    L = np.linalg.cholesky(Sig)
    d = np.full(Sig.shape[0], -c * scale)
    d[0] = 1.0
    th = np.linalg.eigvalsh(L.T @ (d[:, None] * L))
    return np.sort(th)[::-1]
