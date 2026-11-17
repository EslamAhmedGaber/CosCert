"""Certified distribution function of T_H = Z^2 / int_0^1 Btilde(r)^2 dr  (self-normalised / Bartlett b = 1 limit).

Chain of the certificate (manuscript, Section 'Certified critical values'; every step encloses its error):
  (1) exact representation  int Btilde^2 = sum_k V_k^2,  V_k = I_H(h_k)/(k pi),  h_k = sqrt2 cos(k pi .)   (Theorem A'),
      Cov(V_k, V_l) = G_kl/(k pi l pi),  Cov(V_k, Z) = G_0k/(k pi)  with the closed cosine Gram G of Paper 1;
      parity: odd modes are independent of (Z, even modes);
  (2) truncation W_J = sum_{k<=J} V_k^2;  R_J = int Btilde^2 - W_J >= 0 has mean tau_J (exact: tr K_br - sum Var V_k) and
      Laurent-Massart tails with variance proxy v <= mu_max tau_J,  mu_max <= min(tau_J, c_H ((J+1) pi)^(-1-2H))  (Theorem A');
  (3) eigenvector-free characteristic function of X_J = Z^2 - c W_J,  z = 2 i s c:
         phi(s) = det(I + z A_O)^(-1/2) det(I + z A_E)^(-1/2) u^(-1/2) (1/u - 2 i s)^(-1/2),   u = 1 - z g^T (I + z A_E)^(-1) g,
      every factor with a positive real part (principal logarithms are continuous);
  (4) head-tail Schur complement: log det(I + z A) = log det(I + z A_hh) + log det(I + z S),  S = A_tt - z A_th (I + z A_hh)^(-1) A_ht;
      the head block is handled through an exact similarity Q^(-1) A_hh Q = diag(lam) + F computed in ball arithmetic, the tail through
      tr A_tt, ||A_tt||_F^2 and the couplings W = A_th^T A_th, with explicit balls for every neglected term;
  (5) Gil-Pelaez midpoint sum (Davies 1973): the infinite sum S_Delta(x) satisfies F(x) - S_Delta(x) in [-P(X > x + L), P(X <= x - L)],
      L = 2 pi / Delta, with P(X > y) <= P(Z^2 > y) and a Chernoff bound for P(c W_J >= y); series truncation bounded through certified
      lower bounds of the eigenvalues (Cauchy interlacing, Kahan's residual theorem).
All floating-point inputs to step (4) carry entrywise rounding bounds (standard model, u = 2^-53); all other arithmetic is in arb.
"""
import math
import time
from fractions import Fraction

import numpy as np
from flint import arb, acb, arb_mat, ctx

U = 2.0 ** -53
INF = float("inf")


def gam(n):
    return n * U / (1.0 - n * U)


def fup(x):
    """Float upper bound of an arb."""
    return math.nextafter(float(x.upper()), INF)


def flo(x):
    return math.nextafter(float(x.lower()), -INF)


def disk(r):
    """Complex ball containing the closed disk of radius r (float, rounded up)."""
    r = math.nextafter(float(r), INF)
    return acb(arb(0, r), arb(0, r))


def exact_H(H):
    fr = H if isinstance(H, Fraction) else Fraction(str(H))
    return fr, arb(fr.numerator) / fr.denominator


def c_H_arb(Ha):
    return (2 * Ha + 1).gamma() * (arb.pi() * Ha).sin()


def trace_bridge_arb(Ha):
    return 1 / ((2 * Ha + 1) * (Ha + 1)) - arb(1) / 6


# ============================================================================ closed cosine Gram in ball arithmetic
class Kummer:
    """p_k + i r_k = 1F1(b; b + 1; i k pi), b = 2H - 1, k = 0..kmax, in arb; double copies with error bounds."""

    def __init__(self, H, kmax, prec=128):
        ctx.prec = prec
        self.prec = prec
        self.fr, self.Ha = exact_H(H)
        Ha = self.Ha
        self.b = 2 * Ha - 1
        self.C = 2 * Ha
        self.pi = arb.pi()
        self.kmax = kmax
        P, R = [arb(1)], [arb(0)]
        b = self.b
        for k in range(1, kmax + 1):
            f = acb(0, k * self.pi).hypgeom_1f1(b, b + 1)
            P.append(f.real)
            R.append(f.imag)
        self.P, self.R = P, R
        self.r = np.array([float(x.mid()) for x in R])
        self.er = np.array([float(x.rad()) for x in R]) * 1.01 + 1.2e-16 * np.abs(self.r) + 1e-300
        kap = 2 * self.C / self.pi
        self.kap = float(kap.mid())
        self.dkap = (float(kap.rad()) + abs(float(kap.mid()) - self.kap)) / abs(self.kap) + 1.2e-16
        pi2 = self.pi * self.pi
        self.pi2 = float(pi2.mid())
        self.dpi2 = 2.3e-16
        # exact diagonal A_kk = G_kk / (k pi)^2 and g_k = G_0k / (k pi) in arb
        self.Ad_arb = [arb(0)] + [self.C * (P[k] - (1 - b) * R[k] / (k * self.pi)) / (k * self.pi) ** 2 for k in range(1, kmax + 1)]
        self.Ad = np.array([float(x.mid()) for x in self.Ad_arb])
        self.Ad_err = np.array([float(x.rad()) for x in self.Ad_arb]) * 1.01 + 1.2e-16 * np.abs(self.Ad) + 1e-300
        sq2 = arb(2).sqrt()
        self.g_arb = [arb(0)] * (kmax + 1)
        for k in range(2, kmax + 1, 2):
            self.g_arb[k] = -sq2 * self.C * R[k] / (pi2 * k * k)

    # ----- exact entries
    def A_arb(self, k, l):
        if k == l:
            return self.Ad_arb[k]
        if (k + l) % 2:
            return arb(0)
        G = 2 * self.C / self.pi * (l * self.R[l] - k * self.R[k]) / (k * k - l * l)
        return G / (k * l * self.pi * self.pi)

    # ----- double blocks with entrywise rounding bounds
    def block(self, K, L):
        K = np.asarray(K, dtype=np.int64)
        L = np.asarray(L, dtype=np.int64)
        assert np.all((K[:, None] + L[None, :]) % 2 == 0)
        Kf = K.astype(float)[:, None]
        Lf = L.astype(float)[None, :]
        t1 = Lf * self.r[L][None, :]
        t2 = Kf * self.r[K][:, None]
        num = t1 - t2
        den = Kf * Kf - Lf * Lf
        dg = den == 0
        den1 = np.where(dg, 1.0, den)
        q = num / den1
        G = self.kap * q
        KL = Kf * Lf
        A = G / (self.pi2 * KL)
        errnum = Lf * self.er[L][None, :] + Kf * self.er[K][:, None] + 2 * U * (np.abs(t1) + np.abs(t2))
        errG = abs(self.kap) * (errnum / np.abs(den1) + U * np.abs(q)) + (U + self.dkap) * np.abs(G)
        rho = (errG / (self.pi2 * KL) + (2 * U + self.dpi2) * np.abs(A)) * 1.001
        if dg.any():
            ii, jj = np.nonzero(dg)
            A[ii, jj] = self.Ad[K[ii]]
            rho[ii, jj] = self.Ad_err[K[ii]]
        return A, rho


def arb_from(mid, rad=None):
    if rad is None:
        return arb(float(mid))
    return arb(float(mid), math.nextafter(float(rad), INF))


def frob_up(M):
    """Upper bound of the Frobenius norm of an arb_mat."""
    s = arb(0)
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            a = M[i, j].abs_upper()
            s += a * a
    return fup(s.sqrt())


def vec_norm_up(v):
    s = arb(0)
    for x in v:
        a = x.abs_upper()
        s += a * a
    return fup(s.sqrt())


def gersh_sorted(T):
    """Certified sorted eigenvalue bounds of a real matrix T that is similar to a symmetric matrix (real spectrum).
    Gershgorin discs (centre T_ii, radius sum_{j != i} |T_ij|) in ball arithmetic; a connected union of k discs contains
    exactly k eigenvalues (continuity argument), so every member of a cluster gets the cluster hull and the sorted
    endpoints pair correctly with the sorted eigenvalues.  Returns (lo, hi), both sorted descending (floats, outward)."""
    n = T.nrows()
    lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        r = arb(0)
        for j in range(n):
            if j != i:
                r += T[i, j].abs_upper()
        lo[i] = flo(T[i, i] - r); hi[i] = fup(T[i, i] + r)
    order = np.argsort(lo)
    cl_lo, cl_hi = lo.copy(), hi.copy()
    groups, cur = [], [order[0]]
    for a, b_ in zip(order[:-1], order[1:]):
        if lo[b_] <= max(hi[i] for i in cur):
            cur.append(b_)
        else:
            groups.append(cur); cur = [b_]
    groups.append(cur)
    for g in groups:
        L_, H_ = min(lo[i] for i in g), max(hi[i] for i in g)
        for i in g:
            cl_lo[i], cl_hi[i] = L_, H_
    return np.sort(cl_lo)[::-1], np.sort(cl_hi)[::-1]


def _sumsq_block(km, rows, cols, chunk_entries=2_000_000):
    """sum A_kl^2 over rows x cols (same parity) with a rigorous error bound: returns (S, err)."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.size == 0 or cols.size == 0:
        return 0.0, 0.0
    step = max(1, chunk_entries // cols.size)
    parts, E, nsum = [], 0.0, 0
    for i0 in range(0, rows.size, step):
        A, rho = km.block(rows[i0:i0 + step], cols)
        parts.append(float(np.sum(A * A)))
        E += float(np.sum(2 * np.abs(A) * rho + rho * rho))
        nsum = max(nsum, A.size)
    S = math.fsum(parts)
    return S, gam(nsum + len(parts)) * S + E * (1 + 1e-6)


def tail_frobenius(km, tl, chunk_entries=2_000_000):
    return _sumsq_block(km, tl, tl, chunk_entries)


def nested_tail_frobenius(km, modes, heads, chunk_entries=2_000_000):
    """||A_tt(h)||_F^2 for the tails modes[h:], h in heads (ascending): direct for the largest head, strips for the others."""
    heads = sorted(heads)
    out = {}
    hmax = heads[-1]
    S, err = _sumsq_block(km, modes[hmax:], modes[hmax:], chunk_entries)
    out[hmax] = (S, err)
    for a, b in zip(heads[-2::-1], heads[:0:-1]):
        S1, e1 = _sumsq_block(km, modes[a:b], modes[b:], chunk_entries)
        S2, e2 = _sumsq_block(km, modes[a:b], modes[a:b], chunk_entries)
        Sb, eb = out[b]
        Sa = math.fsum([Sb, 2 * S1, S2])
        out[a] = (Sa, (eb + 2 * e1 + e2) * (1 + 1e-12) + 4 * U * Sa)
    return out


# ============================================================================ one parity block: head-tail data
class Block:
    def __init__(self, km, modes, nh, with_g=False, chunk_entries=2_000_000, verbose=False, tail_fro=None):
        t0 = time.perf_counter()
        modes = np.asarray(modes, dtype=np.int64)
        nh = min(nh, modes.size)
        self.nh = nh
        self.modes = modes
        hd, tl = modes[:nh], modes[nh:]
        self.nt = tl.size
        self.first_tail = int(tl[0]) if tl.size else None
        # ---- head block in arb, similarity transform, Kahan bound
        n = nh
        Ahh = arb_mat(n, n)
        for i in range(n):
            for j in range(i, n):
                a = km.A_arb(int(hd[i]), int(hd[j]))
                Ahh[i, j] = a
                Ahh[j, i] = a
        Amid = np.array([[float(Ahh[i, j].mid()) for j in range(n)] for i in range(n)])
        lam, Q = np.linalg.eigh(Amid)
        order = np.argsort(lam)[::-1]
        lam, Q = lam[order], Q[:, order]
        self.lam = lam
        Qa = arb_mat(n, n, Q.ravel().tolist())
        Qi = Qa.inv()
        D = arb_mat(n, n)
        for i in range(n):
            D[i, i] = arb(float(lam[i]))
        F = Qi * Ahh * Qa - D
        self.nF = frob_up(F)
        Rr = Ahh * Qa - Qa * D
        self.nQ = frob_up(Qa)
        self.nQi = frob_up(Qi)
        self.eta = math.nextafter(frob_up(Rr) * self.nQi, INF)       # Bauer-Fike radius (reported only)
        # certified sorted eigenvalue bounds of A_hh: Gershgorin clusters of D + F = Q^{-1} A_hh Q (exact similarity, balls)
        self.eig_lo, self.eig_hi = gersh_sorted(F + D)
        self.fro2_hh = fup(sum((Ahh[i, j] * Ahh[i, j] for i in range(n) for j in range(n)), arb(0)))
        t1 = time.perf_counter()
        # ---- tail: trace (arb), Frobenius norm (double with bounds), coupling W = A_th^T A_th
        self.trAtt = sum((km.Ad_arb[int(k)] for k in tl), arb(0))
        S = 0.0
        E = 0.0
        nsum = 0
        if self.nt:
            if tail_fro is None:
                S, err = tail_frobenius(km, tl, chunk_entries)
            else:
                S, err = tail_fro
            self.fro2Att = arb(S, math.nextafter(err, INF))
            fro_up = math.sqrt(S + err) * (1 + 1e-15)
            # coupling
            Ath, rth = km.block(tl, hd)
            Wd = Ath.T @ Ath
            aA = np.abs(Ath)
            Wb = (gam(self.nt) * (aA.T @ aA) + aA.T @ rth + rth.T @ aA + rth.T @ rth) * (1 + 2 * gam(self.nt)) + 1e-300
            self.trW = (float(np.sum(Ath * Ath)) * (1 + gam(Ath.size)) + float(np.sum(2 * aA * rth + rth * rth))) * (1 + 1e-6)
            Wa = arb_mat(n, n)
            for i in range(n):
                for j in range(n):
                    Wa[i, j] = arb_from(Wd[i, j], Wb[i, j])
            Wt = Qi * Wa * Qa
            self.w = [Wt[j, j] for j in range(n)]
            # operator norm of A_tt: min(Frobenius, Theorem A' bound c_H (L pi)^(-1-2H), L = first tail mode)
            thA = c_H_arb(km.Ha) * (self.first_tail * km.pi) ** (-1 - 2 * km.Ha)
            self.nAtt_thm = fup(thA)
            self.nAtt_fro = fro_up
            self.nAtt = min(fro_up, self.nAtt_thm)
            self.fAtt = fro_up
        else:
            self.fro2Att = arb(0)
            self.trW = 0.0
            self.w = [arb(0)] * n
            self.nAtt = self.fAtt = self.nAtt_thm = self.nAtt_fro = 0.0
            Ath = rth = None
        t2 = time.perf_counter()
        # ---- even block: correlation with Z
        self.with_g = with_g
        if with_g:
            gh = arb_mat(n, 1, [km.g_arb[int(k)] for k in hd])
            a = Qa.transpose() * gh
            bq = Qi * gh
            self.zeta = [a[j, 0] * bq[j, 0] for j in range(n)]
            self.ngh = vec_norm_up([km.g_arb[int(k)] for k in hd])
            self.nQtg = vec_norm_up([a[j, 0] for j in range(n)])
            self.nQig = vec_norm_up([bq[j, 0] for j in range(n)])
            if self.nt:
                gt_arb = [km.g_arb[int(k)] for k in tl]
                self.gt2 = sum((x * x for x in gt_arb), arb(0))
                self.ngt = fup(self.gt2.sqrt())
                gt = np.array([float(x.mid()) for x in gt_arb])
                egt = np.array([float(x.rad()) for x in gt_arb]) * 1.01 + 1.2e-16 * np.abs(gt) + 1e-300
                vd = Ath.T @ gt
                vb = (gam(self.nt) * (aA.T @ np.abs(gt)) + rth.T @ np.abs(gt) + aA.T @ egt + rth.T @ egt) * (1 + 2 * gam(self.nt)) + 1e-300
                va = arb_mat(n, 1, [arb_from(vd[j], vb[j]) for j in range(n)])
                qv = Qa.transpose() * va
                self.xi = [qv[j, 0] * bq[j, 0] for j in range(n)]
                self.nQtv = vec_norm_up([qv[j, 0] for j in range(n)])
            else:
                self.gt2 = arb(0)
                self.ngt = 0.0
                self.xi = [arb(0)] * n
                self.nQtv = 0.0
        self.trA = sum((km.Ad_arb[int(k)] for k in modes), arb(0))
        self.fro2_up = self.fro2_hh + 2 * self.trW + (fup(self.fro2Att) if self.nt else 0.0)
        self.lmax_up = fup(arb(max(float(self.eig_hi[0]), self.nAtt)) + arb(self.trW).sqrt())
        self.timing = (t1 - t0, t2 - t1, time.perf_counter() - t2)
        # double copies for the fast (computed) path
        self.w_d = np.array([float(x.mid()) for x in self.w])
        if with_g:
            self.zeta_d = np.array([float(x.mid()) for x in self.zeta])
            self.xi_d = np.array([float(x.mid()) for x in self.xi])
            self.gt2_d = float(self.gt2.mid())
        self.trAtt_d = float(self.trAtt.mid())
        self.fro2Att_d = float(self.fro2Att.mid())
        if verbose:
            print(f"  block nh={n} nt={self.nt}: head {self.timing[0]:.1f}s tail {self.timing[1]:.1f}s g {self.timing[2]:.1f}s; "
                  f"nF={self.nF:.2e} eta={self.eta:.2e} trW={self.trW:.2e} nAtt={self.nAtt:.2e} (thm {self.nAtt_thm:.2e}, fro {self.nAtt_fro:.2e})")

    # ----- ball evaluation of log det(I + zA) (and gamma for the even block)
    def terms(self, z, az):
        n = self.nh
        xF = az * self.nF
        if not xF < 0.5:
            raise ValueError(f"head similarity residual too large: |z| ||F|| = {xF:.3e}")
        nD = xF / (1 - xF)
        s2 = self.nAtt + az * self.trW
        if not az * s2 < 0.5:
            raise ValueError(f"tail not small: |z| ||S|| = {az * s2:.3e}")
        sF = self.fAtt + az * self.trW
        LH = acb(0)
        sw = acb(0)
        sz = acb(0)
        sx = acb(0)
        for j in range(n):
            d = 1 + z * float(self.lam[j])
            inv = 1 / d
            LH += d.log()
            sw += self.w[j] * inv
            if self.with_g:
                sz += self.zeta[j] * inv
                sx += self.xi[j] * inv
        LH += disk(n * xF / (1 - xF))
        sw += disk(nD * self.nQ * self.nQi * self.trW)
        trS = self.trAtt - z * sw
        E2 = az ** 2 / 2 * (2 * az * self.nAtt * self.trW + az ** 2 * self.trW ** 2)
        E3 = az ** 3 * s2 * sF ** 2 / (3 * (1 - az * s2))
        L = LH + z * trS - z * z / 2 * self.fro2Att + disk((E2 + E3) * (1 + 1e-12))
        if not self.with_g:
            return L, None
        sz += disk(nD * self.nQtg * self.nQig)
        sx += disk(nD * self.nQtv * self.nQig)
        ete = self.gt2 - 2 * z * sx + disk(az ** 2 * self.trW * self.ngh ** 2 * (1 + 1e-12))
        e2 = (self.ngt + az * math.sqrt(self.trW) * self.ngh) ** 2
        gamma = sz + ete + disk(az * s2 * e2 / (1 - az * s2) * (1 + 1e-12))
        return L, gamma

    # ----- float estimate of the ball radii produced by terms() (used only to choose the head size)
    def rad_estimate(self, az):
        xF = az * self.nF
        s2 = self.nAtt + az * self.trW
        if not (xF < 0.25 and az * s2 < 0.25):
            return INF, INF
        nD = xF / (1 - xF)
        sF = self.fAtt + az * self.trW
        E2 = az ** 2 / 2 * (2 * az * self.nAtt * self.trW + az ** 2 * self.trW ** 2)
        E3 = az ** 3 * s2 * sF ** 2 / (3 * (1 - az * s2))
        rL = self.nh * xF / (1 - xF) + az * nD * self.nQ * self.nQi * self.trW + E2 + E3
        if not self.with_g:
            return rL, 0.0
        e2 = (self.ngt + az * math.sqrt(self.trW) * self.ngh) ** 2
        rg = (nD * self.nQtg * self.nQig + 2 * az * nD * self.nQtv * self.nQig + az ** 2 * self.trW * self.ngh ** 2
              + az * s2 * e2 / (1 - az * s2))
        return rL, rg

    # ----- double-precision mirror (computed), vectorised over z
    def terms_d(self, z):
        d = 1.0 + z[:, None] * self.lam[None, :]
        inv = 1.0 / d
        LH = np.log(d).sum(axis=1)
        trS = self.trAtt_d - z * (inv @ self.w_d)
        L = LH + z * trS - z * z / 2 * self.fro2Att_d
        if not self.with_g:
            return L, None
        gamma = inv @ self.zeta_d + self.gt2_d - 2 * z * (inv @ self.xi_d)
        return L, gamma


# ============================================================================ the full system for given (H, J)
class BridgeSystem:
    def __init__(self, H, J, heads=(8, 16, 32, 64, 128), prec=128, km=None, verbose=False, nh_odd=None, nh_even=None):
        assert J % 2 == 0
        if nh_odd is not None:
            heads = (nh_odd,)
        heads = tuple(sorted(h for h in heads if h <= J // 2))
        t0 = time.perf_counter()
        self.H = H
        self.J = J
        self.heads = heads
        self.km = km if km is not None and km.kmax >= J and km.fr == exact_H(H)[0] else Kummer(H, J, prec)
        km = self.km
        t1 = time.perf_counter()
        mo = np.arange(1, J, 2)
        me = np.arange(2, J + 1, 2)
        fo = nested_tail_frobenius(km, mo, heads)
        fe = nested_tail_frobenius(km, me, heads)
        self.odd_blocks = [Block(km, mo, h, with_g=False, tail_fro=fo[h], verbose=verbose) for h in heads]
        self.even_blocks = [Block(km, me, h, with_g=True, tail_fro=fe[h], verbose=verbose) for h in heads]
        self.odd, self.even = self.odd_blocks[-1], self.even_blocks[-1]
        Ha = km.Ha
        self.trK = trace_bridge_arb(Ha)
        self.trA = self.odd.trA + self.even.trA
        self.tau = self.trK - self.trA                                   # E R_J (exact)
        self.cbar = c_H_arb(Ha) * ((J + 1) * km.pi) ** (-1 - 2 * Ha)     # Theorem A'
        self.fro2_up = self.odd.fro2_up + self.even.fro2_up
        self.lmax_up = max(self.odd.lmax_up, self.even.lmax_up)
        self.trA_up = fup(self.trA)
        # certified lower bounds of eigenvalues for the truncation bound: Gershgorin clusters of the head blocks (sorted),
        # Cauchy interlacing head -> full block; the even block carries Z (rank-one update), so its largest is dropped
        nu_o = self.odd.eig_lo
        nu_e = self.even.eig_lo[1:]
        nu = np.concatenate([nu_o, nu_e])
        self.nu = np.sort(nu[nu > 0])[::-1]
        self.timing = {"kummer": t1 - t0, "blocks": time.perf_counter() - t1}
        if verbose:
            print(f"BridgeSystem H={H} J={J}: kummer {t1 - t0:.1f}s, blocks {time.perf_counter() - t1:.1f}s, "
                  f"tau={float(self.tau.mid()):.4e}, cbar={float(self.cbar.mid()):.4e}")

    # ---------------- remainder bracket parameters (Laurent-Massart)
    def remainder(self, t=25.0):
        tau = self.tau
        mu = tau if tau < self.cbar else self.cbar
        v = mu * tau
        ta = arb(t)
        ep = 2 * (v * ta).sqrt() + 2 * mu * ta
        em = 2 * (v * ta).sqrt()
        lo = tau - em
        lo_pos = lo > 0
        return {"tau": tau, "mu": mu, "v": v, "s_hi": fup(tau + ep), "s_lo": (flo(lo) if lo_pos else 0.0),
                "p_fail": math.exp(-t), "lo_uses_tail": bool(lo_pos)}

    # ---------------- tail bounds for the aliasing error
    def lower_tail_bound(self, c, y):
        """P(c W_J >= y) <= min_theta exp(-theta y / c + theta tr A + theta^2 ||A||_F^2 / (1 - 2 theta lmax))."""
        if y <= 0:
            return 1.0
        best = 1.0
        thmax = 0.5 / self.lmax_up
        for f in np.linspace(0.02, 0.98, 49):
            th = f * thmax
            e = -th * y / c + th * self.trA_up + th * th * self.fro2_up / (1 - 2 * th * self.lmax_up)
            best = min(best, math.exp(e))
        return best * (1 + 1e-9)

    def choose_L(self, c, xs, eps=1e-18):
        L = 10.0
        while True:
            ok = True
            for x in xs:
                up = math.erfc(math.sqrt(max(x + L, 0) / 2))
                lo = self.lower_tail_bound(c, L - x)
                if up > eps or lo > eps:
                    ok = False
                    break
            if ok:
                return L
            L *= 1.1

    def trunc_bound(self, c, uK):
        a = 2 * uK * c * self.nu
        sel = a[a > 1.0]
        m = sel.size
        if m == 0:
            return INF
        return 2.0 / (math.pi * m) * math.exp(-0.5 * np.sum(np.log(sel)))

    def choose_K(self, c, Delta, eps=1e-16, kmax=200000):
        # terms k = 0..K-1 are summed; the omitted tail is bounded by (1/pi)(2/m) prod (2 u c nu)^(-1/2) at u = (K - 1/2) Delta
        K = 64
        while K < kmax:
            if self.trunc_bound(c, (K - 0.5) * Delta) <= eps:
                return K
            K = int(K * 1.15) + 1
        raise RuntimeError("series truncation did not converge")

    # ---------------- computed (double) CDF of X_J at several x, vectorised
    def cdf_X_double(self, c, xs, eps_alias=1e-18, eps_trunc=1e-16):
        L = self.choose_L(c, xs, eps_alias)
        Delta = 2 * math.pi / L
        K = self.choose_K(c, Delta, eps_trunc)
        u = (np.arange(K) + 0.5) * Delta
        z = 2j * u * c
        LO, _ = self.odd.terms_d(z)
        LE, gm = self.even.terms_d(z)
        w = 1.0 - z * gm
        logphi = -0.5 * (LO + LE) - 0.5 * np.log(w) - 0.5 * np.log(1.0 / w - 2j * u)
        out = []
        for x in xs:
            terms = np.imag(np.exp(logphi - 1j * u * x)) / (np.arange(K) + 0.5)
            out.append(0.5 - math.fsum(terms.tolist()) / math.pi)
        return out, {"L": L, "Delta": Delta, "K": K}

    def cdf_T_computed(self, c, t=25.0):
        """Computed P(T_H <= c) with the remainder replaced by its mean (point value) and the certified-style bracket
        evaluated in double precision (for root finding and exploration; not a certificate)."""
        rem = self.remainder(t)
        tau = float(rem["tau"].mid())
        xs = [c * rem["s_lo"], c * tau, c * rem["s_hi"]]
        F, info = self.cdf_X_double(c, xs)
        lo = F[0] - (rem["p_fail"] if rem["lo_uses_tail"] else 0.0)
        hi = F[2] + rem["p_fail"]
        return {"point": F[1], "lo": lo, "hi": hi, **info}

    # ---------------- certified CDF of X_J (ball arithmetic)
    def cdf_X_certified(self, c, xs, eps_alias=1e-18, eps_trunc=1e-16):
        c_a = arb(float(c))
        L = self.choose_L(c, xs, eps_alias)
        Delta = 2 * math.pi / L
        K = self.choose_K(c, Delta, eps_trunc)
        D_a = arb(Delta)
        # validity of the Schur expansions on the whole range s <= u_K
        uK = (K + 0.5) * Delta
        azmax = 2 * uK * c * (1 + 1e-12)
        for blk in (self.odd, self.even):
            if not azmax * blk.nF < 0.5 or not azmax * (blk.nAtt + azmax * blk.trW) < 0.5:
                raise ValueError("head too small for the required frequency range")
        sums = [acb(0) for _ in xs]
        xs_a = [arb(float(x)) for x in xs]
        for k in range(K):
            s = (k + arb(0.5)) * D_a
            z = acb(0, 2 * s * c_a)
            az = fup(2 * s * c_a)
            LO, _ = self.odd.terms(z, az)
            LE, gm = self.even.terms(z, az)
            w = 1 - z * gm
            if not w.real > 0 or not (1 / w).real > 0:
                raise ValueError("branch check failed: Re(1 - z gamma) not positive")
            logphi = -(LO + LE) / 2 - w.log() / 2 - (1 / w - acb(0, 2 * s)).log() / 2
            ph = logphi.exp()
            for i, xa in enumerate(xs_a):
                e = acb(0, -s * xa).exp() if xs[i] != 0 else acb(1)
                sums[i] += (ph * e).imag / (k + arb(0.5))
        tb = self.trunc_bound(c, (K - 0.5) * Delta)      # already contains the factor 1/pi
        out = []
        for i, x in enumerate(xs):
            Sx = arb(0.5) - sums[i].real / arb.pi()
            Sx = Sx + arb(0, math.nextafter(tb, INF))        # omitted terms k >= K (|Im[...]| <= |phi|)
            up_alias = math.erfc(math.sqrt(max(x + L, 0) / 2)) * (1 + 1e-9)
            lo_alias = self.lower_tail_bound(c, L - x)
            lo = flo(Sx) - up_alias
            hi = fup(Sx) + lo_alias
            out.append((lo, hi))
        return out, {"L": L, "Delta": Delta, "K": K, "trunc": tb}

    def cdf_T_certified(self, c, t=25.0):
        rem = self.remainder(t)
        xs = [c * rem["s_lo"], c * rem["s_hi"]]
        (Flo, Fhi), info = self.cdf_X_certified(c, xs)
        lo = Flo[0] - (rem["p_fail"] if rem["lo_uses_tail"] else 0.0)
        hi = Fhi[1] + rem["p_fail"]
        return {"lo": lo, "hi": hi, "width": hi - lo, "X_radius": max(Flo[1] - Flo[0], Fhi[1] - Fhi[0]) / 2, **info,
                "s_lo": rem["s_lo"], "s_hi": rem["s_hi"], "tau": float(rem["tau"].mid())}


# ============================================================================ tilted Bromwich inversion (production path)
#   P(X > x) = (Delta/pi) Re sum_{k>=0} M(theta + i s_k) e^{-(theta + i s_k) x} / (theta + i s_k) + E,   s_k = (k + 1/2) Delta,
#   |E| <= e^{-theta L}/(1 - e^{-theta L}) + sum_{m>=1} e^{theta m L} P(Z^2 > x + m L),   L = 2 pi / Delta,   0 < theta < 1/2,
#   (Poisson summation for f(t) = e^{-theta t} 1{t > 0} on the half-shifted lattice; manuscript Lemma 'tilted midpoint rule').
#   M(theta + i s) = phi(s - i theta); with z = 2 i (s - i theta) c (Re z > 0) every bound of Block.terms stays valid (||R_h|| <= 1).
#   Branch: log det_E(1 - 2 i s Sigma M) = L_E(z) + Log(1 - 2 i s theta_+) + Log r,  r = (1 - 2 i s u)/(1 - 2 i s theta_+),
#   Re r > 0 by interlacing of the ratio-form eigenvalues (manuscript Lemma 'branch'); theta_+ = unique positive ratio-form eigenvalue,
#   the root of u(c / theta) = theta, certified by a sign change.

def _alias_tilt(x, L, theta):
    a = math.exp(-theta * L)
    e = a / (1 - a)
    m = 1
    while True:
        y = x + m * L
        t = math.exp(theta * m * L) * math.erfc(math.sqrt(max(y, 0.0) / 2)) if y > 0 else math.exp(theta * m * L)
        e += t
        if t < 1e-30 or m > 200:
            break
        m += 1
    return e * (1 + 1e-9)


def alias_tilt_arb(x, L_a, theta, m_max=400):
    """Rigorous bound eps_+(x) + eps_- of the two alias sums (ball arithmetic; L_a = exact lattice period as a ball).
    eps_- = e^{-theta L}/(1 - e^{-theta L});  eps_+ = sum_{m>=1} e^{theta m L} P(Z^2 > x + mL), with P(Z^2 > y) = erfc(sqrt(y/2))
    summed in balls for m <= M and the remainder bounded through erfc(u) <= e^{-u^2}:
    sum_{m>M} e^{theta m L} e^{-(x+mL)/2} = e^{-x/2} r^{M+1}/(1-r),  r = e^{-(1/2 - theta) L}  (needs x >= 0, 0 < theta < 1/2)."""
    assert x >= 0 and 0 < theta < 0.5
    th, xa = arb(theta), arb(float(x))
    a = (-th * L_a).exp()
    em = a / (1 - a)
    ep = arb(0)
    m = 0
    tiny = arb(2) ** -200
    while m < m_max:
        m += 1
        y = xa + m * L_a
        term = (th * m * L_a).exp() * (y / 2).sqrt().erfc()
        ep += term
        if term.upper() < (tiny * (1 + ep)).lower():
            break
    r = (-(arb(0.5) - th) * L_a).exp()
    tail = (-xa / 2).exp() * r ** (m + 1) / (1 - r)
    return fup(em + ep + tail), {"m_terms": m, "eps_minus": fup(em), "eps_plus": fup(ep + tail), "tail": fup(tail)}


def _sys_trunc_tilt_arb(self, c, sK_a, theta, x):
    """Rigorous truncation bound (ball arithmetic) for the omitted nodes s_k = (k + 1/2) Delta, k >= K, sK_a = (K - 1/2) Delta:
    |psi(s)| <= e^{-theta x} (1 - 2 theta)^{-1/2} s^{-1} prod_{i in I} (2 s c nu_i)^{-1/2} for s >= sK with 2 sK c nu_i >= 1,
    nu_i certified lower bounds (Gershgorin + interlacing) of the negative-eigenvalue magnitudes / c, lambda_+ <= 1;
    Delta * sum_{k>=K} f(s_k) <= int_{sK}^inf f, f decreasing, gives (2/(pi m)) e^{-theta x} (1-2theta)^{-1/2} prod (2 sK c nu_i)^{-1/2}."""
    ca, th = arb(float(c)), arb(theta)
    sel = []
    for v in self.nu:
        ai = 2 * sK_a * ca * arb(float(v))
        if ai > 1:
            sel.append(ai)
    m = len(sel)
    if m == 0:
        return INF, 0
    lp = sum((ai.log() for ai in sel), arb(0))
    b = (-th * arb(float(x))).exp() / (1 - 2 * th).sqrt() * 2 / (arb.pi() * m) * (-lp / 2).exp()
    return fup(b), m


def _sys_theta_plus_d(self, c):
    from scipy.optimize import brentq

    def h(th):
        z = np.array([c / th + 0j])
        _, gm = self.even.terms_d(z)
        return float((1.0 - z * gm)[0].real) - th
    if h(1.0) >= -1e-15:
        return 1.0
    return brentq(h, 1e-8, 1.0, xtol=1e-15, rtol=1e-15)


def _sys_h_arb(self, c, th):
    zr = arb(float(c)) / arb(th)
    z = acb(zr)
    LE, gm = self.even.terms(z, fup(zr))
    u = 1 - z * gm
    return u.real - arb(th)


def _sys_theta_plus_cert(self, c):
    th = self.theta_plus_d(c)
    for d in (1e-13, 1e-11, 1e-9, 1e-7, 1e-5):
        lo, hi = th - d, th + d
        if self.h_arb(c, lo) > 0 and self.h_arb(c, hi) < 0:
            return arb((lo + hi) / 2, (hi - lo) / 2 * (1 + 1e-12)), th
    raise ValueError("theta_+ not certified")


def _sys_choose_L_tilt(self, xs, theta, eps):
    L = 20.0
    while max(_alias_tilt(x, L, theta) for x in xs) > eps:
        L *= 1.05
    return L


def _sys_trunc_tilt(self, c, sK, theta, xmin):
    a = 2 * sK * c * self.nu
    sel = a[a > 1.0]
    m = sel.size
    if m == 0:
        return INF
    return (math.exp(-theta * xmin) / math.sqrt(1 - 2 * theta) * 2.0 / (math.pi * m)
            * math.exp(-0.5 * np.sum(np.log(sel))))


def _sys_choose_K_tilt(self, c, Delta, theta, xmin, eps, kmax=4000000):
    # kmax raised 16 Nov 2026 (was 4e5): evaluating T_H CDFs far below the bulk (e.g. H >= 0.95 at the H = 1/2
    # critical values, E6) needs ~5e5-1e6 nodes; the rigorous stopping rule itself is unchanged.
    K = 32
    while K < kmax:
        if self.trunc_tilt(c, (K - 0.5) * Delta, theta, xmin) <= eps:
            return K
        K = int(K * 1.1) + 1
    raise RuntimeError("series truncation did not converge")


def _sys_cdf_X_double_tilt(self, c, xs, theta=0.25, eps=1e-14):
    L = self.choose_L_tilt(xs, theta, eps)
    Delta = 2 * math.pi / L
    K = self.choose_K_tilt(c, Delta, theta, min(xs), eps)
    tp = self.theta_plus_d(c)
    sg = (np.arange(K) + 0.5) * Delta
    s = sg - 1j * theta
    z = 2j * s * c
    LO, _ = self.odd.terms_d(z)
    LE, gm = self.even.terms_d(z)
    u = 1.0 - z * gm
    tis = 2j * s
    p1 = 1.0 - tis * tp
    r = (1.0 - tis * u) / p1
    M = np.exp(-0.5 * (LO + LE + np.log(p1) + np.log(r)))
    w = theta + 1j * sg
    out = []
    for x in xs:
        terms = np.real(M * np.exp(-w * x) / w)
        out.append(1.0 - Delta / math.pi * math.fsum(terms.tolist()))
    return out, {"L": L, "Delta": Delta, "K": K, "theta_plus": tp}


def _sys_cdf_T_computed_tilt(self, c, t=25.0, theta=0.25):
    rem = self.remainder(t)
    tau = float(rem["tau"].mid())
    xs = [c * rem["s_lo"], c * tau, c * rem["s_hi"]]
    F, info = self.cdf_X_double_tilt(c, xs, theta)
    lo = F[0] - (rem["p_fail"] if rem["lo_uses_tail"] else 0.0)
    hi = F[2] + rem["p_fail"]
    return {"point": F[1], "lo": lo, "hi": hi, **info}


def _sys_select(self, az, sg, allowance):
    for i in range(len(self.odd_blocks)):
        ro, _ = self.odd_blocks[i].rad_estimate(az)
        re_, rg = self.even_blocks[i].rad_estimate(az)
        if ro + re_ + 2 * az * (abs(sg) + 1) * rg <= allowance:
            return i
    return len(self.odd_blocks) - 1


def _sys_cdf_X_certified_tilt(self, c, xs, theta=0.25, eps=1e-14, eps_rad=1e-12):
    c_a = arb(float(c))
    th_a = arb(theta)
    L = self.choose_L_tilt(xs, theta, eps)
    Delta = 2 * math.pi / L
    K = self.choose_K_tilt(c, Delta, theta, min(xs), eps)
    sK = (K + 0.5) * Delta
    azmax = 2 * c * math.hypot(sK, theta) * (1 + 1e-12)
    for blk in (self.odd, self.even):
        if not azmax * blk.nF < 0.5 or not azmax * (blk.nAtt + azmax * blk.trW) < 0.5:
            raise ValueError("largest head too small for the required frequency range")
    for blk in self.odd_blocks + self.even_blocks:
        if not np.all(blk.lam > 0):
            raise ValueError("head eigenvalue estimates must be positive")
    tp, tp_d = self.theta_plus_cert(c)
    # double-precision |M| estimates guide the head size per node (the certificate does not depend on them)
    sgd = (np.arange(K) + 0.5) * Delta
    sd = sgd - 1j * theta
    zd = 2j * sd * c
    LOd, _ = self.odd.terms_d(zd)
    LEd, gmd = self.even.terms_d(zd)
    ud = 1.0 - zd * gmd
    p1d = 1.0 - 2j * sd * tp_d
    Md = np.abs(np.exp(-0.5 * (LOd + LEd + np.log(p1d) + np.log((1.0 - 2j * sd * ud) / p1d))))
    D_a = arb(Delta)
    xs_a = [arb(float(x)) for x in xs]
    sums = [arb(0) for _ in xs]
    used = np.zeros(len(self.odd_blocks), dtype=int)
    for k in range(K):
        sg = (k + arb(0.5)) * D_a
        z = acb(2 * c_a * th_a, 2 * c_a * sg)          # z = 2 i (sg - i theta) c
        az = fup(abs(z))
        allowance = math.pi * eps_rad * (k + 0.5) / (K * max(Md[k], 1e-300))
        i = self.select(az, float(sgd[k]), allowance)
        used[i] += 1
        LO, _ = self.odd_blocks[i].terms(z, az)
        LE, gm = self.even_blocks[i].terms(z, az)
        u = 1 - z * gm
        tis = acb(2 * th_a, 2 * sg)                     # 2 i s
        p1 = 1 - tis * tp
        r = (1 - tis * u) / p1
        if not (p1.real > 0 and r.real > 0):
            raise ValueError(f"branch check failed at k={k}")
        M = (-(LO + LE + p1.log() + r.log()) / 2).exp()
        w = acb(th_a, sg)
        Mw = M / w
        for j, xa in enumerate(xs_a):
            if xs[j] == 0:
                sums[j] += Mw.real
            else:
                sums[j] += (Mw * (-(w * xa)).exp()).real
    out = []
    L_a = 2 * arb.pi() / D_a                 # exact period of the lattice actually summed (ball), not the float L
    sK_a = (K - arb(0.5)) * D_a
    budget = []
    for j, x in enumerate(xs):
        S_ = D_a / arb.pi() * sums[j]
        tb, m_sel = self.trunc_tilt_arb(c, sK_a, theta, x)
        al, ainfo = alias_tilt_arb(x, L_a, theta)
        err = fup(arb(tb) + arb(al))
        P = S_ + arb(0, err)
        F = 1 - P
        out.append((flo(F), fup(F)))
        budget.append({"x": float(x), "sum_radius": float(S_.rad()), "trunc": tb, "trunc_m": m_sel, "alias": al, **ainfo})
    return out, {"L": L, "Delta": Delta, "K": K, "theta_plus": tp_d, "theta_plus_rad": float(tp.rad()),
                 "heads_used": dict(zip(self.heads, used.tolist())), "budget": budget}


def _sys_cdf_T_certified_tilt(self, c, t=25.0, theta=0.25):
    rem = self.remainder(t)
    ca = arb(float(c))
    # evaluation points rounded outward: P(X_J <= x) is increasing in x, so x_lo <= c s_lo and x_hi >= c s_hi keep both
    # inequalities of the first statement valid for the exact real products
    xs = [0.0 if rem["s_lo"] == 0.0 else max(0.0, flo(ca * arb(rem["s_lo"]))), fup(ca * arb(rem["s_hi"]))]
    (Flo, Fhi), info = self.cdf_X_certified_tilt(c, xs, theta)
    p_up = fup(arb(-float(t)).exp())          # e^{-t} rounded up
    lo = flo(arb(Flo[0]) - arb(p_up)) if rem["lo_uses_tail"] else Flo[0]
    hi = fup(arb(Fhi[1]) + arb(p_up))
    return {"lo": lo, "hi": hi, "width": hi - lo, "X_radius": max(Flo[1] - Flo[0], Fhi[1] - Fhi[0]) / 2, **info,
            "s_lo": rem["s_lo"], "s_hi": rem["s_hi"], "tau": float(rem["tau"].mid())}


BridgeSystem.theta_plus_d = _sys_theta_plus_d
BridgeSystem.h_arb = _sys_h_arb
BridgeSystem.theta_plus_cert = _sys_theta_plus_cert
BridgeSystem.choose_L_tilt = _sys_choose_L_tilt
BridgeSystem.trunc_tilt = _sys_trunc_tilt
BridgeSystem.trunc_tilt_arb = _sys_trunc_tilt_arb
BridgeSystem.choose_K_tilt = _sys_choose_K_tilt
BridgeSystem.cdf_X_double_tilt = _sys_cdf_X_double_tilt
BridgeSystem.cdf_T_computed_tilt = _sys_cdf_T_computed_tilt
BridgeSystem.cdf_X_certified_tilt = _sys_cdf_X_certified_tilt
BridgeSystem.select = _sys_select
BridgeSystem.cdf_T_certified_tilt = _sys_cdf_T_certified_tilt


# ============================================================================ law of W = int Btilde^2 alone (KPSS-type F3 functional)
#   For W >= 0 and theta > 0, L > x (f(t) = e^{theta t} 1{t < 0}, Poisson summation on the half-shifted lattice):
#   P(W < x) = (Delta/pi) Re sum_{k>=0} M(-theta + i s_k) e^{(theta - i s_k) x} / (theta - i s_k) + E,  |E| <= e^{-theta L}/(1 - e^{-theta L}),
#   M(w) = E e^{w W} = exp(-(L_O(z) + L_E(z))/2),  z = -2 w = 2 theta - 2 i s  (Re z > 0: every Block bound applies).

def _sys_W_params(self, x, eps, Lfac=8.0):
    # window L = Lfac * x: the factor e^{theta x} = (1/eps + 1)^(1/Lfac) multiplies every ball radius, so Lfac trades
    # nodes (K ~ Lfac) against radius amplification; Lfac = 8 keeps it near 56 for eps = 1e-14.
    L = Lfac * max(x, 1e-300)
    theta = math.log(1.0 / eps + 1.0) / L
    return L, theta


def _sys_trunc_W(self, sK, theta, x, nu_all):
    a = 2 * sK * nu_all
    sel = a[a > 1.0]
    m = sel.size
    if m == 0:
        return INF
    return math.exp(theta * x) * 2.0 / (math.pi * m) * math.exp(-0.5 * np.sum(np.log(sel)))


def _sys_nu_all(self):
    """Certified lower bounds of the eigenvalues of the bridge covariance (W = int Btilde^2 has no Z update, so every
    head eigenvalue of both parity blocks is used): Gershgorin clusters of the head blocks, sorted, and Cauchy interlacing
    head -> full block (17 Nov 2026; replaces lam - eta, whose pairing with the sorted eigenvalues was not guaranteed)."""
    nu = np.concatenate([self.odd.eig_lo, self.even.eig_lo])
    return np.sort(nu[nu > 0])[::-1]


def _sys_trunc_W_arb(self, sK_a, theta, x, nu):
    """Ball-arithmetic version of trunc_W: e^{theta x} (2/(pi m)) prod_{i in I} (2 sK nu_i)^{-1/2}, I = {2 sK nu_i > 1}."""
    th = arb(theta)
    sel = [a for a in (2 * sK_a * arb(float(v)) for v in nu) if a > 1]
    m = len(sel)
    if m == 0:
        return INF
    lp = sum((a.log() for a in sel), arb(0))
    return fup((th * arb(float(x))).exp() * 2 / (arb.pi() * m) * (-lp / 2).exp())


def _sys_cdf_WJ_double(self, x, eps=1e-14):
    L, theta = self.W_params(x, eps)
    Delta = 2 * math.pi / L
    nu = self.nu_all()
    K = 32
    while self.trunc_W((K - 0.5) * Delta, theta, x, nu) > eps:
        K = int(K * 1.1) + 1
    sg = (np.arange(K) + 0.5) * Delta
    z = 2 * theta - 2j * sg
    LO, _ = self.odd.terms_d(z)
    LE, _ = self.even.terms_d(z)
    M = np.exp(-0.5 * (LO + LE))
    terms = np.real(M * np.exp((theta - 1j * sg) * x) / (theta - 1j * sg))
    return Delta / math.pi * math.fsum(terms.tolist()), {"L": L, "theta": theta, "K": K}


def _sys_cdf_WJ_certified(self, x, eps=1e-14, eps_rad=1e-13):
    L, theta = self.W_params(x, eps)
    Delta = 2 * math.pi / L
    nu = self.nu_all()
    K = 32
    while self.trunc_W((K - 0.5) * Delta, theta, x, nu) > eps:
        K = int(K * 1.1) + 1
    th_a, D_a, x_a = arb(theta), arb(Delta), arb(float(x))
    sgd = (np.arange(K) + 0.5) * Delta
    zd = 2 * theta - 2j * sgd
    LOd, _ = self.odd.terms_d(zd)
    LEd, _ = self.even.terms_d(zd)
    Md = np.abs(np.exp(-0.5 * (LOd + LEd)))
    S = arb(0)
    used = np.zeros(len(self.odd_blocks), dtype=int)
    for k in range(K):
        sg = (k + arb(0.5)) * D_a
        z = acb(2 * th_a, -2 * sg)
        az = fup(abs(z))
        allowance = math.pi * eps_rad * (k + 0.5) / (K * max(Md[k] * math.exp(theta * x), 1e-300))
        i = self.select(az, float(sgd[k]), allowance)
        used[i] += 1
        LO, _ = self.odd_blocks[i].terms(z, az)
        LE, _ = self.even_blocks[i].terms(z, az)
        w = acb(th_a, -sg)
        S += ((-(LO + LE) / 2).exp() * (w * x_a).exp() / w).real
    P = D_a / arb.pi() * S
    # alias on the exact period L' = 2 pi / Delta of the lattice used; W >= 0 and L' > x, so only e^{-theta m L'} terms remain
    L_a = 2 * arb.pi() / D_a
    assert L_a > arb(float(x)), "window must exceed x"
    a = (-th_a * L_a).exp()
    tb = self.trunc_W_arb((K - arb(0.5)) * D_a, theta, x, nu)
    P = P + arb(0, fup(arb(tb) + a / (1 - a)))
    return (flo(P), fup(P)), {"L": L, "theta": theta, "K": K, "heads_used": dict(zip(self.heads, used.tolist()))}


def _sys_cdf_Wfull(self, x, certified=False, t=25.0):
    """P(int_0^1 Btilde^2 <= x): bracket from W_J and the remainder R_J (Laurent-Massart)."""
    rem = self.remainder(t)
    if not certified:
        x_hi_arg = x - rem["s_lo"]      # W_J <= x - R_J, R_J >= s_lo (w.p. >= 1 - e^-t) -> upper bound
        x_lo_arg = x - rem["s_hi"]
        a1, i1 = self.cdf_WJ_double(x_lo_arg)
        b2, i2 = self.cdf_WJ_double(x_hi_arg)
        lo = a1 - rem["p_fail"]
        hi = b2 + (rem["p_fail"] if rem["lo_uses_tail"] else 0.0)
        return {"lo": lo, "hi": hi, "width": hi - lo, "K": max(i1["K"], i2["K"])}
    # certified (17 Nov 2026): evaluation points rounded outward, e^{-t} rounded up, assembly in balls
    xa = arb(float(x))
    x_lo_arg = flo(xa - arb(rem["s_hi"]))
    x_hi_arg = fup(xa - arb(rem["s_lo"]))
    pf = arb(fup((-arb(t)).exp()))
    if x_lo_arg > 0:
        (a1, _), i1 = self.cdf_WJ_certified(x_lo_arg)
    else:
        a1, i1 = 0.0, {"K": 0}
    (_, b2), i2 = self.cdf_WJ_certified(x_hi_arg)
    lo = max(0.0, flo(arb(a1) - pf))
    hi = min(1.0, fup(arb(b2) + pf)) if rem["lo_uses_tail"] else min(1.0, b2)
    return {"lo": lo, "hi": hi, "width": hi - lo, "K": max(i1["K"], i2["K"])}


BridgeSystem.W_params = _sys_W_params
BridgeSystem.trunc_W = _sys_trunc_W
BridgeSystem.trunc_W_arb = _sys_trunc_W_arb
BridgeSystem.nu_all = _sys_nu_all
BridgeSystem.cdf_WJ_double = _sys_cdf_WJ_double
BridgeSystem.cdf_WJ_certified = _sys_cdf_WJ_certified
BridgeSystem.cdf_Wfull = _sys_cdf_Wfull
