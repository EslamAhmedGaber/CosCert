"""Laws of finite Gaussian quadratic forms X = sum_i theta_i xi_i^2 (xi_i iid N(0,1)) and the H = 1/2 closed forms.

Methods
  imhof_cdf        : Imhof (1961) real integral, double precision (computed).
  ruben_beta_cdf0  : P(X <= 0) for forms with exactly ONE positive weight (all ratio statistics here:
                     stat^2 <= c  <=>  Z^2 - c * PSD form <= 0).  Ruben's (1962) mixture of the negative part,
                         W = sum a_i xi_i^2 = beta * chi2_{r+2K},  P(K = k) = c_k >= 0,  sum_k c_k = 1,  beta = min a_i,
                     turns the CDF into  sum_k c_k I_u(1/2, (r+2k)/2),  u = beta/(theta_+ + beta)
                     (regularised incomplete beta).  I_u(1/2, b) increases with b, so truncation after K terms is
                     enclosed exactly:  S_K + I_K tail <= F <= S_K + tail,  tail = 1 - sum_{k<=K} c_k.
                     (Corrected 16 Nov 2026: an earlier draft used S_K + I_K tail as the upper end.)
                     Double version (scipy) and arb version (python-flint; rigorous enclosure).
Pitfall recorded 15 Nov 2026: python-flint passes the evaluation point as `self`:
  z.beta_lower(a, b, regularized=True) = I_z(a, b);  z.hypgeom_1f1(a, b) = 1F1(a; b; z).
"""
import math

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.special import betainc, betaln


# ---------------------------------------------------------------- Imhof (computed)
def imhof_cdf(theta, y=0.0, epsabs=1e-13, epsrel=1e-12, limit=4000):
    """P(sum theta_i xi_i^2 <= y), double precision."""
    th = np.asarray(theta, dtype=float)

    def f(u):
        a = th * u
        ang = 0.5 * np.sum(np.arctan(a)) - 0.5 * y * u
        lrho = 0.25 * np.sum(np.log1p(a * a))
        return math.sin(ang) * math.exp(-lrho) / u      # exp(-lrho) underflows harmlessly (16 Nov 2026 fix)

    val, err = integrate.quad(f, 0.0, np.inf, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return 0.5 - val / math.pi, err / math.pi


# ---------------------------------------------------------------- Ruben-Beta for one positive weight
def _split(theta):
    th = np.sort(np.asarray(theta, dtype=float))[::-1]
    if not (th[0] > 0 and np.all(th[1:] < 0)):
        raise ValueError("form must have exactly one positive weight and the rest negative")
    return th[0], -th[1:]


def ruben_beta_cdf0(theta, tol=1e-15, kmax=None):
    """P(X <= 0), double precision (computed), with the truncation enclosure.
    Returns (value, width): midpoint and width of [S_K + I_K tail, S_K + tail].
    The number of terms is capped at the a-priori geometric count (in double precision the running mass cannot
    certify 1 - mass < 1e-15 because of rounding; 16 Nov 2026)."""
    tp, a = _split(theta)
    if kmax is None:
        kmax = min(200000, _ruben_terms_needed(tp, a, tol))
    r = a.size
    beta = a.min()
    gam = 1.0 - beta / a
    c0 = math.exp(0.5 * np.sum(np.log(beta / a)))
    u = beta / (tp + beta)
    lu, l1u = math.log(u), math.log1p(-u)
    g = np.empty(kmax + 1)
    d = np.empty(kmax + 2)
    d[0] = 1.0
    gpow = np.ones_like(gam)
    b = 0.5 * r
    I = betainc(0.5, b, u)
    S = c0 * I
    mass = c0
    k = 0
    while 1.0 - mass > tol and k < kmax:
        k += 1
        gpow = gpow * gam
        g[k - 1] = gpow.sum()
        d[k] = np.dot(g[:k], d[k - 1::-1]) / (2.0 * k)
        # I_u(1/2, b+1) = I_u(1/2, b) + u^(1/2) (1-u)^b / (b B(1/2, b))
        I = I + math.exp(0.5 * lu + b * l1u - math.log(b) - betaln(0.5, b))
        b += 1.0
        ck = c0 * d[k]
        S += ck * I
        mass += ck
    tail = max(0.0, 1.0 - mass)
    return S + 0.5 * tail * (1.0 + I), tail * (1.0 - I)


def ruben_beta_cdf0_arb(theta_lo, theta_hi=None, prec=160, tol_bits=60, kmax=100000):
    """Rigorous enclosure of P(X <= 0) for a one-positive-weight form whose weights lie in the given
    componentwise intervals [theta_lo, theta_hi] (sorted descending, same length).  Uses monotone coupling:
    X is stochastically increasing in every weight, so F(0; theta_hi) <= F(0; theta) <= F(0; theta_lo).
    Returns (lo, hi) as Python floats rounded outward (via arb endpoints)."""
    from flint import arb, ctx
    if theta_hi is None:
        theta_hi = theta_lo
    old = ctx.prec
    ctx.prec = prec
    try:
        out = []
        for th in (np.asarray(theta_hi, float), np.asarray(theta_lo, float)):   # hi weights -> lower CDF
            tp_f, a_f = _split(th)
            tp = arb(float(tp_f))                      # exact binary values (rigorous monotone coupling)
            a = [arb(float(x)) for x in a_f]
            r = len(a)
            beta = a[int(np.argmin(a_f))]
            gam = [1 - beta / ai for ai in a]
            logc0 = sum(((beta / ai).log() for ai in a), arb(0)) / 2
            c0 = logc0.exp()
            u = beta / (tp + beta)
            b = arb(r) / 2
            I = u.beta_lower(arb(0.5), b, regularized=True)
            gpow = [arb(1)] * r
            gs, d = [], [arb(1)]
            S = c0 * I
            mass = c0
            k = 0
            eps = arb(2) ** (-tol_bits)
            while k < kmax:
                rem = 1 - mass
                if rem.upper() < eps.upper():
                    break
                k += 1
                gpow = [gp * g for gp, g in zip(gpow, gam)]
                gs.append(sum(gpow, arb(0)))
                dk = sum((gs[j - 1] * d[k - j] for j in range(1, k + 1)), arb(0)) / (2 * k)
                d.append(dk)
                I = I + (u.log() * arb(0.5) + b * (1 - u).log() - b.log() - (arb(0.5).lgamma() + b.lgamma() - (b + arb(0.5)).lgamma())).exp()
                b = b + 1
                ck = c0 * dk
                S = S + ck * I
                mass = mass + ck
            tail = 1 - mass
            lo = S + tail * I          # remaining terms have I_u(1/2, b) in [I_K, 1]
            hi = S + tail
            out.append((math.nextafter(float(lo.lower()), -math.inf), math.nextafter(float(hi.upper()), math.inf), k))
        lo_cdf = out[0][0]     # from theta_hi
        hi_cdf = out[1][1]     # from theta_lo
        return lo_cdf, hi_cdf, max(out[0][2], out[1][2])
    finally:
        ctx.prec = old


# ---------------------------------------------------------------- H = 1/2 closed forms (mpmath)
def _log_sinhz_over_z(z):
    """Branch-continuous log(sinh z / z) for Re z > 0."""
    return z + mp.log(1 - mp.exp(-2 * z)) - mp.log(2 * z)


def cdf_T_half(c, dps=30):
    """P(T <= c), T = Z^2 / int_0^1 (W(r) - r W(1))^2 dr, Z independent of the Brownian bridge (H = 1/2)."""
    with mp.workdps(dps):
        c = mp.mpf(c)

        def f(s):
            if s == 0:
                return mp.mpf(0)
            z = mp.sqrt(2j * s * c)
            lphi = -0.5 * mp.log(1 - 2j * s) - 0.5 * _log_sinhz_over_z(z)
            return mp.im(mp.exp(lphi)) / s
        I = mp.quad(f, [0, 0.01, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 300, 1000, 4000])
        return mp.mpf(0.5) - I / mp.pi


def cdf_cvm(x, dps=30):
    """P(int_0^1 V^2 <= x), V the Brownian bridge (Cramer-von Mises); also the law of int (W - int W)^2."""
    with mp.workdps(dps):
        x = mp.mpf(x)

        def f(s):
            z = mp.sqrt(-2j * s)
            if mp.re(z) < 0:
                z = -z
            phi = mp.exp(-0.5 * _log_sinhz_over_z(z))
            return mp.im(phi * mp.exp(-1j * s * x)) / s
        I = mp.quad(f, [0, 1, 5, 20, 100, 500, 2000, 10000, 50000])
        return mp.mpf(0.5) - I / mp.pi


def cdf_l2_bm(x, dps=30):
    """P(int_0^1 W^2 <= x) for Brownian motion, from E exp(-a int W^2) = (cosh sqrt(2a))^(-1/2)."""
    with mp.workdps(dps):
        x = mp.mpf(x)

        def f(s):
            z = mp.sqrt(-2j * s)
            if mp.re(z) < 0:
                z = -z
            lcosh = z + mp.log(1 + mp.exp(-2 * z)) - mp.log(2)
            phi = mp.exp(-0.5 * lcosh)
            return mp.im(phi * mp.exp(-1j * s * x)) / s
        I = mp.quad(f, [0, 1, 5, 20, 100, 500, 2000, 10000, 50000])
        return mp.mpf(0.5) - I / mp.pi


def quantile(cdf, p, bracket, tol=1e-22):
    return mp.findroot(lambda c: cdf(c) - mp.mpf(p), bracket, solver="anderson", tol=tol)



# ---------------------------------------------------------------- fast closed-form inversions (double precision, controlled error)
#   For W >= 0 with moment generating function M(w) = E e^{w W}, theta > 0 and L > x (Poisson summation, f(t) = e^{theta t} 1{t<0}):
#   P(W < x) = (Delta/pi) Re sum_{k>=0} M(-theta + i s_k) e^{(theta - i s_k) x}/(theta - i s_k) + E,  |E| <= e^{-theta L}/(1 - e^{-theta L}),
#   s_k = (k + 1/2) Delta, Delta = 2 pi / L.  Terms are summed until |M| e^{theta x}/s_k drops below 1e-18 for 50 consecutive k
#   (the closed-form transforms below decay like exp(-c sqrt(s))).
def _lcosh(z):
    return z + np.log1p(np.exp(-2 * z)) - math.log(2.0)


def _lsinh_over(z):
    """log(sinh z / z) for Re z > 0 (vectorised)."""
    return z + np.log1p(-np.exp(-2 * z)) - math.log(2.0) - np.log(z)


def _bromwich_W(logM, x, Lfac=8.0, eps=1e-15, kmax=4_000_000, block=20000):
    L = Lfac * x
    theta = math.log(1.0 / eps + 1.0) / L
    Delta = 2 * math.pi / L
    total, k0 = [], 0
    while k0 < kmax:
        sg = (np.arange(k0, k0 + block) + 0.5) * Delta
        w = -theta + 1j * sg
        t = np.real(np.exp(logM(w) + (theta - 1j * sg) * x) / (theta - 1j * sg))
        total.append(math.fsum(t.tolist()))
        if np.max(np.abs(np.exp(logM(w[-50:]).real + theta * x) / sg[-50:])) < 1e-18:
            break
        k0 += block
    a = math.exp(-theta * L)
    return Delta / math.pi * math.fsum(total), a / (1 - a)


def cdf_l2_bm_fast(x):
    """P(int_0^1 W^2 <= x), M(w) = cosh(sqrt(-2w))^(-1/2)."""
    return _bromwich_W(lambda w: -0.5 * _lcosh(np.sqrt(-2 * w)), x)[0]


def cdf_cvm_fast(x):
    """P(int_0^1 V^2 <= x) (Brownian bridge), M(w) = (sqrt(-2w)/sinh sqrt(-2w))^(1/2)."""
    return _bromwich_W(lambda w: -0.5 * _lsinh_over(np.sqrt(-2 * w)), x)[0]



def _ruben_terms_needed(tp_f, a_f, tol):
    """A-priori number of Ruben terms (sizes the series only; the tail mass is verified in arb afterwards).
    The mixture weights are the coefficients of prod_i (beta/a_i)^(1/2) (1 - gamma_i x)^(-1/2); their tail decays
    like gamma_max^K times a polynomial factor, so K = 1.3 log(tol)/log(gamma_max) + 64 is ample."""
    beta = a_f.min()
    gmax = float(np.max(1.0 - beta / a_f))
    if gmax <= 0.0:
        return 1
    return int(1.3 * math.log(tol) / math.log(gmax)) + 64


def ruben_beta_cdf0_arb_fast(theta_lo, theta_hi=None, prec=128, tol_bits=60):
    """As ruben_beta_cdf0_arb, but the mixture weights c_k are the coefficients of the power series
    c0 * exp(-1/2 sum_i log(1 - gamma_i x)) computed by FLINT (O(K log K)); identical enclosure semantics."""
    from flint import arb, arb_series, ctx
    if theta_hi is None:
        theta_hi = theta_lo
    old, old_cap = ctx.prec, ctx.cap
    ctx.prec = prec
    try:
        out = []
        eps = arb(2) ** (-tol_bits)
        for th in (np.asarray(theta_hi, float), np.asarray(theta_lo, float)):
            tp_f, a_f = _split(th)
            K = max(16, _ruben_terms_needed(tp_f, a_f, 2.0 ** (-tol_bits)) + 16)
            ctx.cap = K + 1                      # python-flint truncates every series to ctx.cap terms
            ctx.prec = max(prec, 160 + K // 2)   # ball series exp loses about 1/2 bit per term (measured 16 Nov 2026)
            tp = arb(float(tp_f))
            a = [arb(float(x)) for x in a_f]
            r = len(a)
            beta = a[int(np.argmin(a_f))]
            c0 = (sum(((beta / ai).log() for ai in a), arb(0)) / 2).exp()
            while True:
                S = arb_series([0], prec=K + 1)
                for ai in a:
                    S = S + arb_series([1, -(1 - beta / ai)], prec=K + 1).log()
                D = (S * arb(-0.5)).exp()          # (series / int) hits a python-flint attribute bug
                cks = [c0 * x for x in D.coeffs()[:K + 1]]
                mass = sum(cks, arb(0))
                tail = 1 - mass
                if tail.upper() < eps.upper():
                    break
                K *= 2
                ctx.cap = K + 1
                ctx.prec = max(prec, 160 + K // 2)
                if K > 5_000_000:
                    raise RuntimeError("Ruben series did not converge")
            u = beta / (tp + beta)
            b = arb(r) / 2
            I = u.beta_lower(arb(0.5), b, regularized=True)
            lu, l1u, lg_half = u.log(), (1 - u).log(), arb(0.5).lgamma()
            Ssum = arb(0)
            for k in range(len(cks)):
                Ssum += cks[k] * I
                I = I + (lu / 2 + b * l1u - b.log() - (lg_half + b.lgamma() - (b + arb(0.5)).lgamma())).exp()
                b = b + 1
            # after the loop, I = I_u(1/2, r/2 + K + 1) bounds the omitted terms from below
            lo, hi = Ssum + tail * I, Ssum + tail
            out.append((math.nextafter(float(lo.lower()), -math.inf), math.nextafter(float(hi.upper()), math.inf), K))
        return out[0][0], out[1][1], max(out[0][2], out[1][2])
    finally:
        ctx.prec = old
        ctx.cap = old_cap



def cdf_T_half_fast(c, theta=0.25, eps=1e-15):
    """P(T <= c) at H = 1/2 from the closed-form transform of X = Z^2 - c int V^2:
    E e^{w X} = (1 - 2w)^(-1/2) (sqrt(2wc)/sinh sqrt(2wc))^(1/2), tilted Bromwich line Re w = theta (0 < theta < 1/2):
    P(X > 0) = (Delta/pi) Re sum_k M(theta + i s_k)/(theta + i s_k) + E, |E| <= e^{-theta L}/(1 - e^{-theta L})
               + sum_m e^{theta m L} P(Z^2 > m L).  Principal branches are continuous on the line (Re(1 - 2w) > 0, Re sqrt(2wc) > 0)."""
    L = 20.0
    while True:
        a = math.exp(-theta * L)
        from scipy.special import log_ndtr
        al = a / (1 - a) + sum(math.exp(theta * m * L + math.log(2.0) + float(log_ndtr(-math.sqrt(m * L)))) for m in range(1, 50))
        if al < eps:
            break
        L *= 1.05
    Delta = 2 * math.pi / L
    total, k0, block = [], 0, 20000
    while True:
        sg = (np.arange(k0, k0 + block) + 0.5) * Delta
        w = theta + 1j * sg
        z = np.sqrt(2 * w * c)
        logM = -0.5 * np.log(1 - 2 * w) - 0.5 * _lsinh_over(z)
        t = np.real(np.exp(logM) / w)
        total.append(math.fsum(t.tolist()))
        if np.max(np.abs(np.exp(logM[-50:].real) / sg[-50:])) < 1e-18 or k0 > 4_000_000:
            break
        k0 += block
    return 1.0 - Delta / math.pi * math.fsum(total)
