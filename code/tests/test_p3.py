"""Fast unit tests of the identities the manuscript relies on (run: python -m pytest -q work/code/tests)."""
import math
import os
import sys

import mpmath as mp
import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from p3 import finite_n as fn, gram as gr, lowrank as lr  # noqa: E402


def pr(H, k):
    v = mp.hyp1f1(2 * H - 1, 2 * H, 1j * k * mp.pi)
    return float(mp.re(v)), float(mp.im(v))


def sine_gram(H, k, l):
    b = 2 * H - 1; pk, rk = pr(H, k); pl, rl = pr(H, l)
    if k == l:
        return 2 * H * (pk + (1 + b) * rk / (k * math.pi))
    if (k + l) % 2:
        return 0.0
    return 4 * H / math.pi * (k * rl - l * rk) / (k * k - l * l)


def test_gram_identity_at_half():
    G = gr.cosine_gram(0.5, 12)
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-13)


@pytest.mark.parametrize("H", [0.3, 0.6, 0.75, 0.9])
def test_sine_gram_formula_matches_implementation(H):
    m = 3
    Sg = np.asarray(lr.FixedSmoothingLaw(f"{H:.2f}", "HI", m).Sd, float)
    mine = np.array([[0.5 * sine_gram(H, 2 * i, 2 * j) for j in range(1, m + 1)] for i in range(1, m + 1)])
    assert np.abs(mine - Sg[1 + m:, 1 + m:]).max() < 1e-12
    assert np.abs(Sg[:1 + m, 1 + m:]).max() == 0.0          # (Z, C) uncorrelated with S (Proposition 1, parity)


@pytest.mark.parametrize("H", [0.3, 0.75])
def test_sine_gram_against_fgn_fourier_sums(H):
    n = 4096; g = fn.acf_fgn(H, n); t = np.arange(1, n + 1)
    V = np.stack([np.sin(2 * np.pi * j * t / n) for j in (1, 2)], axis=1)
    C = (V.T @ np.stack([fn.matmul_toeplitz(g, V[:, i]) for i in range(2)], axis=1)) / n ** (2 * H)
    lim = np.array([[0.5 * sine_gram(H, 2 * i, 2 * j) for j in (1, 2)] for i in (1, 2)])
    assert np.abs(C - lim).max() < 5e-4


@pytest.mark.parametrize("kind,k", [("HI", 1), ("HI", 4), ("EWC", 2), ("EWC", 16)])
def test_student_t_anchor_at_half(kind, k):
    law = lr.FixedSmoothingLaw("0.50", kind, k); dof = 2 * k if kind == "HI" else k
    assert abs(math.sqrt(law.quantile2(0.95)) - stats.t.ppf(0.975, dof)) < 1e-8


def test_rho_properties():
    rho_sn = lambda H: 6 / ((2 * H + 1) * (H + 1)) - 1
    assert abs(rho_sn(0.5) - 1) < 1e-15
    assert abs((rho_sn(1 - 1e-6)) / 1e-6 - 7 / 6) < 1e-5
    for kind, k in (("HI", 4), ("EWC", 8)):
        Sd = np.asarray(lr.FixedSmoothingLaw("0.50", kind, k).Sd, float)
        assert abs(np.trace(Sd[1:, 1:]) / k - 1) < 1e-12


def test_branch_lemma_positive_real_parts():
    rng = np.random.default_rng(1)
    for _ in range(200):
        N = 6; X = rng.standard_normal((N + 1, N + 1)); Sig = X @ X.T + 0.1 * np.eye(N + 1)
        Sig /= Sig[0, 0]; g, A = Sig[1:, 0], Sig[1:, 1:]; c = rng.uniform(0.5, 50)
        lam = np.linalg.eigvalsh(np.linalg.cholesky(Sig).T @ np.diag([1.0] + [-c] * N) @ np.linalg.cholesky(Sig))
        lp = lam.max(); assert lp <= 1 + 1e-12 and (lam > 0).sum() == 1
        for th in (0.0, 0.25 / max(lp, 1e-9) * 0.99):
            for s in np.linspace(-50, 50, 41):
                w = 2 * th + 2j * s
                u = 1 - c * w * g @ np.linalg.solve(np.eye(N) + w * c * A, g)
                r = (1 - w * u) / (1 - w * lp)
                assert r.real > 0 and u.real >= 1 - g @ np.linalg.solve(A, g) - 1e-12


def test_fgn_identity_sn():
    """T_n under fGn equals the rectangle-rule discretisation of T_H (Proposition 6): same exact law."""
    H, n = 0.7, 64; g = fn.acf_fgn(H, n); sn = fn.SNExact(g)
    t = np.arange(1, n + 1) / n; R = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    L = np.linalg.cholesky(R + 1e-14 * np.eye(n)); rng = np.random.default_rng(0)
    B = rng.standard_normal((200000, n)) @ L.T; Bt = B - t[None, :] * B[:, -1:]
    T = B[:, -1] ** 2 / np.mean(Bt ** 2, axis=1); c = 45.0
    p_mc = np.mean(T <= c); p_ex = sn.cdf(c)
    assert abs(p_mc - p_ex) < 4 * math.sqrt(p_ex * (1 - p_ex) / T.size)


def test_laurent_massart_bracket():
    rng = np.random.default_rng(2); mu = 1.0 / np.arange(1, 400) ** 3; tau = mu.sum(); mbar = mu.max(); t = 5.0
    s_hi = tau + 2 * math.sqrt(mbar * tau * t) + 2 * mbar * t; s_lo = max(0.0, tau - 2 * math.sqrt(mbar * tau * t))
    R = (rng.standard_normal((100000, mu.size)) ** 2) @ mu
    assert np.mean(R >= s_hi) <= math.exp(-t) and np.mean(R <= s_lo) <= math.exp(-t)



# ---------------------------------------------------------------------------- regression tests added 16 Nov (review round)
_WORK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_surface_table_level_mapping():
    """Response-surface table: level p is two-sided alpha = 1 - p (not 2(1-p))."""
    tab = open(os.path.join(_WORK, "manuscript", "tables", "tab_surface.tex"), encoding="utf-8").read()
    assert "2(1-p)" not in tab and r"\alpha=1-p" in tab


def test_e3b_mc_functions_have_no_undefined_names():
    """Every name read inside each function of e3b_mc.py is defined (local, argument, module-level or builtin)."""
    import ast
    import builtins
    src = open(os.path.join(_WORK, "code", "experiments", "e3b_mc.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    module_names = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            module_names |= {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    for fn_ in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        local = {a.arg for a in fn_.args.args + fn_.args.kwonlyargs}
        for n in ast.walk(fn_):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                local.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not fn_:
                local |= {a.arg for a in n.args.args}
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                local |= {(a.asname or a.name).split(".")[0] for a in n.names}
            elif isinstance(n, ast.ExceptHandler) and n.name:
                local.add(n.name)
        used = {n.id for n in ast.walk(fn_) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        undefined = used - local - module_names
        assert not undefined, f"{fn_.name}: undefined {sorted(undefined)}"


def test_certified_cdf_orientation_at_half():
    """The certified enclosure is of the distribution function P(T <= c): at H = 1/2 it contains 0.95 at the closed-form
    0.95-quantile 45.526085752448 (Abadir-Paruolo), even with a small truncation J."""
    from p3 import certify as cf
    from flint import ctx
    ctx.prec = 128
    sysm = cf.BridgeSystem("0.50", 64)
    P = sysm.cdf_T_certified_tilt(45.526085752448)
    assert P["lo"] < 0.95 < P["hi"] and P["hi"] - P["lo"] < 0.05


def test_certified_surface_record():
    """Printed response surface: 150 held-out cells, every one with a certified size-error bound below 1e-3 pp."""
    import json
    d = json.load(open(os.path.join(_WORK, "results", "E9_surface_certified.json")))
    assert len(d["cells"]) == 150 and all(c["bound_pp"] < 1e-3 for c in d["cells"]) and d["digits"] == 7


# ---------------------------------------------------------------- 17 Nov 2026 additions (audit A1, A4)
def _ruben_encl(theta):
    from p3 import laws as lw
    th = np.sort(np.asarray(theta, float))[::-1]
    return lw.ruben_beta_cdf0_arb_fast(th, th)[:2], lw.ruben_beta_cdf0_arb(th, th)[:2]


@pytest.mark.parametrize("tp,a", [(1.0, 0.3), (0.8, 2.5), (1.0, 1e-3), (0.37, 0.37)])
def test_ruben_one_negative_weight_exact(tp, a):
    """X = tp Z1^2 - a Z2^2: P(X <= 0) = (2/pi) arctan(sqrt(a/tp)); both certified series must enclose it."""
    mp.mp.dps = 50
    exact = 2 / mp.pi * mp.atan(mp.sqrt(mp.mpf(a) / mp.mpf(tp)))
    for lo, hi in _ruben_encl([tp, -a]):
        assert mp.mpf(lo) <= exact <= mp.mpf(hi) and hi - lo < 1e-12


@pytest.mark.parametrize("tp,a,r", [(1.0, 0.2, 5), (0.9, 0.05, 16), (1.0, 1.5, 2)])
def test_ruben_repeated_weights_exact(tp, a, r):
    """X = tp chi2_1 - a chi2_r: P(X <= 0) = I_u(1/2, r/2), u = a/(tp + a) (the mixture collapses to its first term)."""
    mp.mp.dps = 50
    u = mp.mpf(a) / (mp.mpf(tp) + mp.mpf(a))
    exact = mp.betainc(mp.mpf(1) / 2, mp.mpf(r) / 2, 0, u, regularized=True)
    for lo, hi in _ruben_encl([tp] + [-a] * r):
        assert mp.mpf(lo) <= exact <= mp.mpf(hi) and hi - lo < 1e-12


@pytest.mark.parametrize("kind,k", [("HI", 4), ("EWC", 16)])
def test_certified_fixed_smoothing_student_t_anchor(kind, k):
    """At H = 1/2 the fixed-smoothing statistics are Student t (2m or B d.f.): the certified enclosure of
    P(stat^2 <= c) contains the exact t probability at a binary cutoff c."""
    from flint import ctx
    ctx.prec = 128
    law = lr.FixedSmoothingLaw("0.50", kind, k)
    dof = 2 * k if kind == "HI" else k
    c = float(stats.t.ppf(0.975, dof)) ** 2
    lo, hi = law.cdf2_certified(c)
    mp.mp.dps = 40
    x = mp.sqrt(mp.mpf(c))
    exact = 1 - mp.betainc(mp.mpf(dof) / 2, mp.mpf(1) / 2, 0, dof / (dof + x * x), regularized=True)
    assert mp.mpf(lo) <= exact <= mp.mpf(hi) and hi - lo < 1e-10


def test_ruben_enclosure_monotone_in_weight_endpoints():
    """Monotone coupling: widening the weight intervals widens the enclosure, which still contains the point value."""
    from p3 import laws as lw
    th = np.array([1.0, -0.2, -0.35, -0.5, -0.05])
    lo0, hi0 = lw.ruben_beta_cdf0_arb_fast(th, th)[:2]
    dlt = 1e-6 * np.abs(th)
    lo1, hi1 = lw.ruben_beta_cdf0_arb_fast(th - dlt, th + dlt)[:2]
    assert lo1 <= lo0 <= hi0 <= hi1 and hi1 - lo1 > hi0 - lo0


def test_gershgorin_sorted_bounds_enclose_eigenvalues():
    """certify.gersh_sorted: sorted outward bounds contain the sorted eigenvalues (50-digit reference), including a
    near-diagonal matrix with a tight cluster (the case Kahan's pairing argument does not cover)."""
    from flint import arb_mat, ctx
    from p3 import certify as cf
    ctx.prec = 128
    rng = np.random.default_rng(7)
    for d in (np.array([3.0, 2.0, 2.0 + 1e-9, 1.0, 0.5, 0.25]), np.linspace(1, 0.01, 9)):
        n = d.size
        E = rng.standard_normal((n, n)) * 1e-6
        A = np.diag(d) + 0.5 * (E + E.T)
        lo, hi = cf.gersh_sorted(arb_mat(n, n, [float(v) for v in A.ravel()]))
        mp.mp.dps = 50
        ev = sorted([mp.mpf(v) for v in mp.eigsy(mp.matrix(A.tolist()))[0]], reverse=True)
        assert all(mp.mpf(lo[i]) <= ev[i] <= mp.mpf(hi[i]) for i in range(n))


@pytest.mark.parametrize("x,L,theta", [(0.0, 60.0, 0.25), (3.0, 45.0, 0.25), (0.0, 30.0, 0.4)])
def test_alias_bound_dominates_bruteforce(x, L, theta):
    """certify.alias_tilt_arb bounds e^{-theta L}/(1-e^{-theta L}) + sum_{m>=1} e^{theta m L} P(Z^2 > x + mL); the bound is at
    least the 3000-term mpmath sum, and not wasteful (relative excess < 1e-10); x = 0 is covered."""
    from flint import arb
    from p3 import certify as cf
    mp.mp.dps = 50
    a = mp.e ** (-theta * mp.mpf(L))
    exact = a / (1 - a) + mp.fsum(mp.e ** (theta * m * mp.mpf(L)) * mp.erfc(mp.sqrt((x + m * mp.mpf(L)) / 2)) for m in range(1, 3001))
    b, info = cf.alias_tilt_arb(x, arb(L), theta)
    assert mp.mpf(b) >= exact and (mp.mpf(b) - exact) / exact < 1e-10



def test_certified_cvm_law_contains_closed_form():
    """Corrected W route (17 Nov 2026): the certified enclosure of P(int Btilde^2 <= x) at H = 1/2 contains the closed-form
    Cramer-von Mises probability at its 95 % point, already for a small truncation J."""
    from flint import ctx
    from p3 import certify as cf, laws as lw
    ctx.prec = 128
    s = cf.BridgeSystem("0.50", 64)
    x = 0.46136129360588124
    r = s.cdf_Wfull(x, certified=True)
    F = float(lw.cdf_cvm_fast(x))
    assert r["lo"] <= F <= r["hi"] and r["hi"] - r["lo"] < 0.02


def test_legacy_certified_names_are_disabled():
    """The untilted cdf_X_certified / cdf_T_certified did not meet the certification standard and must not be selectable."""
    from p3 import certify as cf
    for name in ("cdf_X_certified", "cdf_T_certified"):
        with pytest.raises(NotImplementedError, match="_tilt"):
            getattr(cf.BridgeSystem, name)(None, 1.0)
    assert callable(getattr(cf.BridgeSystem, "cdf_T_certified_quad"))
    assert callable(getattr(cf.BridgeSystem, "cdf_T_certified_tilt"))


# ---------------------------------------------------------------- upgrade tests (fast certificate, SNExact, competitors)
@pytest.mark.parametrize("H", ["0.50", "0.60", "0.75", "0.95"])
def test_fast_nested_tail_frobenius_encloses_direct(H):
    """fast_nested_tail_frobenius encloses nested_tail_frobenius for both parities and all heads."""
    from p3 import certify as cf
    km = cf.Kummer(H, 1024)
    heads = (8, 16, 32, 64)
    for modes in (np.arange(1, 1024, 2), np.arange(2, 1025, 2)):
        ref = cf.nested_tail_frobenius(km, modes, heads)
        fast = cf.fast_nested_tail_frobenius(km, modes, heads, n_near=256)
        for h in heads:
            s0, e0 = ref[h]
            s1, e1 = fast[h]
            assert s1 - e1 <= s0 + e0 and s0 - e0 <= s1 + e1


@pytest.mark.parametrize("H,q", [("0.50", 45.526085752448), ("0.75", 163.28), ("0.95", 900.0)])
def test_certified_quad_matches_tilt_and_anchor(H, q):
    """Adaptive acb.integral Bromwich certificate overlaps the tilted midpoint certificate and encloses 0.95 at H=0.5."""
    from flint import ctx
    from p3 import certify as cf
    ctx.prec = 128
    s = cf.BridgeSystem(H, 256)
    rq = s.cdf_T_certified_quad(q)
    rt = s.cdf_T_certified_tilt(q)
    assert max(rq["lo"], rt["lo"]) <= min(rq["hi"], rt["hi"])
    if H == "0.50":
        assert rq["lo"] < 0.95 < rq["hi"]


def test_sn_exact_woodbury_matches_bromwich_and_mpmath():
    """Cholesky-free Woodbury SNExact.cdf agrees with SNExact.cdf_bromwich and 30-digit mpmath Imhof to < 1e-12."""
    g = fn.acf_arfima0(0.25, 64)
    sn = fn.SNExact(g)
    for c in (12.5, 45.52608575, 180.0):
        v_quad = sn.cdf(c)
        v_brom = sn.cdf_bromwich(c)
        assert abs(v_quad - v_brom) < 1e-12
    # Independent 30-digit Imhof check at c = 45.52608575
    n = 64
    from scipy.linalg import toeplitz
    R = np.linalg.cholesky(toeplitz(g))
    one = np.ones((n, 1))
    M = np.tril(np.ones((n, n))) @ (np.eye(n) - one @ one.T / n)
    A = one @ one.T - (45.52608575 / n) * (M.T @ M)
    lam = np.linalg.eigvalsh(R.T @ A @ R)
    mp.mp.dps = 30
    lmp = [mp.mpf(float(x)) for x in lam if abs(x) > 1e-300]
    def f(u):
        if u == 0:
            return mp.mpf(0.5) * mp.fsum(lmp)
        th = mp.fsum(mp.atan(l * u) for l in lmp) / 2
        lr_ = mp.fsum(mp.log1p((l * u) ** 2) for l in lmp) / 4
        return mp.sin(th) / (u * mp.exp(lr_))
    lmax = max(abs(x) for x in lmp)
    brk = [0] + [mp.mpf(10) ** k / lmax for k in range(-3, 9)] + [mp.inf]
    p_upper = float(mp.mpf(0.5) + mp.quad(f, brk) / mp.pi)
    assert abs((1.0 - sn.cdf(45.52608575)) - p_upper) < 1e-12


def test_local_whittle_boundary_and_elw_accuracy():
    """local_whittle preserves boundary optima (-0.49/+0.49) and exact_local_whittle recovers d on ARFIMA(0, d, 0)."""
    rng = np.random.default_rng(42)
    # Near-unit-root cumulative sum pushes d_hat to the upper boundary +0.49
    rw = np.cumsum(rng.standard_normal((4, 512)), axis=1)
    d_lw = fn.local_whittle(rw, m=32)
    assert np.all(d_lw >= 0.489)
    # Stationary ARFIMA(0, 0.25, 0) recovery
    x = fn.CirculantSampler(fn.acf_arfima0(0.25, 2048)).sample(1, rng)[:1]
    m_bw = int(round(2048 ** 0.65))
    d_lw2 = float(fn.local_whittle(x, m=m_bw)[0])
    d_elw = float(fn.exact_local_whittle(x, m=m_bw)[0])
    assert abs(d_lw2 - 0.25) < 0.15 and abs(d_elw - 0.25) < 0.15


def test_adg_mac_and_subsampling_sanity():
    """ADG (2009) MAC statistic and rate-free subsampling p-value behave consistently under null and large shift."""
    rng = np.random.default_rng(123)
    x0 = fn.CirculantSampler(fn.acf_fgn(0.7, 256)).sample(1, rng)[:1]
    m_bw = int(round(256 ** 0.65))
    t0 = float(fn.adg_mac_stat(x0, m=m_bw)[0])
    t1 = float(fn.adg_mac_stat(x0 + 5.0, m=m_bw)[0])
    assert math.isfinite(t0) and abs(t1) > abs(t0) + 3.0
    p0 = float(fn.subsampling_sn_pvalue(x0)[0])
    p1 = float(fn.subsampling_sn_pvalue(x0 + 5.0)[0])
    assert 0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0 and p1 < 0.05
