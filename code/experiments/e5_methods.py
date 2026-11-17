"""Step 7: computational methods on identical targets (protocol P9); revised 17 Nov 2026 (audit items B1-B4).

Targets (each a distribution function P(X <= y) of a Gaussian quadratic form with exactly one positive weight):
  T1  HI fixed-m limit, H = 0.80, m = 8, at the computed two-sided 5% point squared (17 weights; reference: certified enclosure)
  T2  EWC limit, H = 0.80, B = 16, at its computed two-sided 5% point squared (17 weights; reference: certified enclosure)
  T3  exact finite-n SN law, ARFIMA(0, 0.2, 0), n = 1000, at the computed limit 5% point of T_0.7 (1000 weights;
      reference: finite_n.SNExact, computed in double precision and cross-checked by the accurate direct methods)
  T4  limit law T_H, H = 0.75, at the J = 16384 point q of the pinned record results/E1_benchmark_H0.75.json; reference:
      that record's certified enclosure, or the enclosure recomputed by the 17 Nov revalidation (results/E2r/H0.75.json)
      when present. Competitors receive the Galerkin ratio-form eigenvalues for J = 512 and 2048 plus the exact mean of the
      omitted remainder as a shift y = q tau_J, so their values also carry the Galerkin truncation error of that J.
Cost accounting (B1). The covariance of the fixed-smoothing coordinates (T1, T2) is a shared input and is excluded for every
method; each method is charged its own spectral step (certified: arb weight enclosure; competitors: double Cholesky + eigvalsh).
T3: competitors are charged Cholesky + eigenvalues of the cutoff-dependent ratio form; finite_n.SNExact is charged its one-off
setup (Cholesky + one eigh) and its query; first and repeated queries are timed separately, and an amortisation run over
m cutoffs (competitors redo the eigenvalues at every cutoff) is written to the meta file. T4: every route pays its own Gram build.
Errors (B4). Against an enclosure [L, U] each value v gets the interval [dist(v, [L, U]), max(|v - L|, |v - U|)] that contains
its true error; against the computed T3 reference the plain difference is reported.
Domains (B2). CompQuadForm::liu is documented for non-negative definite forms; on these indefinite forms it is run, labelled
outside its documented domain, and excluded from comparisons. Gauss-Hermite rows are outer-quadrature diagnostics.
Threads. OMP/MKL/OpenBLAS/NumExpr threads are pinned to 1 before numpy is imported; versions and hardware are recorded.
Usage: python e5_methods.py  -> results/E5_methods.csv, results/E5_methods_meta.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import csv            # noqa: E402
import glob           # noqa: E402
import hashlib        # noqa: E402
import json           # noqa: E402
import math           # noqa: E402
import platform       # noqa: E402
import shutil         # noqa: E402
import subprocess     # noqa: E402
import sys            # noqa: E402
import time           # noqa: E402

import numpy as np                       # noqa: E402
import scipy                             # noqa: E402
from scipy import integrate, stats       # noqa: E402
from scipy.optimize import brentq        # noqa: E402
from scipy.special import log_ndtr       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(WORK, "code"))
from p3 import laws as lw, lowrank as lr, finite_n as fn, certify as cf, gram as gr  # noqa: E402
import gx2  # noqa: E402
from flint import ctx  # noqa: E402


def find_rscript():
    """Rscript from $RSCRIPT, then PATH, then $R_HOME/bin, then a sibling conda environment named 'r'
    (located relative to this interpreter: <envs>/<this env>/python.exe -> <envs>/r)."""
    for cand in [os.environ.get("RSCRIPT"), shutil.which("Rscript"),
                 os.path.join(os.environ.get("R_HOME", ""), "bin", "Rscript.exe") if os.environ.get("R_HOME") else None,
                 os.path.join(os.environ.get("R_HOME", ""), "bin", "Rscript") if os.environ.get("R_HOME") else None]:
        if cand and os.path.exists(cand):
            return cand
    anchors = [os.environ.get("CONDA_PREFIX"), sys.base_prefix, sys.prefix, os.path.dirname(os.path.abspath(sys.executable))]
    for a in anchors:
        if not a:
            continue
        envs = os.path.dirname(os.path.abspath(a))
        for rel in (("r", "Lib", "R", "bin", "Rscript.exe"), ("r", "Scripts", "Rscript.exe"), ("r", "bin", "Rscript")):
            cand = os.path.join(envs, *rel)
            if os.path.exists(cand):
                return cand
    return None


RSCRIPT = find_rscript()


def r_env():
    """Environment for Rscript: a conda R on Windows needs its prefix's DLL directories on PATH."""
    env = dict(os.environ)
    if RSCRIPT:
        p = os.path.abspath(RSCRIPT)
        prefix = p.split(os.sep + "Lib" + os.sep + "R" + os.sep)[0] if (os.sep + "Lib" + os.sep + "R" + os.sep) in p \
            else os.path.dirname(os.path.dirname(p))
        extra = [os.path.join(prefix, *d) for d in (("Library", "mingw-w64", "bin"), ("Library", "usr", "bin"),
                                                     ("Library", "bin"), ("Scripts",), ("Lib", "R", "bin", "x64"))]
        env["PATH"] = os.pathsep.join([d for d in extra if os.path.isdir(d)] + [env.get("PATH", "")])
    return env
SHARED_LOAD = os.environ.get("E5_LOAD_NOTE", "no other benchmark jobs running (recorded by the operator)")


def timed(f, reps=3):
    """value, best and median wall time over reps calls."""
    ts, val = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        val = f()
        ts.append(time.perf_counter() - t0)
    return val, min(ts), float(np.median(ts))


# ---------------------------------------------------------------- methods on explicit weights
def bromwich_explicit(th, y, theta=0.25, eps=1e-14):
    """Tilted Bromwich midpoint rule for P(sum th_i chi2_1 <= y), one positive weight th_+ <= 1/(4 theta) (computed)."""
    th = np.asarray(th, float)
    scale = th.max()                     # the event {X <= y} is invariant under (th, y) -> (th, y)/scale
    th, y = th / scale, y / scale
    tp = 1.0
    L = 20.0
    while True:
        a = math.exp(-theta * L)
        al = a / (1 - a) + sum(math.exp(theta * m * L + math.log(2) + float(log_ndtr(-math.sqrt(max(y + m * L, 0) / tp))))
                               for m in range(1, 60))
        if al < eps:
            break
        L *= 1.05
    D = 2 * math.pi / L
    neg = -th[th < 0]
    K = 64
    while True:
        s = (K - 0.5) * D
        q = 2 * s * neg
        sel = q[q > 1]
        if sel.size and (math.exp(-theta * y) / math.sqrt(1 - 2 * theta * tp) * 2 / (math.pi * sel.size)
                         * math.exp(-0.5 * np.sum(np.log(sel)))) < eps:
            break
        K = int(K * 1.1) + 1
    sg = (np.arange(K) + 0.5) * D
    w = theta + 1j * sg
    logM = -0.5 * np.log(1.0 - 2.0 * w[:, None] * th[None, :]).sum(axis=1)
    P = D / math.pi * math.fsum(np.real(np.exp(logM - w * y) / w).tolist())
    return 1.0 - P


def saddlepoint_lr(th, y):
    """Lugannani-Rice saddlepoint approximation (Kuonen 1999) for P(sum th_i chi2_1 <= y)."""
    th = np.asarray(th, float)
    K = lambda t: -0.5 * np.sum(np.log1p(-2 * t * th))
    K1 = lambda t: np.sum(th / (1 - 2 * t * th))
    K2 = lambda t: np.sum(2 * th * th / (1 - 2 * t * th) ** 2)
    lo = 1 / (2 * th.min()) * (1 - 1e-12) if th.min() < 0 else -1e6
    hi = 1 / (2 * th.max()) * (1 - 1e-12)
    t = brentq(lambda t: K1(t) - y, lo, hi, xtol=1e-15)
    wv = np.sign(t) * math.sqrt(2 * (t * y - K(t)))
    u = t * math.sqrt(K2(t))
    return stats.norm.cdf(wv) + stats.norm.pdf(wv) * (1 / wv - 1 / u)


def imhof_cdf_df(w, df, y, epsabs=1e-13, epsrel=1e-12, limit=4000):
    """P(sum_i w_i chi2_{df_i} <= y) for real (possibly fractional) df_i > 0: Imhof (1961) with h_i = df_i (computed)."""
    w = np.asarray(w, float)
    df = np.asarray(df, float)

    def f(u):
        a = w * u
        ang = 0.5 * np.sum(df * np.arctan(a)) - 0.5 * y * u
        lrho = 0.25 * np.sum(df * np.log1p(a * a))
        return math.sin(ang) * math.exp(-lrho) / u
    val, _ = integrate.quad(f, 0.0, np.inf, limit=limit, epsabs=epsabs, epsrel=epsrel)
    return 0.5 - val / math.pi


def leading_plus_satterthwaite(th, y, k=20):
    """Chen-Lumley-type surrogate: the positive weight and the k largest negative weights exactly; the remaining negative
    weights (all of one sign, checked) replaced by the two-moment matched -g chi2_nu with fractional nu (Satterthwaite):
    g = sum r^2 / sum r, nu = (sum r)^2 / sum r^2, r = -rest; the reduced form is evaluated by Imhof with fractional df."""
    th = np.asarray(th, float)
    pos, neg = th[th > 0], th[th < 0]
    neg = neg[np.argsort(neg)]                     # most negative first
    head = np.concatenate([pos, neg[:k]])
    rest = neg[k:]
    if rest.size == 0:
        return lw.imhof_cdf(head, y)[0]
    assert np.all(rest < 0), "Satterthwaite remainder must be of one sign"
    r = -rest
    g = float(np.sum(r * r) / np.sum(r))
    nu = float(np.sum(r) ** 2 / np.sum(r * r))
    return imhof_cdf_df(np.concatenate([head, [-g]]), np.concatenate([np.ones(head.size), [nu]]), y)


def monte_carlo(th, y, N, rng):
    th = np.asarray(th, float)
    hits, done, chunk = 0, 0, max(1, 4_000_000 // th.size)
    while done < N:
        n = min(chunk, N - done)
        X = (rng.standard_normal((n, th.size)) ** 2) @ th
        hits += int(np.sum(X <= y))
        done += n
    p = hits / N
    return p, math.sqrt(p * (1 - p) / N)


FAMILY = {"Imhof": "numerical inversion", "Bromwich": "numerical inversion", "gx2 imhof": "numerical inversion",
          "gx2 ifft": "numerical inversion", "gx2 ray": "ray-trace integration", "gx2 pearson": "moment approximation",
          "gx2 auto": "numerical inversion", "Ruben": "series expansion", "Lugannani": "saddlepoint approximation",
          "leading-20": "moment approximation", "Monte Carlo": "simulation", "davies": "numerical inversion",
          "CompQuadForm::imhof": "numerical inversion", "liu": "moment approximation", "hbe": "moment approximation",
          "lpb4": "moment approximation", "saddlepoint": "saddlepoint approximation", "satterthwaite": "moment approximation",
          "certified": "certified enclosure", "exact law": "numerical inversion"}


def family_of(name):
    for k, v in FAMILY.items():
        if k in name:
            return v
    return ""


def err_interval(v, ref):
    """ref = (L, U, kind). Returns (err_lo, err_hi): interval containing |v - true value|, or |v - ref| twice."""
    L, U, kind = ref
    if not (v == v):
        return math.nan, math.nan
    if kind == "enclosure":
        return max(L - v, 0.0, v - U), max(abs(v - L), abs(v - U))
    return abs(v - L), abs(v - L)


def add_row(rows, target, method, v, ref, sec, sec_med, t_spec, domain="documented domain", note=""):
    e_lo, e_hi = err_interval(v, ref)
    rows.append(dict(target=target, method=method, family=family_of(method), domain=domain, value=v,
                     ref_lo=ref[0], ref_hi=ref[1], ref_kind=ref[2], err_lo=e_lo, err_hi=e_hi,
                     seconds_method=sec, seconds_method_median=sec_med, seconds_spectrum=t_spec,
                     seconds_total=(sec + t_spec) if (sec == sec and t_spec == t_spec) else math.nan, note=note))


def run_python_methods(target, th, y, ref, t_spec, rows, rng, allow_ruben):
    def rec(name, f, reps=3, note=""):
        try:
            v, s, sm = timed(f, reps)
            if isinstance(v, tuple):
                v, note = v[0], f"MC s.e. {v[1]:.1e}"
            v = float(v)
            if not math.isfinite(v):
                note = (note + "; " if note else "") + "failed: non-finite value (overflow in double precision)"
            add_row(rows, target, name, v, ref, s, sm, t_spec, note=note)
        except Exception as e:  # noqa: BLE001
            add_row(rows, target, name, math.nan, ref, math.nan, math.nan, t_spec, note=repr(e)[:120])
    k_, l_ = np.ones_like(th), np.zeros_like(th)
    rec("Imhof (scipy quad; laws.imhof_cdf)", lambda: lw.imhof_cdf(th, y)[0])
    rec("Bromwich tilted midpoint, double (this paper)", lambda: bromwich_explicit(th, y))
    for meth in ("imhof", "ifft", "ray", "pearson", "auto"):
        rec(f"gx2 {meth} (Das 2025)", lambda meth=meth: float(gx2.cdf(y, th, k_, l_, 0.0, 0.0, method=meth)), reps=1)
    if allow_ruben:
        rec("Ruben-Beta mixture, double (laws.ruben_beta_cdf0)", lambda: lw.ruben_beta_cdf0(th)[0])
    rec("Lugannani-Rice saddlepoint (Kuonen 1999)", lambda: saddlepoint_lr(th, y))
    rec("leading-20 + Satterthwaite remainder, fractional df (Chen-Lumley type)", lambda: leading_plus_satterthwaite(th, y))
    N = 1_000_000 if th.size <= 64 else 100_000
    rec(f"Monte Carlo N = {N:,}", lambda: monte_carlo(th, y, N, rng), reps=1)


def run_r_methods(target, th, y, ref, t_spec, rows):
    if RSCRIPT is None:
        add_row(rows, target, "R baselines", math.nan, ref, math.nan, math.nan, t_spec, note="Rscript not found")
        return
    tmp = os.path.join(WORK, "results", "tmp_weights.csv")
    out = os.path.join(WORK, "results", "tmp_r_out.csv")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["theta"])
        w.writerows([[f"{x:.17g}"] for x in th])
    env = r_env()
    subprocess.run([RSCRIPT, os.path.join(HERE, "rq_methods.R"), tmp, f"{y:.17g}", out, "3"], check=False, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=7200)
    if not os.path.exists(out):
        add_row(rows, target, "R baselines", math.nan, ref, math.nan, math.nan, t_spec, note="R run failed")
        return
    with open(out, newline="") as fh:
        for r in csv.DictReader(fh):
            v = float(r["value"]) if r["value"] not in ("", "NA") else math.nan
            s = float(r["seconds"]) if r["seconds"] not in ("", "NA") else math.nan
            sm = float(r["seconds_median"]) if r.get("seconds_median", "") not in ("", "NA") else math.nan
            note = r.get("note", "")
            dom = "outside documented domain" if "outside documented domain" in note else (
                "diagnostic" if "diagnostic" in note else "documented domain")
            add_row(rows, target, r["method"], v, ref, s, sm, t_spec, domain=dom, note=note)
    os.remove(out)


def ratio_weights_from_cov(Sig, c, k):
    L = np.linalg.cholesky(Sig)
    d = np.full(Sig.shape[0], -c / k)
    d[0] = 1.0
    return np.sort(np.linalg.eigvalsh(L.T @ (d[:, None] * L)))[::-1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def environment_record():
    import mpmath
    import flint
    rec = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
               processor=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
               logical_cpus=os.cpu_count(), numpy=np.__version__, scipy=scipy.__version__, mpmath=mpmath.__version__,
               python_flint=getattr(flint, "__version__", "?"), gx2=getattr(gx2, "__version__", "?"),
               threads={v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                       "NUMEXPR_NUM_THREADS")},
               concurrent_load=SHARED_LOAD, rscript=RSCRIPT)
    try:
        cfg = np.show_config(mode="dicts")
        rec["blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
    except Exception:  # noqa: BLE001
        rec["blas"] = "?"
    if RSCRIPT:
        expr = ("cat(R.version.string, as.character(packageVersion('CompQuadForm')), as.character(packageVersion('momentchi2')),"
                " as.character(packageVersion('survey')), sep='|')")
        try:
            o = subprocess.run([RSCRIPT, "-e", expr], capture_output=True, text=True, timeout=120, env=r_env()).stdout.strip().split("|")
            rec.update(R=o[0], CompQuadForm=o[1], momentchi2=o[2], survey=o[3])
        except Exception as e:  # noqa: BLE001
            rec["R"] = f"version query failed: {e!r}"
    return rec


def t4_reference():
    """(q, L, U, source, seconds_build, seconds_eval) from the revalidation record if present, else the step-3 record."""
    e1p = os.path.join(WORK, "results", "E1_benchmark_H0.75.json")
    e1 = json.load(open(e1p))
    row = [r for r in e1["rows"] if r["J"] == 16384 and abs(r["p"] - 0.95) < 1e-12][0]
    q, L, U = row["q"], row["cert_lo"], row["cert_hi"]
    src, tb, te = f"results/E1_benchmark_H0.75.json (sha256 {sha256(e1p)[:16]})", row["build_s"], row["cert_s"]
    e2p = os.path.join(WORK, "results", "E2r", "H0.75.json")
    if os.path.exists(e2p):
        e2 = json.load(open(e2p))
        t4 = e2.get("t4")
        if t4 and t4["q"] == q:
            L, U = t4["F"]
            src = f"results/E2r/H0.75.json (17 Nov revalidation; sha256 {sha256(e2p)[:16]})"
            te = t4.get("seconds", te)
    return q, L, U, src, tb, te


if __name__ == "__main__":
    rng = np.random.default_rng(20261116)
    rows, meta = [], {"environment": environment_record(), "targets": {}}
    ctx.prec = 128
    # T1, T2: fixed-smoothing limits
    for target, kind, k in (("T1 HI m=8 H=0.80", "HI", 8), ("T2 EWC B=16 H=0.80", "EWC", 16)):
        t0 = time.perf_counter()
        law = lr.FixedSmoothingLaw("0.80", kind, k)
        t_cov = time.perf_counter() - t0
        c2 = law.quantile2(0.95)
        (th, ), t_spec, _ = timed(lambda: (ratio_weights_from_cov(law.Sd, c2, k), ), reps=3)
        (lo, hi), t_cert, t_cert_med = timed(lambda: law.cdf2_certified(c2), reps=3)
        ref = (lo, hi, "enclosure")
        add_row(rows, target, "certified (this paper: arb weights + Gershgorin + Ruben series)", 0.5 * (lo + hi), ref,
                t_cert, t_cert_med, 0.0, note=f"enclosure [{lo!r}, {hi!r}]; width {hi - lo:.3e}")
        run_python_methods(target, th, 0.0, ref, t_spec, rows, rng, allow_ruben=True)
        run_r_methods(target, th, 0.0, ref, t_spec, rows)
        meta["targets"][target] = dict(c2=c2, weights=th.size, seconds_shared_covariance_arb=t_cov,
                                       seconds_spectrum_double=t_spec, reference=[lo, hi])
        print(target, "done", flush=True)
    # T3: finite-n SN law
    n, d = 1000, 0.2
    g = fn.acf_arfima0(d, n)
    sysm = cf.BridgeSystem("0.70", 4096)
    c = brentq(lambda cc: sysm.cdf_T_computed_tilt(cc)["point"] - 0.95, 20, 200, xtol=1e-10)
    from scipy.linalg import cholesky, toeplitz
    P = np.full((n, n), 1.0 / n)
    E = np.eye(n) - P
    DE = np.cumsum(E, axis=0)
    DtD = DE.T @ DE

    def chol():
        return cholesky(toeplitz(g), lower=True)

    def ratio_eigs(Lc, cc):
        Mc = P - (cc / n ** 2) * DtD           # T_n <= c  <=>  u^T (11^T/n) u <= (c/n^2) u^T (DE)^T DE u
        th_ = np.sort(np.linalg.eigvalsh(Lc.T @ Mc @ Lc))[::-1]
        return th_[np.abs(th_) > 1e-14 * np.abs(th_).max()]
    Lc, t_chol, _ = timed(chol, reps=3)
    th, t_eig, _ = timed(lambda: ratio_eigs(Lc, c), reps=3)
    t_spec = t_chol + t_eig
    sn, t_setup, t_setup_med = timed(lambda: fn.SNExact(g), reps=3)
    sn_cold = fn.SNExact(g)
    t0 = time.perf_counter()
    ref_v = sn_cold.cdf(c)
    t_first = time.perf_counter() - t0
    _, t_rep, t_rep_med = timed(lambda: sn.cdf(1.01 * c), reps=5)
    ref = (ref_v, ref_v, "computed")
    tgt3 = "T3 SN n=1000 ARFIMA(0,0.2,0)"
    add_row(rows, tgt3, "exact law, one-eigh Bromwich (this paper; finite_n.SNExact) | setup + first query", ref_v, ref,
            t_first, t_first, t_setup, note="reference; setup = Cholesky + one eigh, shared by all cutoffs")
    add_row(rows, tgt3, "exact law, one-eigh Bromwich (this paper; finite_n.SNExact) | repeated query", math.nan,
            ref, t_rep, t_rep_med, 0.0, domain="timing only", note="query at a new cutoff (1.01 c) after setup; timing row")
    run_python_methods(tgt3, th, 0.0, ref, t_spec, rows, rng, allow_ruben=True)
    run_r_methods(tgt3, th, 0.0, ref, t_spec, rows)
    # amortisation over m cutoffs: SNExact (setup once) versus competitors (eigenvalues per cutoff)
    cuts = list(c * np.linspace(0.80, 1.20, 10))
    t0 = time.perf_counter()
    snA = fn.SNExact(g)
    vA = [snA.cdf(cc) for cc in cuts]
    t_ours = time.perf_counter() - t0
    amort = {"m": len(cuts), "cutoffs": cuts, "SNExact_total_s": t_ours, "SNExact_values": vA}
    for name, f in (("Imhof (scipy quad)", lambda w_: lw.imhof_cdf(w_, 0.0)[0]),
                    ("Bromwich tilted midpoint, double", lambda w_: bromwich_explicit(w_, 0.0)),
                    ("gx2 imhof", lambda w_: float(gx2.cdf(0.0, w_, np.ones_like(w_), np.zeros_like(w_), 0.0, 0.0,
                                                          method="imhof")))):
        t0 = time.perf_counter()
        Lx = chol()
        vals = [f(ratio_eigs(Lx, cc)) for cc in cuts]
        amort[name + "_total_s"] = time.perf_counter() - t0
        amort[name + "_max_abs_diff_vs_SNExact"] = float(max(abs(a - b) for a, b in zip(vals, vA)))
    meta["targets"][tgt3] = dict(c=c, weights=th.size, seconds_cholesky=t_chol, seconds_ratio_eigs=t_eig,
                                 SNExact_setup_s=t_setup, SNExact_setup_median_s=t_setup_med, SNExact_first_query_s=t_first,
                                 SNExact_repeat_query_s=t_rep, SNExact_repeat_query_median_s=t_rep_med,
                                 SNExact_clipped_negative_eigenvalues=int(getattr(sn, "n_clipped", -1)),
                                 SNExact_min_raw_eigenvalue=float(getattr(sn, "min_kappa_raw", math.nan)),
                                 amortisation=amort, reference=ref_v)
    print("T3 done", flush=True)
    # T4: the limit law T_H through Galerkin eigenvalues
    H = 0.75
    q, rlo, rhi, src, t_build, t_evalc = t4_reference()
    ref = (rlo, rhi, "enclosure")
    add_row(rows, "T4 T_H H=0.75 (limit)", "certified (this paper: J = 16384, tilted Bromwich in arb)", 0.5 * (rlo + rhi), ref,
            t_evalc, t_evalc, t_build, note=f"pinned record: {src}; build {t_build:.1f} s, evaluation {t_evalc:.1f} s")
    meta["targets"]["T4"] = dict(q=q, reference=[rlo, rhi], source=src, seconds_build=t_build, seconds_eval=t_evalc)
    for J in (512, 2048):
        def build(J=J):
            G = gr.cosine_gram(H, J + 1)
            kk = np.arange(1, J + 1)
            Sig = np.empty((J + 1, J + 1))
            Sig[0, 0] = 1.0
            Sig[0, 1:] = Sig[1:, 0] = G[0, 1:J + 1] / (kk * np.pi)
            Sig[1:, 1:] = G[1:J + 1, 1:J + 1] / np.outer(kk * np.pi, kk * np.pi)
            tau = gr.trace_bridge(H) - float(np.sum(np.diag(Sig)[1:]))
            return ratio_weights_from_cov(Sig, q, 1), tau
        (th, tau), t_spec, _ = timed(build, reps=1)
        tgt = f"T4 T_H H=0.75 via Galerkin J={J}"
        run_python_methods(tgt, th, q * tau, ref, t_spec, rows, rng, allow_ruben=False)
        run_r_methods(tgt, th, q * tau, ref, t_spec, rows)
        meta["targets"][tgt] = dict(J=J, tau=tau, shift=q * tau, weights=th.size, seconds_spectrum=t_spec)
        print(tgt, "done", flush=True)
    cols = ["target", "method", "family", "domain", "value", "ref_lo", "ref_hi", "ref_kind", "err_lo", "err_hi",
            "seconds_method", "seconds_method_median", "seconds_spectrum", "seconds_total", "note"]
    with open(os.path.join(WORK, "results", "E5_methods.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(WORK, "results", "E5_methods_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=float)
    print("wrote E5_methods.csv and E5_methods_meta.json", flush=True)
