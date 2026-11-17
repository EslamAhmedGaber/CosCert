"""E8: spectrum of the fBm bridge covariance operator K_br (Type I fBm), normalised by the Theorem A scale.

Galerkin matrix in the bridge sine basis sqrt2 sin(k pi r): Cov(V_k, V_l) = G_kl / (k pi l pi) (Paper 1 cosine Gram),
eigenvalues lambda_k (k << J accurate), normalised nu_k = lambda_k (k pi)^(1 + 2H) / c_H,  c_H = Gamma(2H+1) sin(pi H).
Also records tr K_br (closed form) and the Galerkin trace deficit tau_J.   Output: results/E8_spectrum.json
"""
import os, sys, json, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import gram as gr  # noqa: E402


def spectrum(H, J):
    G = gr.cosine_gram(H, J + 1); kk = np.arange(1, J + 1)
    S = G[1:J + 1, 1:J + 1] / np.outer(kk * np.pi, kk * np.pi)
    lam = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))[::-1]
    cH = math.gamma(2 * H + 1) * math.sin(math.pi * H)
    return lam, lam * (kk * np.pi) ** (1 + 2 * H) / cH, float(gr.trace_bridge(H) - np.trace(S))


if __name__ == "__main__":
    out = {"J": 2048, "note": "Galerkin eigenvalues; reliable for k up to about J/4", "H": {}}
    for H in (0.5, 0.55, 0.75, 0.95):
        t0 = time.perf_counter(); lam, nu, tau = spectrum(H, 2048)
        out["H"][f"{H:.2f}"] = {"lambda": lam[:1024].tolist(), "nu": nu[:1024].tolist(), "tau_J": tau, "trace": float(gr.trace_bridge(H))}
        print(f"H={H}: nu_1={nu[0]:.4f} nu_10={nu[9]:.4f} nu_100={nu[99]:.5f} nu_500={nu[499]:.5f}  tau_J={tau:.3e}  {time.perf_counter() - t0:.1f}s", flush=True)
    json.dump(out, open(os.path.join(WORK, "results", "E8_spectrum.json"), "w"))
