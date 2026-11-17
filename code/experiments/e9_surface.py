"""E9: response surface for the certified T_H quantiles, with held-out validation in size units (protocol AT3).

Model, for each level p:  log q_p(H) = -log(1 - H) + sum_{i=0}^K a_{p,i} (H - 1/2)^i      (rate q ~ C/(1-H), Prop. rho)
Training: even hundredths H = 0.50, 0.52, ..., 0.98;  validation: odd hundredths 0.51, ..., 0.99 (never used in the fit).
Size error of a fitted value q_hat at a held-out H:  |F_H(q_hat) - p| ~= f_H(q) |q_hat - q|  (certified density f from the
production cell; the second-order term is below 1e-3 of the first at these accuracies).  K in 2..12 chosen by leave-one-out CV on the
training points only.  Domain of validity: 0.50 <= H <= 0.99 only.     Output: results/E9_surface.json
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE))
tab = json.load(open(os.path.join(WORK, "results", "E2_sn_critical_values.json")))["cells"]
Hs = sorted(tab, key=float)
assert float(Hs[-1]) >= 0.99 - 1e-9 and len(Hs) == 50, "needs the complete table H = 0.50..0.99"
out = {"domain": [0.50, 0.99], "fits": {},
       "protocol": "Common degree K* = smallest K whose leave-one-out CV error on the training (even-hundredth) points is below 1/10 of the training certification floor at every level; odd hundredths never used "
                   "for fitting or selection. Two errors at held-out H: (i) fit error vs the tabulated point quantile, "
                   "f |q_hat - q|; (ii) guaranteed error vs the true law, f (|q_hat - q| + max(q_hi - q, q - q_lo)), "
                   "q_true in the certified bracket [q_lo, q_hi]."}
KMAX = 12
def fit(x, y, K):
    return np.linalg.lstsq(np.vander(x, K + 1, increasing=True), y, rcond=None)[0]
cv_all = {}
for p in ("0.800", "0.900", "0.950", "0.975", "0.980", "0.990"):
    H = np.array([float(h) for h in Hs]); q = np.array([tab[h][p]["q"] for h in Hs]); f = np.array([tab[h][p]["density"] for h in Hs])
    qlo = np.array([tab[h][p]["q_lo"] for h in Hs]); qhi = np.array([tab[h][p]["q_hi"] for h in Hs])
    y = np.log(q) + np.log(1 - H); x = H - 0.5
    tr = np.array([round(100 * h) % 2 == 0 for h in H]); te = ~tr
    itr = np.where(tr)[0]; cv = {}
    for K in range(2, KMAX + 1):                     # leave-one-out on training points (size units)
        e = []
        for j in itr:
            keep = itr[itr != j]; c_ = fit(x[keep], y[keep], K)
            qh_j = np.exp(np.polyval(c_[::-1], x[j])) / (1 - H[j]); e.append(100 * f[j] * abs(qh_j - q[j]))
        cv[K] = max(e)
    hw = np.maximum(qhi - q, q - qlo)
    floor_tr = float(100 * np.max(f[tr] * hw[tr]))       # certification floor on the training points (no held-out data used)
    cv_all[p] = (cv, floor_tr, H, q, f, qlo, qhi, x, y, tr, te)
for p, (cv, floor_tr, H, q, f, qlo, qhi, x, y, tr, te) in cv_all.items():
    pass
KSTAR = min(K for K in range(2, KMAX + 1) if all(cv_all[p][0][K] < cv_all[p][1] / 10 for p in cv_all))
print("common degree K* (LOO error < 1/10 of the training certification floor at every level):", KSTAR, flush=True)
for p, (cv, floor_tr, H, q, f, qlo, qhi, x, y, tr, te) in cv_all.items():
    K = KSTAR; coef = fit(x[tr], y[tr], K)
    qh = np.exp(np.vander(x, K + 1, increasing=True) @ coef) / (1 - H)
    hw = np.maximum(qhi - q, q - qlo)
    fit_te = 100 * f[te] * np.abs(qh[te] - q[te]); bound_te = 100 * f[te] * (np.abs(qh[te] - q[te]) + hw[te])
    out["fits"][p] = {"K": int(K), "coef": coef.tolist(), "loo_cv_err_pp": {str(k): float(v) for k, v in cv.items()},
                      "certification_floor_train_pp": floor_tr,
                      "max_heldout_fit_err_pp": float(fit_te.max()), "max_heldout_guaranteed_err_pp": float(bound_te.max()),
                      "argmax_guaranteed_H": float(H[te][np.argmax(bound_te)]),
                      "max_heldout_rel_err_vs_point": float(np.max(np.abs(qh[te] / q[te] - 1))),
                      "max_train_fit_err_pp": float(100 * np.max(f[tr] * np.abs(qh[tr] - q[tr])))}
    r_ = out["fits"][p]
    print(p, "K=%d  heldout fit %.2e pp  guaranteed %.2e pp (H=%.2f)  rel %.1e" % (K, r_["max_heldout_fit_err_pp"], r_["max_heldout_guaranteed_err_pp"],
          r_["argmax_guaranteed_H"], r_["max_heldout_rel_err_vs_point"]), flush=True)
out["K"] = max(r["K"] for r in out["fits"].values())
for p, r in out["fits"].items():            # pad to a common K for the table (zeros beyond the fitted degree)
    r["coef"] = r["coef"] + [0.0] * (out["K"] + 1 - len(r["coef"]))
out["max_heldout_fit_err_pp"] = max(r["max_heldout_fit_err_pp"] for r in out["fits"].values())
out["max_heldout_guaranteed_err_pp"] = max(r["max_heldout_guaranteed_err_pp"] for r in out["fits"].values())
json.dump(out, open(os.path.join(WORK, "results", "E9_surface.json"), "w"), indent=1)
print("max held-out fit error (pp): %.3e ; guaranteed (incl. certification) (pp): %.3e" % (out["max_heldout_fit_err_pp"], out["max_heldout_guaranteed_err_pp"]))
