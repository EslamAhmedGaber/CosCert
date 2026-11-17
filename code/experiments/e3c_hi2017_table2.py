"""E3c: exact finite-T rejection rates for Hualde & Iacone (2017) Table 2 (validation of the exact finite-n laws).

Design (HI 2017, Table 2): u_t = 0.5 u_{t-1} + e_t, e_t iid N(0,1) (delta = 0 known), T = 64 and 256, fixed-m statistic
tau (eq. 2), m = 1..16, two-sided 5 % tests with t_{2m} and with N(0,1) critical values; published: 10,000 replications.
Exact rates from finite_n.LowRankExact (C' Gamma C), compared with the ledger (z = (published - exact)/sqrt(e(1-e)/N)).
Output: results/E3c_hi2017_table2.csv
"""
import os, sys, csv, math
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CODE = os.path.dirname(HERE); WORK = os.path.dirname(CODE)
sys.path.insert(0, CODE)
from p3 import finite_n as fn  # noqa: E402

led = pd.read_csv(os.path.join(WORK, "literature", "ledger_rejection_rates.csv"))
pub = led[(led.source_id == "HI2017") & led.location.str.contains("Table 2")].copy()
rows = []
for T in (64, 256):
    g = fn.acf_ar1(0.5, T)
    for m in range(1, 17):
        lrk = fn.LowRankExact(g, "HI", m)
        for lab, cv in (("t2m", stats.t.ppf(0.975, 2 * m)), ("N01", stats.norm.ppf(0.975))):
            e = float(lrk.reject_two_sided(cv))
            sel = pub[(pub.n == T) & pub.ledger_id.str.endswith(f"-{lab}-m{m}")]
            p = float(sel.published_rate.iloc[0]) if len(sel) else float("nan")
            N = int(sel.replications_N.iloc[0]) if len(sel) else 10000
            z = (p - e) / math.sqrt(e * (1 - e) / N) if p == p else float("nan")
            rows.append([T, m, lab, cv, e, p, N, z, sel.ledger_id.iloc[0] if len(sel) else ""])
out = os.path.join(WORK, "results", "E3c_hi2017_table2.csv")
with open(out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["T", "m", "critical_value", "cv", "exact_rate", "published_rate", "N", "z", "ledger_id"]); w.writerows(rows)
D = pd.DataFrame(rows, columns=["T", "m", "lab", "cv", "e", "p", "N", "z", "id"])
print(f"cells with published value: {D.p.notna().sum()}  max|z| = {D.z.abs().max():.2f}  share |z|>2 = {(D.z.abs() > 2).mean():.3f}")
print("wrote", out)
