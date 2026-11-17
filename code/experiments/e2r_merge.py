"""E2r merge (17 Nov 2026): consolidate the recomputation with the final certificate into the result files the manuscript
reads. Requires every per-H record of e2r_revalidate.py (parts a-d) and e2r_print.py (part e, version 2).
  E2_sn_critical_values.json : F at the bracket ends, bracket flag, certified size of the printed value (old values kept, *_16nov)
  E9_surface_certified.json  : rebuilt from part (c) (exact printed surface expression, held-out odd hundredths)
  E6_naive_sn_*.json, E6b_rescaled_sn_*.json : certified 5 % sizes at J = 4096 (old values kept, *_16nov)
  E2r/SUMMARY.json           : counts and largest changes
Every file written is first copied to <file>.bak17nov.  Usage: python e2r_merge.py
"""
import glob, json, os, shutil, sys  # noqa: E401
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); RES = os.path.join(WORK, "results")
E2f = os.path.join(RES, "E2_sn_critical_values.json"); E2 = json.load(open(E2f)); cells = E2["cells"]
Hs = sorted(cells, key=float)
rb = {h: json.load(open(os.path.join(RES, "E2r", f"H{h}.json"))) for h in Hs}
rp = {h: json.load(open(os.path.join(RES, "E2r", "print", f"H{h}.json"))) for h in Hs}
assert all(r.get("version") == 2 for r in rp.values()), "print records incomplete"
r4f = {os.path.basename(f)[1:-5]: json.load(open(f)) for f in glob.glob(os.path.join(RES, "E2r", "J4096", "H*.json"))}


def backup(f):
    if not os.path.exists(f + ".bak17nov"):
        shutil.copy2(f, f + ".bak17nov")


summ = {"n_brackets": 0, "n_ok": 0, "max_abs_change_F_bracket": 0.0, "max_abs_change_size_print": 0.0, "max_size_print_err_pp": 0.0}
for h in Hs:
    for p, c in cells[h].items():
        n = rb[h]["sn"][p]; q = rp[h]["p"][p]
        summ["n_brackets"] += 1; summ["n_ok"] += int(n["bracket_ok"])
        for a, b in zip(n["F_at_q_lo"] + n["F_at_q_hi"], c["F_at_q_lo"] + c["F_at_q_hi"]):
            summ["max_abs_change_F_bracket"] = max(summ["max_abs_change_F_bracket"], abs(a - b))
        for a, b in zip(q["size_at_print"], c["size_at_print"]):
            summ["max_abs_change_size_print"] = max(summ["max_abs_change_size_print"], abs(a - b))
        alpha = 1 - float(p)
        summ["max_size_print_err_pp"] = max(summ["max_size_print_err_pp"], 100 * max(abs(q["size_at_print"][0] - alpha), abs(q["size_at_print"][1] - alpha)))
        c.setdefault("F_at_q_lo_16nov", c["F_at_q_lo"]); c.setdefault("F_at_q_hi_16nov", c["F_at_q_hi"])
        c.setdefault("size_at_print_16nov", c["size_at_print"])
        c["F_at_q_lo"], c["F_at_q_hi"] = n["F_at_q_lo"], n["F_at_q_hi"]
        c["certified_bracket"] = bool(n["bracket_ok"])
        c["size_at_print"] = q["size_at_print"]
        c["print_decimal"], c["print_neighbours"] = q["decimal"], [q["lo_pt"], q["hi_pt"]]
        c["certificate"] = "final (17 Nov 2026): Arb scalar assembly, Gershgorin eigenvalue bounds"
assert summ["n_ok"] == summ["n_brackets"], "a bracket failed"
backup(E2f); json.dump(E2, open(E2f + ".tmp", "w"), indent=1); os.replace(E2f + ".tmp", E2f)
# E9b certified surface
E9f = os.path.join(RES, "E9_surface_certified.json"); old9 = json.load(open(E9f))
c9 = []
for h in Hs:
    for p, v in rb[h].get("e9b", {}).items():
        c9.append({"H": h, "p": p, "q_hat_lo": v["q_hat_lo"], "q_hat_hi": v["q_hat_hi"], "route": v["route"], "F_lo": v["F_lo"], "F_hi": v["F_hi"],
                   "bound_pp": v["bound_pp"]})
assert len(c9) == 150, len(c9)
mx = max(c9, key=lambda r: r["bound_pp"])
new9 = {"digits": old9["digits"], "J": 16384, "t": old9.get("t", 25), "tilt": old9.get("tilt"), "target": "exact real value of the printed expression (ball arithmetic)",
        "cells": c9, "max_bound_pp": mx["bound_pp"], "argmax": [mx["H"], mx["p"]],
        "max_bound_pp_by_p": {p: max(r["bound_pp"] for r in c9 if r["p"] == p) for p in sorted({r["p"] for r in c9})},
        "n_direct": sum(r["route"] == "direct certificate" for r in c9), "max_bound_pp_16nov": old9["max_bound_pp"],
        "certificate": "final (17 Nov 2026)"}
backup(E9f); json.dump(new9, open(E9f + ".tmp", "w"), indent=1); os.replace(E9f + ".tmp", E9f)
summ["e9b_max_bound_pp"] = mx["bound_pp"]; summ["e9b_argmax"] = [mx["H"], mx["p"]]; summ["e9b_n_direct"] = new9["n_direct"]
summ["e9b_max_bound_pp_16nov"] = old9["max_bound_pp"]
# E6 / E6b
ch6 = 0.0; n6 = 0
for pat, key in (("E6_naive_sn_*.json", "E6"), ("E6b_rescaled_sn_*.json", "E6b")):
    for f in glob.glob(os.path.join(RES, pat)):
        d = json.load(open(f)); touched = False
        for h, rec in d.items():
            if not isinstance(rec, dict) or "0.05" not in rec or "size_lo" not in rec["0.05"]:
                continue
            r = r4f.get(h, {}).get(key)
            assert r is not None, f"missing J4096 revalidation for {key} H={h}"
            e = rec["0.05"]; e.setdefault("size_lo_16nov", e["size_lo"]); e.setdefault("size_hi_16nov", e["size_hi"])
            ch6 = max(ch6, abs(r["size_lo"] - e["size_lo"]), abs(r["size_hi"] - e["size_hi"]))
            e["size_lo"], e["size_hi"] = r["size_lo"], r["size_hi"]; e["certificate"] = "final (17 Nov 2026)"; touched = True; n6 += 1
        if touched:
            backup(f); json.dump(d, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
summ["e6_n_updated"] = n6; summ["e6_max_abs_change"] = ch6
t4 = rb["0.75"].get("t4"); summ["t4"] = t4
json.dump(summ, open(os.path.join(RES, "E2r", "SUMMARY.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
