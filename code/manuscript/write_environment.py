"""Record the computing environment and checksums of all result files (17 Nov 2026, audit D4).
Writes results/ENVIRONMENT.json and results/SHA256SUMS.txt (backups, temporary and log files excluded).
Usage: python write_environment.py
"""
import hashlib, importlib.metadata as md, json, os, platform, re, sys, time  # noqa: E401
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); RES = os.path.join(WORK, "results")
PKG = ["numpy", "scipy", "pandas", "mpmath", "python-flint", "matplotlib", "gx2", "pytest"]
env = {"recorded": time.strftime("%Y-%m-%d %H:%M"), "python": sys.version.split()[0], "implementation": platform.python_implementation(),
       "platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()}
pk = {}
for p in PKG:
    try:
        pk[p] = md.version(p)
    except md.PackageNotFoundError:
        pk[p] = None
env["packages"] = pk
try:
    import numpy as np
    cfg = np.show_config(mode="dicts")
    env["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {})
except Exception as e:  # noqa: BLE001
    env["numpy_blas"] = repr(e)
m5 = os.path.join(RES, "E5_methods_meta.json")
if os.path.exists(m5):
    e5 = json.load(open(m5))["environment"]
    env["R"] = {k: e5.get(k) for k in ("R", "CompQuadForm", "momentchi2", "survey")}
    env["threads_in_E5_timing"] = e5.get("threads")
lg = os.path.join(WORK, "manuscript", "main.log")
if os.path.exists(lg):
    first = open(lg, encoding="utf-8", errors="replace").readline().strip()
    env["tex"] = first[:200]
env["note"] = ("E0(d), E2r, E3c, E5 and E7b pinned OpenMP/BLAS to one thread; the other runs used library defaults. Results do not "
               "depend on the thread count beyond floating-point rounding, and certified enclosures are unaffected.")
json.dump(env, open(os.path.join(RES, "ENVIRONMENT.json"), "w"), indent=1)
skip = re.compile(r"(\.bak|\.tmp|tmp_|\.log$)")
lines = []
for root, _, files in os.walk(RES):
    for f in sorted(files):
        p = os.path.join(root, f)
        if skip.search(f) or f == "SHA256SUMS.txt":
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {os.path.relpath(p, RES).replace(os.sep, '/')}")
open(os.path.join(RES, "SHA256SUMS.txt"), "w", encoding="utf-8").write("\n".join(sorted(lines, key=lambda s: s[66:])) + "\n")
print("ENVIRONMENT.json and SHA256SUMS.txt written;", len(lines), "files hashed")
