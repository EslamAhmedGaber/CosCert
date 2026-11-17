"""Driver: E3b plug-in grid (4 models x H in .5..9), at most 3 concurrent processes (host RAM limit)."""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.dirname(os.path.dirname(HERE)); LOGS = os.path.join(WORK, "logs")
JOBS = [(m, h) for h in ("0.5", "0.6", "0.7", "0.8", "0.9") for m in ("fgn", "arfima0", "arfima1p", "arfima1m")]
def run(job):
    m, h = job; out = os.path.join(WORK, "results", f"E3b_plugin_v2_{m}_H{float(h):.1f}.csv")
    if os.path.exists(out): return job, "exists", 0.0
    t0 = time.time()
    with open(os.path.join(LOGS, f"E3b_plugin_v2_{m}_H{h}.log"), "w") as lf:
        rc = subprocess.run([sys.executable, "-u", os.path.join(HERE, "e3b_mc.py"), "plugin", m, h], cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
    return job, rc, time.time() - t0
with ThreadPoolExecutor(max_workers=3) as ex:
    for job, rc, dt in ex.map(run, JOBS):
        print(job, rc, f"{dt:.0f}s", flush=True)
print("ALL DONE", flush=True)
