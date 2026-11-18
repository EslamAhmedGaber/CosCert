# CosCert code and data

Code, numerical results and manuscript sources for *CosCert: certified critical values and exact size for long-memory
tests of the mean* (E. Ahmed, M.A. El-Beltagy, A. Etman, S.M. Elnady). Public repository:
https://github.com/EslamAhmedGaber/CosCert. No submission or publication status is asserted here.

| Folder | Content |
|---|---|
| `code/p3/` | library: `gram.py` (closed-form Gram matrices), `lowrank.py` (certified fixed-m and EWC limits), `certify.py` (CosCert: certified distribution function of T_H, `BridgeSystem.cdf_T_certified_tilt`, and of the Cramer-von Mises law), `finite_n.py` (exact finite-sample laws, circulant-embedding sampler, local Whittle), `laws.py` (Imhof, Ruben and reference laws) |
| `code/experiments/` | one driver per experiment E0-E12 (Table E.18 of the paper) and the revalidation scripts `e0d_revalidate.py`, `e2r_revalidate.py`, `e2r_print.py`, `e2r_merge.py` |
| `code/figures/` | one script per figure (matplotlib PGF backend, typeset with `manuscript/p3fonts.tex`) |
| `code/manuscript/` | `make_numbers.py` (writes every number and generated table of the paper), `build.ps1`, `write_environment.py` |
| `code/tests/` | unit tests (`python -m pytest code/tests -q`) |
| `results/` | every result file; `ENVIRONMENT.json` (versions, hardware) and `SHA256SUMS.txt` (checksums); `stored_runs_16nov/` holds the outputs of 16 November 2026 that the reproducibility check of Table E.19 compares with the re-runs |
| `literature/` | ledgers of the published critical values and rejection rates audited in Section 6 (inputs of experiment E4 and of `make_numbers.py`) |
| `figures/` | the figures as printed (vector PDF and PNG) |
| `manuscript/` | LaTeX source in its working layout (`main.tex`, `sections/`, `tables/`, `p3fonts.tex`, `references.bib`) |
| `PROTOCOL.md` | the experimental protocol with its dated amendments |

Requirements: Python 3.12 with NumPy, SciPy, pandas, mpmath, python-flint (Arb), matplotlib, gx2 and pytest; R 4.5 with
CompQuadForm, momentchi2 and survey for experiment E5; XeLaTeX (TeX Live) for the figures and the manuscript.
The versions used are in `results/ENVIRONMENT.json`.

Rebuild the numbers, tables and manuscript from the stored results:
`python code/manuscript/make_numbers.py`, then `xelatex main`, `bibtex main`, `xelatex main` twice in `manuscript/`
(or `powershell -File code/manuscript/build.ps1 -Python <python>`). Every number in the text is a `\pthree{key}` macro
written by `make_numbers.py`. It prints its consistency checks and exits with an error if an input file is missing, a
number key is undefined or a check fails (`--allow-incomplete` reports instead of failing, for development only).

Status labels: every number in the result files is labelled exact, certified, computed or simulated; simulated numbers
carry the number of replications and the seed. Certified values come from the tilted route
(`cdf_X_certified_tilt`, `cdf_T_certified_tilt`); the earlier untilted functions `cdf_X_certified` and
`cdf_T_certified` assembled part of their bound in floating point, produced no reported number and now raise an error.
Local absolute paths in `results/ENVIRONMENT.json`, `code/ENVIRONMENT.json`, `results/E5_methods_meta.json`,
`code/manuscript/build.ps1` and `code/vendor/MANIFEST_SHA256.json` were replaced by placeholders such as `<conda>`.
