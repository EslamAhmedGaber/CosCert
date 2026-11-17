# CosCert code and data

Reproducibility companion for *CosCert: certified critical values and exact size for long-memory tests of the mean*
(E. Ahmed, M.A. El-Beltagy, A. Etman, S.M. Elnady).

This is a research manuscript companion. No submission or publication status is asserted here.

| Folder | Content |
|---|---|
| `code/p3/` | library: `gram.py` (closed-form Gram matrices), `lowrank.py` (certified fixed-m and EWC limits), `certify.py` (CosCert: certified distribution function of T_H and of the Cramer-von Mises law), `finite_n.py` (exact finite-sample laws, circulant-embedding sampler, local Whittle), `laws.py` (Imhof, Ruben and reference laws) |
| `code/experiments/` | one driver per experiment E0-E12 (Table E.18 of the paper) and the revalidation scripts `e0d_revalidate.py`, `e2r_revalidate.py`, `e2r_print.py`, `e2r_merge.py` |
| `code/figures/` | one script per figure (matplotlib PGF backend, typeset with `manuscript/p3fonts.tex`) |
| `code/manuscript/` | `make_numbers.py` (writes every number and generated table of the paper from `results/`), `build.ps1`, `write_environment.py` |
| `code/tests/` | unit tests (`python -m pytest code/tests -q`) |
| `results/` | every result file; `ENVIRONMENT.json` (versions, hardware), `SHA256SUMS.txt` (checksums) |
| `figures/` | the figures as printed (vector PDF and PNG) |
| `manuscript/` | LaTeX source in its working layout (`main.tex`, `sections/`, `tables/`, `p3fonts.tex`, `references.bib`) |
| `literature/` | the two CSV ledgers required by the experiments and manuscript generator |
| `PROTOCOL.md` | the experimental protocol with its dated amendments |

Requirements: Python 3.12 with NumPy, SciPy, pandas, mpmath, python-flint (Arb), matplotlib, gx2 and pytest; R 4.5 with
CompQuadForm, momentchi2 and survey for experiment E5; XeLaTeX (TeX Live) for the figures and the manuscript.
The versions used are in `results/ENVIRONMENT.json`.

Rebuild the numbers, tables and manuscript from the stored results:
`python code/manuscript/make_numbers.py`, then `xelatex main`, `bibtex main`, `xelatex main` twice in `manuscript/`
(or `powershell -File code/manuscript/build.ps1 -Python <python>`). Every number in the text is a `\pthree{key}` macro
written by `make_numbers.py`, which also prints its consistency checks (all must read `ok`).

Status labels: every number in the result files is labelled exact, certified, computed or simulated; simulated numbers
carry the number of replications and the seed. Local absolute paths in `results/ENVIRONMENT.json`, `code/ENVIRONMENT.json`,
`results/E5_methods_meta.json`, `code/manuscript/build.ps1` and `code/vendor/MANIFEST_SHA256.json` were replaced by
placeholders such as `<conda>` in this archive, and
`results/SHA256SUMS.txt` was recomputed for the archived files.

## Repository preparation and verification (17 November 2026)

The release copy includes the two literature ledgers and the four historical
`results/*.bak16nov` CSV files used by the E11 reproducibility comparison.
These historical CSV files are intentional research inputs.

Fresh local verification: 34 tests passed; the table generator wrote 617 number
macros, with 224 used in the manuscript, zero undefined keys, no missing inputs,
and all printed consistency checks passing. This reruns tests and regenerates
tables from stored results; it does not rerun every production experiment.

The paper's corrected SN certificate uses `cdf_T_certified_tilt`; the legacy
`cdf_T_certified` / `cdf_X_certified` routines still need the same rounding
review and are not covered by this release verification.

To create and upload the private repository using the saved Windows Git Credential
Manager login, run `python create_github_repo.py` from a normal terminal.
The script verifies the account and remote commit and prints the actual URL on
success. No repository creation or upload has yet been verified in this copy.
Do not claim public code availability until the repository is actually public.
