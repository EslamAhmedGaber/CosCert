# CosCert

Reproducible code, empirical series, numerical results and LaTeX sources for **CosCert: certified critical values and exact size for long-memory tests of the mean**, by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman and Sroor M. Elnady.

## Reproduce the paper

Use Python 3.12 with NumPy, SciPy, pandas, mpmath, python-flint and pytest. XeLaTeX and BibTeX build the manuscripts. Matplotlib is needed to redraw figures; the E5 benchmark additionally uses gx2 and the R packages CompQuadForm, momentchi2 and survey. Recorded software versions are in `results/ENVIRONMENT.json` and `code/ENVIRONMENT.json`.

From the repository root run:

```sh
python -m pytest -q code/tests
python code/manuscript/make_numbers.py
cd manuscript
latexmk -xelatex main.tex
latexmk -xelatex supplementary.tex
```

The table generator reads the archived results and reports its consistency checks. To rerun the empirical applications, run `python code/experiments/e13_empirical.py` from the repository root. To rerun the 24-design comparison, use `python code/experiments/e3d_plugin_extended.py competitors`. Each experiment writes to `results/`.

| Directory | Contents |
|---|---|
| `code/p3/` | Gaussian covariance, certified limit probabilities, finite-sample laws and simulation routines |
| `code/experiments/` | Drivers for E0–E13, including alternative numerical and Monte Carlo checks |
| `data/` | Three empirical series, with sources, transformations and missing-hour handling in the data README |
| `results/` | Numerical outputs, retained benchmark records and SHA-256 checksums |
| `literature/` | Source ledgers for published critical values and rejection rates |
| `code/figures/` and `figures/` | Plotting scripts, vector figures, PNG previews and input metadata |
| `manuscript/` | Main article, supplement, bibliography and generated tables |

The Bartlett solver provides adaptive ball quadrature (`cdf_T_certified_quad`) and a tilted midpoint implementation (`cdf_T_certified_tilt`). The original production quantile grid retains its recorded certificates. The adaptive benchmark measures one probability query including construction at H = 0.75 and J = 16,384. The numerical certificate treats H as fixed. Feasible rejection rates with estimated H are studied by simulation.

The experiment protocol and dated amendments are in `PROTOCOL.md`. Supplementary Tables S.8 and S.9 connect results to the corresponding scripts and verification methods.
