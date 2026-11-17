# Paper 3 — experiment protocol (frozen 15 Nov 2026, before any production run)

Owner: executing agent. Origin of the criteria: the reviewer-model design of 15 Nov 2026
(`work\design_review_20261115\reviewer_T3_experiments_figures_execution.md`, section 6, proposals P1–P10) and the author's
requirements of 15 Nov 2026 (acceptance targets in test-size units, labelling of every rejection rate, response-surface reporting,
separate signed and squared anchors, Monte Carlo under each model's own covariance). The earlier 14 Nov plan (L0–L11) was never run
and is superseded; its pilots stay labelled **pilot** until reproduced.

**Amendment rule.** A criterion below may be revised only by appending a dated amendment at the end of this file that quotes the
original wording, states the new wording and gives the reason. Results are always reported against the criterion in force when the
run started, and failures are reported with their numbers.

## 0. Central question

For a researcher who wants a level-α test (α = 10 %, 5 %, 1 %) on the mean of a long-memory series: which critical value should be
used for each procedure (self-normalised / Bartlett b = 1, Hualde–Iacone fixed-m, EWC cosine series), what rejection rate does the
test actually have at the researcher's sample size, and how accurately are both answers established?

## 1. Acceptance targets in test-size units (primary)

* **AT1 — limit critical values.** For every reported limit critical value c of a procedure at nominal α, the certified (or, where
  stated, computed) interval for the limit rejection probability P(stat > c) must lie inside [α − 0.01 pp, α + 0.01 pp], i.e. its
  half-width plus the effect of printing c is at most 1·10⁻⁴ in probability. The printed digits are those shared by both bracket
  ends after outward rounding. If a cell misses AT1 its achieved width is printed and the cell is flagged.
* **AT2 — finite-sample rejection rates.** Each exact finite-n rejection rate is reported to 0.01 pp with numerical error ≤ 0.005 pp
  (5·10⁻⁵). Every rate is labelled with: data model (fGn or ARFIMA(p,d,0) with parameters), H (or d), n, the critical value used
  (published / certified limit / naive H = ½ / plug-in with estimated H), and **finite-sample** or **asymptotic**.
* **AT3 — response surface.** A fitted formula q_p(H) is called *validated*, never certified. Report its largest size error on
  held-out H points (target ≤ 0.01 pp on 0.50 ≤ H ≤ 0.95) and state the domain on which any rigorous statement holds (only the
  certified grid points themselves).
* **AT4 — Monte Carlo validation.** Every Monte Carlo check uses the model's own covariance (fGn: γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} +
  |k−1|^{2H}); ARFIMA(0,d,0): γ(k) = Γ(1−2d)Γ(k+d)/(Γ(d)Γ(1−d)Γ(k+1−d)); ARFIMA(1,d,0): AR(1) filter of the latter) with exact
  circulant-embedding or Cholesky sampling. An exact finite-n value passes if it lies inside the 99 % Monte Carlo interval; across
  each family at most one in twelve may fall outside 99 % and none beyond 4 standard errors.
* **AT5 — anchors before any larger grid (gate G0).** At H = ½: (a) T_½ quantiles from the closed-form characteristic function and
  from an explicit-eigenvalue Imhof sum agree to ≥ 10 significant digits; (b) signed and squared fixed-smoothing anchors are checked
  separately: HI τ against t_{2m} and τ² against F(1,2m); EWC t against t_B and t² against F(1,B), each to ≥ 10 digits;
  (c) Cramér–von Mises points (KPSS level case) and the cosh law of ∫B²; (d) the certified route encloses every exact anchor.

## 2. Pre-registered criteria (reviewer proposal P1–P10 of 15 Nov 2026, adopted with the size-unit targets above)

* **P1 (anchors).** H = ½: T, F1, F3 quantiles agree with closed-form inversion to ≥ 10 digits; printed classical digits reproduced.
* **P2 (containment).** For every (H, p) and every J on the ladder: the H = ½ exact value is inside; brackets are nested in J; the
  exact finite-n quantile, extrapolated in n, lies inside or within its extrapolation spread (otherwise *undecided*, not *pass*).
* **P3 (width).** Certified relative quantile width of T_H ≤ 1e-4 for all H ∈ {0.50, …, 0.95}, p ∈ {.90, .95, .975, .99}, and
  ≤ 1e-5 for H ≥ 0.75. *(Superseded as primary target by AT1; still reported.)*
* **P4 (certification).** Every certified CDF value has total radius ≤ 1e-9; every double-precision Imhof value lies inside its
  enclosure.
* **P5 (fixed-m).** Enclosure radius ≤ 1e-10; F(1, 2m) check passes.
* **P6 (finite-n).** Monte Carlo z-scores: no more than 1 of 12 outside its 99 % interval, and none beyond 4σ (= AT4).
* **P7 (formula).** Maximum size error ≤ 0.01 percentage points on [0.5, 0.95] (= AT3).
* **P8 (audit).** No threshold. Every ledger value is reported with its verdict. "Published tables are inaccurate" is allowed only
  for values classified *unexplained*, or if the true size deviates by more than 0.1 percentage points at 5 %; otherwise the
  wording is "accurate to k digits, now certified".
* **P9 (baselines).** No threshold. All cheaper-at-equal-accuracy cases are printed.
* **P10 (integrity).** Every failed criterion appears in the paper with its numbers. Novelty claims at H = ½ are limited to what
  Abadir–Paruolo did not give.

## 3. Audit rules (published values)

1. Enter only values read in a held source (`work\literature\ledger_*.csv`, page/table recorded); a value quoted from memory or from a
   secondary paper is not a published value.
2. Before comparing, match statistic, normalisation (e.g. Q(1) = 2∫B̃² for Bartlett b = 1, so T_H = 2t²), tail convention (one- vs
   two-sided; signed vs squared) and memory parametrisation (H = δ + ½ = (β + 1)/2).
3. Report for each value: the certified true rejection probability at the published value, the relative deviation from the certified
   quantile, the deviation in Monte Carlo standard errors when N is stated (quantile s.e. = √(p(1−p)/N)/f(q)), and the exact
   discretisation effect of the published design (e.g. 10,000-step Riemann sums, n = 2000 grid) when it is reported.
4. Attribution to finite-sample bias or Monte Carlo noise only when the published design supports it; otherwise *unresolved*.

## 4. Status vocabulary

**exact** — closed form, or a finite-dimensional law evaluated to a stated numerical error; **certified** — every numerical step
enclosed (matrix entries, eigenvalues, truncation remainder, inversion, rounding, quantile search) with outward rounding;
**computed** — floating point without a complete enclosure; **simulated** — Monte Carlo; **validated** — empirical error on a
declared held-out set; **pilot** — produced before this protocol and not yet reproduced.

## 5. Pilot results carried in (labelled pilot until reproduced)

* P6c (14 Nov): centred T_H brackets at J = 256, H = ½: q.90 ∈ [28.2015, 28.4337], q.95 ∈ [45.2696, 45.7208], q.99 ∈ [99.5812, 100.9287];
  H = 0.75: [96.3141, 96.3850], [163.1977, 163.3487], [399.4377, 400.0053].
* P6d (14 Nov): Monte Carlo of the discretised statistic, 5·10⁵ paths (H = ½, N = 1024; H = 0.75, N = 1024 and 2048).
* Pre-production check (15 Nov, before this file): H = ½ quantiles of T from the closed-form CF and from an Imhof sum over 4·10⁵
  eigenvalues: 28.330921323509, 45.526085752448, 66.586088479124, 100.34563942809 (p = .90, .95, .975, .99) — to be reproduced by
  the library in gate G0.

## 6. Production choices fixed later by dated amendment

The truncation J(H), the H grids, and the n grid of the finite-sample study are chosen after the full-calculation benchmark
(step 3 of the plan) and recorded below as amendments, with the benchmark numbers that justify them.

## 7. Seeds

Monte Carlo seeds: 20261115 + 1000·round(1000·H) + n (finite-n validation); 20261116 + … (estimated-H studies); recorded in each result file.

## Amendments

### Amendment 1 — 16 Nov 2026 (truncation J and H grid; fixed after the step-3 benchmark, before production)

Section 6 said: *"The truncation J(H), the H grids, and the n grid of the finite-sample study are chosen after the
full-calculation benchmark (step 3 of the plan) and recorded below as amendments, with the benchmark numbers that justify them."*

Decision: **J = 16 384 sine modes for every certified T_H cell, H = 0.50, 0.51, …, 0.95 (46 exact rationals), levels
p = 0.80, 0.90, 0.95, 0.975, 0.98, 0.99**; H = 0.96–0.99 are run with the same J and reported with their achieved status.
Benchmark (`results/E1_benchmark_H*.json`, logs in `logs/E1_benchmark_*.log`): at J = 16 384 the certified width of
P(T_H ≤ q) is 1.9·10⁻⁶ / 1.5·10⁻⁶ / 5.9·10⁻⁷ at H = 0.50 (p = .90/.95/.99), 8·10⁻⁷ at H = 0.55 and 2.4·10⁻⁸ at H = 0.75; the
implied relative quantile width is ≤ 2.4·10⁻⁵ for H ∈ [0.50, 0.55] and ≤ 2·10⁻⁷ for H ≥ 0.75, so P3 (≤ 1e-4; ≤ 1e-5 for
H ≥ 0.75) and AT1 (± 1e-4 in size) hold with margin. Widths scale as J^(−3/2) (Laurent–Massart window), build time
25–35 s, certified CDF evaluation 4–45 s. The certified engine is the tilted-Bromwich / head–tail Schur method of
`code/p3/certify.py` (Bromwich line Re w = 1/4, L = 134), adopted after the untilted Gil-Pelaez sum needed 3–8× more nodes.

### Amendment 2 — 16 Nov 2026 (scope of P4)

P4 said: *"Every certified CDF value has total radius ≤ 1e-9; every double-precision Imhof value lies inside its enclosure."*

New wording: *"Every certified evaluation of a finite-dimensional CDF (the ball-arithmetic inversion of the truncated form
X_J, or of a fixed-smoothing limit) has radius ≤ 1e-9; the truncation remainder of T_H is a separate, reported bracket
governed by AT1 and P3; every double-precision value is compared with its enclosure and any excursion is reported."*
Reason: the remainder bracket of an infinite-dimensional law cannot reach 1e-9 at feasible J (its width scales as J^(−3/2);
1e-9 at H = 1/2 would need J ≈ 2.5·10⁶), whereas the finite-dimensional radius is 1.6·10⁻¹³–9.6·10⁻¹¹ in the benchmark. The
original wording is kept here and failures against it are reported as such.

### Amendment 3 — 16 Nov 2026 (node cap of the tilted inversion; computational parameter, no criterion changed)

`certify._sys_choose_K_tilt` searched the number of quadrature terms K up to 4·10⁵. Evaluating T_H distribution functions far
below the bulk (E6: H ≥ 0.95 at the H = ½ critical values) needs about 5·10⁵–10⁶ terms, so the cap was raised to 4·10⁶. The
rigorous stopping rule (truncation majorant ≤ ε) is unchanged, so every value computed under the old cap is unchanged:
the largest K used by any E2 cell is 36 184 for H ≤ 0.95 and 43 784 for H = 0.96–0.99.

### Amendment 4 — 16 Nov 2026 (AT3: selection of the response-surface degree)

AT3 says: *"A fitted formula q_p(H) is called validated, never certified. Report its largest size error on held-out H points
(target ≤ 0.01 pp on 0.50 ≤ H ≤ 0.95) and state the domain on which any rigorous statement holds."* The target is unchanged.
The first implementation chose the polynomial degree by the held-out error itself, so the held-out points were not
independent of the fit. New procedure: the degree is the smallest common K whose leave-one-out error on the training points
(even hundredths) is below one tenth of the training certification floor at every level (K = 7); the odd hundredths are used
once, for reporting. Two errors are reported: the fit error against the tabulated points, and a guaranteed error against the
true law that adds the certified bracket. Result (`results/E9_surface.json`): held-out fit error ≤ 4.0·10⁻⁷ pp, guaranteed
error ≤ 1.43·10⁻⁴ pp on 0.50 ≤ H ≤ 0.99. AT3 is met.

### Amendment 5 — 16 Nov 2026 (P8: verdict class for values that are exact for the source's own finite design)

P8 says: *"Every ledger value is reported with its verdict. 'Published tables are inaccurate' is allowed only for values
classified unexplained, or if the true size deviates by more than 0.1 percentage points at 5 %."* The verdict classes were
'accurate within MC error', 'deviates > 0.1 pp (within 3 MC s.e.)' and 'unexplained (> 3 MC s.e.)'. Added class, introduced
**after** the pattern at δ = −0.49 had been seen (disclosed as post hoc): *'explained by the source's finite design'*, for a
value whose limit size is more than 3 MC s.e. from nominal but whose size under the exact law of the discretised design the
source simulated lies within 3 MC s.e. Such values are reported with both sizes, as accurate for that design and not as
limit values. Applied to Hualde–Iacone (2017) Table 1, δ < 0 (80 cells): under a stationary ARFIMA(0, δ, 0) path of length
10 000 (their Remark 6: 'summations over 10,000 steps') 3/80 cells exceed 2 MC s.e. and none exceeds 3; under the limit 29/80
exceed 2 s.e. (up to 112 s.e. at δ = −0.49, m = 16); an fGn path of the same length does not explain them. The design is an
inference from the match and from Remark 6; the paper says so.

### Amendment 6 — 16 Nov 2026 (memory cap in the estimated-H Monte Carlo)

Host memory (about 2 GiB free) required at most 2·10⁶/n complex circulant draws per batch in `e3b_mc.py plugin`. Seeds are
unchanged; only the partition of that mode's random stream differs from the BATCH = 4000 design. No other mode is affected.

### Amendment 7 — 16 Nov 2026 (referee round; all items post hoc and reported as such)

1. **Audit count.** The 1990 KPSS working paper prints the same eight values as the 1992 article; they are counted once
   (N = 241 ledger entries → 233 audited values). Kiefer–Vogelsang (2002) values are Abadir–Paruolo (1997) values divided
   by √2 but are separately printed and used, so both are kept, with the derivation stated.
2. **Verdict display.** Figure bands and error bars show ±3 Monte Carlo standard errors, the threshold of the verdict
   classes (previously ±2). The word "inaccurate" is used only for class E (P8 allowed it more widely).
3. **Response surface (AT3).** The "guaranteed" error was a linearisation through the density. It is replaced by a
   certified bound for the coefficients as printed (7 significant digits): monotone bracket bound where the fitted value
   lies in the certified bracket, otherwise a fresh certified evaluation of F_H at the fitted value
   (`e9b_surface_certified.py`, J = 16384, t = 25, tilt 1/4). The table header's level mapping was wrong
   (two-sided α = 2(1−p)); corrected to α = 1−p, one-sided a = (1−p)/2.
4. **Bandwidth sensitivity (P-plug-in).** Local Whittle bandwidths ⌊n^0.5⌋ and ⌊n^0.8⌋ added for fGn and ARFIMA(0,d,0),
   H ∈ {0.7, 0.9}, all n, same seeds as the ⌊n^0.65⌋ production runs (`e3b_mc.py plugin_bw`).
5. **Growth as H → 1 (new Proposition).** E12 (`e12_edge_limit.py`) computes C_p = lim (1−H) q_p(H) by Nyström + Imhof
   (double precision, computed, not certified) and compares with the certified values.
6. **Bug fix.** `e3b_mc.py part_hi2017` referred to an undefined `n` (copied memory-cap line of Amendment 6); the mode now
   uses BATCH = 4000 as originally, and a re-run reproduced E3b_hi2017_table{3,4,5}.csv bit for bit.
7. **Statement fixes (no change to any number).** Theorem 'Certified enclosure' now encloses the exceedance probability
   P(X_J > x) with symmetric alias bounds and tilt ϑ ∈ (0, ½), matching the proof and the code; the power figure caption
   no longer calls the Bartlett powers exact (J = 512 truncation with mean remainder).


### Amendment 8 — 17 Nov 2026 (continued audit; all items post hoc and reported as such)

1. **Certificate arithmetic (A1).** The scalar assembly of the certified distribution function of T_H was moved into ball
   arithmetic: alias sums with a geometric tail bound, the truncation bound in Arb, the exact lattice period 2π/Δ,
   outward-rounded evaluation points, e^-t rounded up; eigenvalue lower bounds by Gershgorin clusters (sorted, with Cauchy
   interlacing) instead of a Bauer–Fike radius whose pairing with the sorted eigenvalues was not guaranteed. The same gaps
   were found in the Cramér–von Mises route (W = ∫B̃², used only at H = ½) and fixed in the same way.
2. **Revalidation, not patching.** Every certified value was recomputed with the final code (`e2r_revalidate.py`,
   `e2r_print.py`, `e0d_revalidate.py`, merged by `e2r_merge.py`): all 300 of 300 SN brackets confirmed
   (J = 16384), endpoints of the stored enclosures moved by at most 3.1e-15; certified sizes of the
   printed values recomputed at both floating-point neighbours of each printed decimal (largest change
   2.3e-15); naive and rescaled SN sizes (J = 4096, 61 values, largest change
   1.3e-15); the held-out surface bound now targets the exact real value of the printed expression
   (ball arithmetic): 2.4e-04 pp at H = 0.51, p = 0.800 (82 of 150
   cells by fresh evaluation; the 16 Nov bound was 2.4e-04 pp); gate G0 part (d) passes
   (all anchors contained).
3. **Verdict classes (P8).** The class-D rule, first applied by hand, is now explicit and uniform: a block (source × statistic ×
   memory regime) with ≥ 3 values, all within 3 s.e., one sign, and |Σz|/√k > 3. Precedence C/E, D, B/A. The rule also
   selects the Hualde–Iacone row δ = 0 (16 values): classes A 83, B 82, C 28,
   D 32, E 8 (16 Nov: 85, 96, 28, 16, 8). The joint z treats values as independent; row-level z is
   reported alongside.
4. **Benchmark E5 (P9).** Cost accounting per method (shared covariance excluded; cold set-up, first and repeated query,
   amortisation over 10 cutoffs); error intervals against certified enclosures; LTZ run but labelled outside its
   documented domain; Satterthwaite remainder with fractional degrees of freedom; adaptive outer quadrature for the R
   positive-form methods (64-node Gauss–Hermite kept as a diagnostic, up to 0.08 pp); BLAS threads pinned to
   one; versions recorded. Methods counted as 21 implementations of 14 methods.
5. **New checks.** SN local power J ladder (E7b: ≤ 9.0e-9 from J = 256 to 512, ≤ 1.1e-9 from 512 to
   1024); noncentral-t check by quadrature (E7c: ≤ 2.3e-13 over 610 cells; the earlier printed 0 was
   a NaN-masked maximum, scipy.stats.nct returns NaN for large noncentrality); eigenvalue clipping in the exact laws (E3c:
   at most one per cell, ≤ 9e-17 relative); McElroy–Politis β = −0.4 coefficient rounding moves sizes by
   ≤ 0.05 pp.
6. **Independent checks (E11)** are assembled by make_numbers.py from result files in six evidence categories; the earlier
   KPSS row compared the closed-form Cramér–von Mises quantile with itself (mislabelled) and was replaced.
