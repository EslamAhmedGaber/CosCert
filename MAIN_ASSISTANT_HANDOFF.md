# Paper 3 — verified follow-up, 17 November 2026

You are the primary assistant and scientific owner of the implementation. These
are review suggestions, not a rigid specification. Accept, reject, reorder,
replace, or improve them when you have a better justification. Explain the
reason for material decisions and keep claims consistent with the actual
evidence. The reviewer may have misunderstood a point; clarify it when needed.

## What was independently checked now

- The current PDF has 45 pages. The source title uses CosCert: certified critical
  values and exact size for long-memory tests of the mean.
- All 34 current tests passed in a fresh local run (35.76 s), including the
  corrected tilted certificate, Ruben anchors, alias bounds, Gershgorin bounds,
  and Cramer–von Mises anchor.
- Direct source inspection confirms the Arb alias sum and geometric remainder,
  exact lattice period, Arb truncation bounds, outward threshold products, and
  outward exponential probability in the tilted certificate.
- The appendix now distinguishes U(c/lambda)=lambda from
  u(1/(2 lambda))=lambda and describes the inflated computed-quantile bracket.
- The stored E2r summary records 300/300 successful bracket revalidations. This
  follow-up inspected the stored evidence; it did not repeat the full expensive
  E2r production grid.
- After adding the missing release inputs below, make_numbers.py regenerated
  617 macros, 224 referenced keys, zero undefined keys, no missing-input warnings,
  all printed consistency checks passing, and all 19 E11 rows. Every regenerated
  table matches the supplied submission package byte for byte.

This is meaningful verification of the fixes, not a new independent proof audit
of every theorem, a rerun of every experiment, or fresh visual inspection of
every PDF page.

## Release defects found and corrected in this repository copy

1. The advertised rebuild command originally failed because the archive omitted
   literature/ledger_rejection_rates.csv and literature/ledger_critical_values.csv.
   Both exact source CSVs have been added.
2. The E11 generator additionally depends on the historical files
   E3b_hi2017_table3.csv.bak16nov, E3b_hi2017_table4.csv.bak16nov,
   E3b_hi2017_table5.csv.bak16nov, and E3c_hi2017_table2.csv.bak16nov.
   Without them it silently generates an incomplete E11 table, reports a missing
   input, and still exits successfully. These four source files have been added.
   Please preserve them as explicit release inputs, or rename/move them to a
   documented provenance directory and update the generator accordingly.
3. results/SHA256SUMS.txt has been regenerated for this completed release copy.
4. README no longer claims the manuscript has already been submitted to CSDA;
   that status requires confirmation from the authors.

The original D: work directory and submission ZIP were not modified. Please
propagate the release-input fixes into your authoritative packaging script and
regenerate the submission ZIP, then test from a clean extracted directory.

## Remaining code review item

The production tilted path has been corrected, but the legacy methods
BridgeSystem.cdf_X_certified and BridgeSystem.cdf_T_certified in code/p3/certify.py
still expose the earlier scalar rounding approach: float thresholds, erfc with
a heuristic factor, and ordinary float endpoint arithmetic. No call sites for
these two legacy methods were found in the packaged experiment scripts.

Please remove/deprecate/disable these methods, relabel them as computed, or apply
the same rigorous repairs and add relevant numerical validation. Do not let a
reader accidentally select the old path because its API name says certified.
The earlier helper _alias_tilt may legitimately remain a numerical grid-selection
heuristic because the final tilted result recomputes its actual bounds in Arb.

Also consider making manuscript generation fail when required inputs are missing
or a consistency check fails. Exit status zero alone currently does not establish
a complete rebuild.

## GitHub and manuscript integration

A local Git repository and complete reproducibility package have been prepared.
The saved Windows GitHub credential was retrievable; the execution session's
network policy blocked the GitHub API connection. No remote repository creation
or upload has been verified. Do not insert an assumed live repository link.

Run `python create_github_repo.py` from a normal network-enabled terminal in this
directory. It uses the existing Git Credential Manager credential, verifies the
account EslamAhmedGaber, creates CosCert as private if absent, commits/pushes the
package without force, and verifies the remote commit. It prints the actual
repository URL and writes a local receipt only after successful verification.

After success, add the verified URL to the code/data availability paragraph and
the reproducibility appendix, and record a specific commit or release. While
private, describe it as a private repository; do not claim public availability.
Once the authors choose to make it public, update the wording accordingly.

## Writing, presentation, and method naming

Please use the scientific-writing-studio skill for the final title, abstract,
captions, narrative flow, sentence division, notation/symbols, and polished
academic phrasing. Review the assembled PDF after the final integration.

CosCert is already the chosen method name. Keep its definition and capitalization
consistent across the title, text, tables, figures, legends, code documentation,
and repository. Use it for the actual proposed method, while preserving the names
of comparators. If you judge the name or scope scientifically misleading, explain
why and propose a better alternative; naming is not evidence of novelty.

Final note: this is a working guideline, not a cage. You may improve the plan,
scope, order, design, and wording. Preserve supported claims, correct certificate
boundaries, the distinction between computed and certified results, and a clear
record of what was actually verified.
