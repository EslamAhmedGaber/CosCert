# Empirical series

`nile_minima_622_1284.csv` contains the series analysed in Section 6: the 663 annual minimum water levels of the Nile, in centimetres, at the Roda Gauge, AD 622–1284 (Toussoun 1925), distributed as `NileMin` in the R package longmemo (Beran 1994). Columns: `year_ad`, `min_level_cm`. There are no missing values. The illustrative reference level in Section 6 is 1,100 cm. [Source series](https://github.com/cran/longmemo/blob/master/data/NileMin.R)

Run `python code/experiments/e13_empirical.py` from the repository root to reproduce `results/E13_empirical.json`, `results/E13_empirical.csv` and, through `code/manuscript/make_numbers.py`, Table 9 and the numbers of Section 6.

In the empirical CSV, columns ending in `_reference` or beginning with `ci95_reference_` give each comparator's own cutoff, interval, p-value and decision. Blank cells denote inapplicable quantities.
