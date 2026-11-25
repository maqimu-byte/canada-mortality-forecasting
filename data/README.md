# Data

## Source
**Statistics Canada, Table 13-10-0837-01**: *Life expectancy and other elements of the complete life table, single-year estimates, Canada, all provinces except Prince Edward Island*.
DOI: <https://doi.org/10.25318/1310083701-eng>. Released under the [Statistics Canada Open Licence](https://www.statcan.gc.ca/en/reference/licence).

## Files in `processed/`

| File | Rows | Columns | What it holds |
|---|---|---|---|
| `canada_qx_1980_2024.csv` | 9,990 | `year, sex, age, qx` | Probability of dying between age *x* and *x+1*, Canada, ages 0–110+, males and females, 1980–2024 |
| `canada_ex_official_1980_2024.csv` | 180 | `year, sex, age, ex` | StatCan's published life expectancy at ages 0 and 65, used only to check the life-table code |

Notes
* Each year is StatCan's single-year estimate (the reference period labelled with that year).
* `qx` is stored to 5 decimal places. At ages 5–12, where `qx` is only a few per 100,000, this rounding is a visible share of the value, so single-age rates there are noisy.
* Age 110 is the open interval "110 and over" (`qx` = 1).
* Data were pulled from StatCan's Web Data Service. `src/data.py → fetch_qx_from_statcan()` rebuilds the qx file (it needs an internet connection).

Any other files you download into `data/` are git-ignored.
