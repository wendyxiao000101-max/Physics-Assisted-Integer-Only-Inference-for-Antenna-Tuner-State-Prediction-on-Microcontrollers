# Data

## Dataset

`x_train.csv` and `y_train.csv` are the public dataset released by Huawei for the "IAT"
(antenna-tuning) enterprise challenge of the 21st China Graduate Electronics Design Contest.
The files are unmodified. `Feature_Description_zh.txt` is the organizer's field description.

The header row of `x_train.csv` is in Chinese (GBK-encoded); the columns are, in order:

| # | Field (organizer's name) | Meaning |
|---|---|---|
| 1, 2 | `gammaIn1Re`, `gammaIn1Im` | Real and imaginary parts of the closed-loop-state reflection coefficient (Γ1), as integers; the code divides by 32,768 |
| 3 | `closeFreqMHz` | Centre frequency in MHz (703-748 and 824-849 MHz, 1-MHz steps) |
| 4 | `itState` | Switch index (0-15) at which Γ1 was recorded; used for surrogate fitting only, not a classifier input |
| 5, 6 | `gammaIn2Re`, `gammaIn2Im` | Real and imaginary parts of the default-state reflection coefficient (Γ2), same scaling |

`y_train.csv` holds the organizer-defined label (0-14) of each of the 96,768 records. The
organizer describes the label as the optimal tuner state; the optimality criterion and the
correspondence between labels and switch indices are not published.

## Splits

Each `fold<k>.npz` holds three integer arrays, `train`, `valid` and `test`, with row indices
into the CSV files (fold k uses seed 42 + k).

* `splits/original/` - the five primary grouped folds over 210 reconstructed trajectories
  (Section II-D). Test parts cover all 96,768 rows exactly once.
* `splits/p95/` - the alternative grouping that removes adjacent-frequency links above the
  pooled 95th-percentile distance (578 groups, Section VI-C). `p95_row_groups.npy` gives the
  group of every row.

The trajectories are algorithmic reconstructions from the unlabeled input geometry; they are
not verified physical scenes.
