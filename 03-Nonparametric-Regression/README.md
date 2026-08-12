# 03 — Nonparametric Regression

Three nonparametric regression estimators — regressogram, running mean smoother, and kernel smoother
— implemented from scratch and compared on the Old Faithful geyser dataset.

> Introduction to Machine Learning — Assignment 3

---

## Problem

Predict the **waiting time until the next eruption** from the **duration of the current eruption**,
using 272 observations of the Old Faithful geyser in Yellowstone National Park.

**Split:** first 150 points → training, remaining 122 → test.
**Bandwidth:** `h = 0.37` for all three estimators, origin at 1.5, so the comparison is like-for-like.

## Method

All three estimators are local averages of the training responses — they differ only in *how* they
decide which neighbors count and how much each one weighs.

**Regressogram** — partition the input range into fixed bins of width `h` and predict the mean of
the training responses in each bin. The fit is piecewise constant and discontinuous at bin edges:

```
b(x, xᵢ) = 1 if xᵢ falls in the same bin as x, else 0
```

**Running mean smoother** — replace the fixed bin grid with a window centered on the query point,
so each prediction averages every training point within `h/2`. Still a hard 0/1 weight, but the
window slides, which removes the artificial bin boundaries:

```
w(u) = 1 if |u| ≤ 0.5, else 0
```

**Kernel smoother** — replace the box weight with a Gaussian kernel, so nearby points count more
than distant ones and the weight decays smoothly rather than cutting off:

```
K(u) = (1/√(2π)) · exp(−u²/2)
```

Each estimator is plotted over the training and test scatter, then scored by RMSE on the test set.

## Results

| Estimator | Test RMSE (h = 0.37) |
|-----------|----------------------|
| Regressogram | 5.963 |
| Running mean smoother | 6.113 |
| **Kernel smoother** | **5.875** |

The kernel smoother wins, which is the expected ordering: its smoothly decaying weights produce a
continuous fit that neither jumps at bin edges (regressogram) nor wobbles as points enter and leave
a hard window (running mean smoother). The margin is small because all three share the same
bandwidth — the choice of `h` matters more than the choice of weighting function.

## Files

| File | Description |
|------|-------------|
| `0069063.ipynb` | Solution notebook |
| `hw03_data_set.csv` | 272 (eruption time, waiting time) pairs |
| `hw03_description.pdf` | Original assignment brief |

## Running

```bash
jupyter notebook 0069063.ipynb
```

Requires `numpy`, `matplotlib`, `scipy`. The notebook ships with its three fitted-curve plots saved,
so the visual comparison is viewable on GitHub without executing anything.
