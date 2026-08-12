# 04 — Decision Tree Regression

A CART-style regression tree built from scratch, with pre-pruning as the complexity control and a
sweep over the pruning parameter to expose the bias–variance trade-off.

> Introduction to Machine Learning — Assignment 4

---

## Problem

The same Old Faithful geyser dataset as [assignment 03](../03-Nonparametric-Regression): predict
**waiting time until the next eruption** from **eruption duration**, over 272 observations.

**Split:** first 150 points → training, remaining 122 → test.

Using the same data as the nonparametric assignment is deliberate — it makes the tree directly
comparable to the regressogram and kernel smoother.

## Method

The tree is grown by recursive binary splitting over a dictionary-based node structure, where node
`k` has children `2k` and `2k + 1`:

1. **Candidate splits.** At each node, take the unique training values in that node and use the
   midpoints between consecutive values as candidate split positions.
2. **Split criterion.** Score each candidate by the impurity (sum of squared deviations from the
   node mean) of the resulting left and right children, and take the split that minimizes it.
3. **Pre-pruning rule.** If a node holds **P or fewer** data points, make it terminal and stop
   splitting. Its prediction is the mean response of the points it holds.
4. **Prediction.** Walk from the root, going left when `x < split` and right otherwise, until
   reaching a terminal node.

Pre-pruning is the only regularizer — there is no post-pruning pass.

## Results

**At P = 25:**

| Set | RMSE |
|-----|------|
| Training (150) | 4.541 |
| Test (122) | 6.618 |

The gap between training and test error is the tree overfitting the training points it was allowed
to isolate.

**Sweeping P from 5 to 50** (10 values), the notebook re-fits the tree at each setting and plots
training and test RMSE together against `P`. This is the point of the exercise: `P` directly
controls model complexity — a small `P` grows a deep tree with fine-grained terminal nodes that can
fit training noise, a large `P` forces a coarse step function that may underfit — so the two curves
pull in different directions and the useful value of `P` is read off where the test curve bottoms
out. See the plot in the notebook for the fitted curves.

Compared against [assignment 03](../03-Nonparametric-Regression) on identical data, the tree's 6.618
test RMSE at P = 25 lands behind the kernel smoother's 5.875 — unsurprising, since a regression tree
on a single continuous input is essentially an adaptive-width regressogram, and inherits the same
discontinuous, piecewise-constant fit.

## Files

| File | Description |
|------|-------------|
| `0069063.ipynb` | Solution notebook |
| `hw04_data_set.csv` | 272 (eruption time, waiting time) pairs |
| `hw04_description.pdf` | Original assignment brief |

## Running

```bash
jupyter notebook 0069063.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`. Note that the node dictionaries are module-level state, so
the notebook resets them before each re-fit in the `P` sweep — run the cells in order.
