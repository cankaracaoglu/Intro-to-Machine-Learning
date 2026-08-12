# 02 — Discrimination by Regression

Ten-class clothing image classification using a discrimination-by-regression model trained with
hand-derived gradients.

> Introduction to Machine Learning — Assignment 2

---

## Problem

Classify 15,000 Fashion-MNIST clothing images (28 × 28 = 784 pixels each) into ten categories:

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 1 | T-shirt/top | 6 | Sandal |
| 2 | Trouser | 7 | Shirt |
| 3 | Pullover | 8 | Sneaker |
| 4 | Dress | 9 | Bag |
| 5 | Coat | 10 | Ankle boot |

**Split:** first 10,000 images → training, remaining 5,000 → test.

## Method

This is a deliberate modification of the standard linear discrimination algorithm, and the point of
the assignment is deriving the update rules that fall out of it:

1. **K independent sigmoids instead of softmax.** Each class gets its own sigmoid output, so the ten
   predicted values are *not* constrained to sum to 1. Prediction takes the `argmax` anyway.
2. **Sum of squared errors instead of cross-entropy.** The objective is
   `0.5 · Σᵢ Σ_c (y_ic − ŷ_ic)²`.

Swapping the loss changes the gradient. Because the sigmoid derivative no longer cancels against the
log-likelihood, an extra `ŷ(1 − ŷ)` factor survives:

```
∂E/∂w_c  = −Σᵢ (y_ic − ŷ_ic) · ŷ_ic (1 − ŷ_ic) · xᵢ
∂E/∂w_c0 = −Σᵢ (y_ic − ŷ_ic) · ŷ_ic (1 − ŷ_ic)
```

Training is batch gradient descent: **η = 0.00001**, **1000 iterations**, with `W` and `w0` seeded
from the provided initialization files so results are reproducible. Labels use one-of-K encoding.

## Results

The notebook plots the sum-of-squared-errors objective across all 1000 iterations to confirm the
descent converges.

| Set | Accuracy |
|-----|----------|
| Training (10,000) | **79.99%** |
| Test (5,000) | **78.84%** |

The 1-point train/test gap indicates the linear model is underfitting rather than overfitting — with
784 input features and no hidden layer, capacity is the limiting factor, not generalization.

**Where the errors concentrate.** The confusion matrices show the model is near-perfect on visually
distinctive classes (trouser, bag, ankle boot) and struggles exactly where a human would: *shirt*
(class 7) is the worst class by a wide margin, recovering only 218 of its 519 test images, with the
bulk leaking into *T-shirt/top*, *pullover*, and *coat*. The upper-body garments occupy overlapping
regions of pixel space that a linear boundary cannot separate.

## Files

| File | Description |
|------|-------------|
| `0069063.ipynb` | Solution notebook |
| `hw02_class_labels.csv` | Image labels (1–10) |
| `hw02_W_initial.csv` | Initial weight matrix (784 × 10) |
| `hw02_w0_initial.csv` | Initial bias vector (10) |
| `hw02_description.pdf` | Original assignment brief |

### Missing input data

`hw02_data_points.csv` — the 15,000 × 784 pixel matrix — is **not included**; the raw file is far too
large to keep in the repository. The notebook expects it in this folder:

```python
x = np.genfromtxt("hw02_data_points.csv", delimiter = ",").astype(float)
```

To run the notebook, rebuild it from Fashion-MNIST as a headerless CSV with one image per row and
784 comma-separated grayscale values per image, ordered to match `hw02_class_labels.csv`.

## Running

```bash
jupyter notebook 0069063.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scipy`. The saved outputs (convergence plot and both
confusion matrices) are viewable without re-running.
