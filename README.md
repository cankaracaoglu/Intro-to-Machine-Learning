# Introduction to Machine Learning

Coursework for an introductory machine learning course.

Every algorithm in this repository is implemented **from scratch with NumPy** — no scikit-learn, no
pre-built estimators. Each assignment derives the model, implements the training loop, and evaluates
it with confusion matrices or RMSE against a held-out test set.

**Author:** Yakup Can Karacaoğlu

---

## Assignments

| # | Project | Topic | Dataset | Headline result |
|---|---------|-------|---------|-----------------|
| 01 | [Naive Bayes Classifier](01-Naive-Bayes-Classifier) | Parametric classification, MLE | 400 DNA nucleotide sequences (splice sites) | 90.0% test accuracy |
| 02 | [Discrimination by Regression](02-Discrimination-by-Regression) | Multiclass logistic regression, gradient descent | 15,000 Fashion-MNIST clothing images | 78.8% test accuracy |
| 03 | [Nonparametric Regression](03-Nonparametric-Regression) | Regressogram, mean smoother, kernel smoother | Old Faithful geyser (272 points) | 5.875 RMSE (kernel smoother) |
| 04 | [Decision Tree Regression](04-Decision-Tree-Regression) | CART regression tree with pre-pruning | Old Faithful geyser (272 points) | 6.618 test RMSE (P = 25) |
| 05 | [Expectation-Maximization Clustering](05-Expectation-Maximization-Clustering) | Gaussian mixture model, EM algorithm | 1,000 points from 5 bivariate Gaussians | Recovered all 5 true means |

Each folder contains the solution notebook, the dataset, the original assignment brief
(`hw0N_description.pdf`), and a `README.md` explaining the method and results.

---

## Concepts covered

- **Parametric classification** — maximum-likelihood parameter estimation, class priors, discriminant
  functions, log-space scoring to avoid numerical underflow (01)
- **Discriminative learning** — sigmoid activations, sum-of-squared-errors objective, hand-derived
  gradients, batch gradient descent, one-of-K encoding (02)
- **Nonparametric estimation** — bin-based and kernel-weighted local averaging, the bandwidth
  bias–variance trade-off (03)
- **Tree-based models** — recursive binary splitting on impurity reduction, pre-pruning as a
  complexity control, train/test error vs. tree size (04)
- **Unsupervised learning** — mixture models, soft assignments, the E-step/M-step alternation,
  k-means initialization (05)

---

## Running the code

Every assignment is a self-contained Jupyter notebook that reads its CSV files from its own folder.

```bash
pip install numpy pandas matplotlib scipy jupyter
```

```bash
jupyter notebook 01-Naive-Bayes-Classifier/0069063.ipynb
```

Run the cells top to bottom. The notebooks ship with their outputs saved, so the figures and
printed results are viewable on GitHub without executing anything.

> **Note:** `02-Discrimination-by-Regression` is missing its `hw02_data_points.csv` input — the raw
> pixel matrix was too large to commit. See that folder's README for how to reconstruct it.

---

## Repository layout

```
.
├── 01-Naive-Bayes-Classifier/
│   ├── 0069063.ipynb                 # solution notebook
│   ├── hw01_data_points.csv          # 400 nucleotide sequences
│   ├── hw01_class_labels.csv         # splice site / not splice site
│   ├── hw01_description.pdf          # assignment brief
│   └── README.md
├── 02-Discrimination-by-Regression/
├── 03-Nonparametric-Regression/
├── 04-Decision-Tree-Regression/
└── 05-Expectation-Maximization-Clustering/
```

---

## Tech stack

`Python` · `NumPy` · `pandas` · `Matplotlib` · `SciPy` · `Jupyter`
