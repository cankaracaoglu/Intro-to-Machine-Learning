# 01 — Naive Bayes Classifier

Predicting whether a DNA nucleotide sequence is a **splice site** using a naive Bayes classifier
implemented from scratch.

> Introduction to Machine Learning — Assignment 1

---

## Problem

Given 400 nucleotide sequences of length 7 (each position holding `A`, `C`, `G`, or `T`), classify
each sequence into one of two classes:

| Class | Meaning | Count |
|-------|---------|-------|
| 1 | Splice site | 200 |
| 2 | Not a splice site | 200 |

**Split:** first 300 sequences → training, remaining 100 → test.

## Method

The naive Bayes assumption treats the 7 nucleotide positions as conditionally independent given the
class. Training is pure counting — for each class `c` and position `d`, estimate the probability of
observing each nucleotide by maximum likelihood:

```
p_Acd = P(nucleotide at position d is A | class c)
```

and likewise for `C`, `G`, `T`. Class priors come from the training label frequencies (0.5 / 0.5
here, since the classes are balanced).

Scoring is done in **log space** so that multiplying seven small probabilities does not underflow:

```
g_c(x) = Σ_d log P(x_d | c) + log P(c)
```

A sequence is assigned to whichever class gives the higher score.

## Results

**Training set (300 sequences) — 93.7% accuracy**

| y_pred \ y_truth | 1 | 2 |
|---|---|---|
| **1** | 145 | 14 |
| **2** | 5 | 136 |

**Test set (100 sequences) — 90.0% accuracy**

| y_pred \ y_truth | 1 | 2 |
|---|---|---|
| **1** | 48 | 8 |
| **2** | 2 | 42 |

The learned parameters show why the classifier works: at positions 3 and 6, class 1 sequences carry
guanine roughly 83% and 76% of the time, versus ~23% for class 2 — the model latches onto the
conserved `GG` motif that marks a splice site. Class 2 probabilities sit near 0.25 everywhere, which
is what a non-site sequence looks like: uniform noise.

## Files

| File | Description |
|------|-------------|
| `0069063.ipynb` | Solution notebook |
| `hw01_data_points.csv` | 400 sequences × 7 nucleotides |
| `hw01_class_labels.csv` | Class labels (1 or 2) |
| `hw01_description.pdf` | Original assignment brief |

## Running

```bash
jupyter notebook 0069063.ipynb
```

Requires `numpy`, `pandas`, `matplotlib`, `scipy`. Run the cells in order.
