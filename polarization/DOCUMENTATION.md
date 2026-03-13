# polarization — Library Documentation

A Python library for detecting and quantifying **polarization** and **consensus** in survey data using weighted kernel scoring and bootstrap inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Concepts](#concepts)
   - [The Scoring Idea](#the-scoring-idea)
   - [Weight Matrix](#weight-matrix)
   - [Kernels](#kernels)
   - [Polarization vs Consensus](#polarization-vs-consensus)
   - [Bootstrap Inference](#bootstrap-inference)
4. [API Reference](#api-reference)
   - [calculate_scores](#calculate_scores)
   - [weight_matrix_generator](#weight_matrix_generator)
   - [dashboard_pair](#dashboard_pair)
   - [weight_matrix_visualization](#weight_matrix_visualization)
5. [Return Value Schemas](#return-value-schemas)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Examples](#examples)

---

## Overview

Given two Likert-scale survey variables, this library answers questions like:

- *How polarized are Muslim and Kemalist identity self-ratings among respondents?*
- *Is there more consensus or divergence between left-wing and right-wing self-placement?*
- *Which group contributes more to the observed polarization?*

The core idea: build a joint frequency table of the two variables, multiply each cell by a geometrically derived weight, and sum the result into a scalar score. Confidence intervals are produced via non-parametric bootstrap resampling.

---

## Installation

The library has no build step — place the `polarization/` folder next to your notebook or script, then import directly:

```python
from polarization import calculate_scores, dashboard_pair
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`

---

## Concepts

### The Scoring Idea

For a pair of variables `(x, y)` measured on a common integer scale `[start_value, end_value]`:

1. Build an `(N+1) × (N+1)` joint count matrix `A` where `N = end_value - start_value`.
   `A[i, j]` = number of respondents who answered `i` on `x` and `j` on `y`.

2. Normalise each region of `A` independently (upper triangle, lower triangle, diagonal) to get a probability matrix `P`.

3. Build a weight matrix `W` encoding how much each cell `(i, j)` contributes to the score geometrically.

4. The score is `sum(W * P)` over the relevant region.

### Weight Matrix

Each weight `W[i, j]` is the product of two terms:

| Term | Symbol | Meaning |
|---|---|---|
| Distance | `d` | How far apart `i` and `j` are on the scale |
| Agreement | `a` | How much `i` and `j` tend toward the same region |

The exact formula depends on the chosen `type` and `kernel` (see below). The matrix is always normalised so its maximum absolute value equals 1.

### Kernels

Two kernels control how `d` maps to a weight:

| Kernel | Formula | Effect |
|---|---|---|
| `'power'` | `M = (d**p) * (a**q)` | Smooth, monotone. `p < 1` emphasises moderate distances; `p > 1` emphasises extremes. |
| `'gaussian'` | `M = (1 - exp(-d² / 2p²)) * sign(d) * (a**q)` | S-shaped. `p` is the bandwidth sigma; small `p` makes the weight rise steeply near the diagonal. |

The parameter `q` always controls the sharpness of the agreement term — higher `q` suppresses pairs that are not sufficiently close to the relevant axis/anti-diagonal.

### Polarization vs Consensus

These are two distinct geometric views of the same joint table:

#### Polarization

The main diagonal (`i == j`) is the "agreement axis". The two off-diagonal triangles represent the two groups:

- **Lower triangle** (`i > j`): respondents where `x > y` — the "x-leaning" group.
- **Upper triangle** (`i < j`): respondents where `x < y` — the "y-leaning" group.

The weight is highest for cells that are:
- Far from the diagonal (high disagreement between `x` and `y` ratings), **and**
- Close to the middle of the scale (not at both extremes simultaneously).

High polarization score → the two variables split respondents into opposing camps.

#### Consensus

The anti-diagonal (`i + j == N`) is the "midpoint axis". The two halves represent:

- **Negative half** (`i + j < N`): both ratings tend low.
- **Positive half** (`i + j > N`): both ratings tend high.

The weight is highest for cells where:
- The pair sum is far from `N` (both high or both low), **and**
- `i ≈ j` (the two ratings are similar, i.e. strong internal consistency).

High consensus score → respondents tend to rate both variables similarly (either both high or both low).

### Bootstrap Inference

All public scores are accompanied by bootstrap confidence intervals. The procedure:

1. Compute the point estimate on the full dataset.
2. Draw `B` bootstrap samples (resampling rows with replacement).
3. Compute the score on each sample.
4. Report `mean`, `se` (standard deviation of bootstrap distribution), and the `ci`-level percentile interval.

This is a non-parametric approach — no distributional assumptions are made about the responses.

---

## API Reference

### `calculate_scores`

```python
calculate_scores(
    df,
    first_variable,
    second_variable,
    *,
    start_value=0,
    end_value=10,
    p=1.0,
    q=1.0,
    score_type="polarization",
    kernel="power",
    dropna=False,
    B=2000,
    ci=0.95,
    random_state=42,
    keep_dists=False,
)
```

The main entry point. Computes point estimates and bootstrap CIs for the variable pair.

**Required arguments**

| Argument | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Survey data; one row per respondent |
| `first_variable` | `str` | Column name — becomes the row axis |
| `second_variable` | `str` | Column name — becomes the column axis |

**Keyword arguments**

| Argument | Default | Description |
|---|---|---|
| `start_value` | `0` | Minimum value on the response scale |
| `end_value` | `10` | Maximum value on the response scale |
| `p` | `1.0` | Distance exponent (`power`) or sigma bandwidth (`gaussian`) |
| `q` | `1.0` | Agreement exponent |
| `score_type` | `"polarization"` | `"polarization"` or `"consensus"` |
| `kernel` | `"power"` | `"power"` or `"gaussian"` |
| `dropna` | `False` | NaN handling flag (both columns are always inner-aligned) |
| `B` | `2000` | Number of bootstrap replications |
| `ci` | `0.95` | Confidence level (e.g. `0.95` → 95% CI) |
| `random_state` | `42` | Integer seed for reproducibility; pass `None` for random |
| `keep_dists` | `False` | If `True`, include raw bootstrap arrays in the result |

**Returns:** `dict` — see [Return Value Schemas](#return-value-schemas).

---

### `weight_matrix_generator`

```python
weight_matrix_generator(
    start_value=0,
    end_value=10,
    p=1.0,
    q=1.0,
    type="polarization",
    kernel="power",
)
```

Low-level function. Builds and returns the raw weight matrix as a NumPy array.

Useful for inspecting or customising the weighting geometry before running scores.

**Returns:** `np.ndarray` of shape `(N+1, N+1)`, values in `[-1, 1]`, max absolute value = 1.

---

### `dashboard_pair`

```python
dashboard_pair(
    df,
    x,
    y,
    *,
    score_type="polarization",
    start_value=0,
    end_value=10,
    p=1.0,
    q=1.0,
    kernel="power",
    dropna=False,
    B_boot=2000,
    ci=0.95,
    random_state=42,
)
```

Runs `calculate_scores` and displays a three-panel matplotlib figure:

| Panel | Content |
|---|---|
| **A** (top-left) | Horizontal bar chart of point estimates with bootstrap CI error bars |
| **B** (top-right) | Overlaid histograms of the bootstrap distributions for each score key |
| **C** (bottom, full-width) | Heatmap of the `W × P` matrix; diagonal or anti-diagonal separator; column contribution labels |

Also prints a text summary of the bootstrap results and sparsity diagnostic to stdout.

**Arguments** are identical to `calculate_scores`, with `B_boot` in place of `B`.

**Returns:** `{"scores": <calculate_scores result>}`

---

### `weight_matrix_visualization`

```python
weight_matrix_visualization(
    start_value=0,
    end_value=10,
    p=1.0,
    q=1.0,
    type="polarization",
    kernel="power",
)
```

Displays a single annotated heatmap of the weight matrix. Useful for understanding the effect of `p`, `q`, `type`, and `kernel` before running any data through the scorer.

**Returns:** `None`

---

## Return Value Schemas

### `calculate_scores` return value

```python
{
    "type": "polarization",          # or "consensus"

    "point": {
        # --- polarization ---
        "<first_variable>":  float,  # score from lower triangle (x > y)
        "<second_variable>": float,  # score from upper triangle (x < y)
        "overall":           float,  # total score (sum of both triangles)

        # --- consensus ---
        "negatives": float,          # score from anti-upper half (both low)
        "positives": float,          # score from anti-lower half (both high)
        "overall":   float,

        # --- diagnostics (both types) ---
        "_sparsity":  float,         # fraction of empty cells in the count matrix
        "_wp_matrix": np.ndarray,    # full W×P matrix (shape: N+1 × N+1)
        "_x_labels":  list[int],     # scale tick labels
        # polarization only:
        "_n_above": float,           # count of respondents in upper triangle
        "_n_below": float,           # count of respondents in lower triangle
        # consensus only:
        "_n_anti_above": float,
        "_n_anti_below": float,
    },

    "bootstrap": {
        "<key>": {
            "mean": float,           # bootstrap mean
            "se":   float,           # bootstrap standard error
            "ci":   [float, float],  # [lower, upper] percentile interval
            # "dist": np.ndarray     # present only if keep_dists=True
        },
        # one entry per public score key
    },

    "meta": {
        "B":      int,
        "ci":     float,
        "kernel": str,
    },
}
```

> **Diagnostic keys** (`_sparsity`, `_wp_matrix`, `_x_labels`, `_n_*`) are always present in `"point"` but are never included in `"bootstrap"` — they are not scalar scores and cannot be bootstrapped in the same way.

> **Sparsity warning:** when `_sparsity` is high (e.g. > 0.5), many cells of the count matrix are empty. This can bias scores downward and widen bootstrap CIs. Consider collapsing the scale or increasing sample size.

---

## Parameter Tuning Guide

### Choosing `score_type`

| Goal | Recommended |
|---|---|
| Detect opposing camps (high vs low on different variables) | `"polarization"` |
| Detect shared conviction (both high or both low) | `"consensus"` |

### Choosing `kernel`

| Kernel | When to use |
|---|---|
| `"power"` | Default; interpretable; monotone distance effect |
| `"gaussian"` | When you want a sharp threshold: cells very close to the diagonal/anti-diagonal get near-zero weight, but weight rises steeply and saturates beyond a distance of ~`p` |

### Choosing `p`

| Value (power kernel) | Effect |
|---|---|
| `p < 1` | Concave distance curve — moderate disagreements weighted heavily, extreme disagreements not much more |
| `p = 1` | Linear distance — proportional to distance |
| `p > 1` | Convex distance curve — only large disagreements count |

For the gaussian kernel, `p` = sigma: smaller `p` means the curve saturates faster (larger weight for even short distances from diagonal).

### Choosing `q`

| Value | Effect |
|---|---|
| `q = 0` | Agreement term disabled — pure distance scoring |
| `q = 1` | Linear agreement — default, balanced |
| `q > 1` | Only cells near the scale midpoint (polarization) or anti-diagonal (consensus) are weighted; extremes suppressed |

### Choosing `B`

| `B` | When appropriate |
|---|---|
| `200–500` | Quick exploration, prototyping |
| `2000` | Standard reporting |
| `5000+` | Publication-quality, narrow CIs |

---

## Examples

### Basic polarization score

```python
import pandas as pd
from polarization import calculate_scores

# df is your survey DataFrame
result = calculate_scores(df, "idemus", "idekemalist", score_type="polarization")

point = result["point"]
boot  = result["bootstrap"]

print(f"Overall polarization: {point['overall']:.4f}")
print(f"95% CI: {boot['overall']['ci']}")
print(f"SE: {boot['overall']['se']:.4f}")
```

Output:
```
Overall polarization: 0.6215
95% CI: [0.5665, 0.6848]
SE: 0.0298
```

### Consensus score with gaussian kernel

```python
result = calculate_scores(
    df, "ideleft", "ideright",
    score_type="consensus",
    kernel="gaussian",
    p=0.5,
    q=1.0,
    B=2000,
)
print(result["point"]["overall"])
```

### Inspect the weight matrix before scoring

```python
from polarization import weight_matrix_visualization

# See how weights look for polarization with a steep distance curve
weight_matrix_visualization(p=2.0, q=1.0, type="polarization", kernel="power")

# Compare with gaussian kernel
weight_matrix_visualization(p=0.3, q=1.0, type="polarization", kernel="gaussian")
```

### Full dashboard

```python
from polarization import dashboard_pair

dash = dashboard_pair(df, "idemus", "idekemalist", score_type="polarization", B_boot=2000)

# Access the underlying scores programmatically
overall = dash["scores"]["point"]["overall"]
```

### Accessing the raw WP matrix

```python
result = calculate_scores(df, "idemus", "idekemalist")
wp = result["point"]["_wp_matrix"]   # numpy array
labels = result["point"]["_x_labels"]

# e.g. inspect the most polarizing cell
i, j = divmod(wp.argmax(), wp.shape[1])
print(f"Highest cell: ({labels[i]}, {labels[j]}) = {wp[i, j]:.4f}")
```

### Comparing multiple pairs

```python
pairs = [
    ("idemus", "ideathe"),
    ("idemus", "idekemalist"),
    ("ideleft", "ideright"),
    ("idesec", "ideislamist"),
]

for x, y in pairs:
    r = calculate_scores(df, x, y, score_type="polarization", B=500)
    ci = r["bootstrap"]["overall"]["ci"]
    print(f"{x} × {y}: {r['point']['overall']:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### Keeping bootstrap distributions for custom analysis

```python
result = calculate_scores(
    df, "idemus", "idekemalist",
    keep_dists=True,
)

import matplotlib.pyplot as plt
dist = result["bootstrap"]["overall"]["dist"]
plt.hist(dist, bins=50)
plt.axvline(result["point"]["overall"], color="red", linestyle="--")
plt.title("Bootstrap distribution of overall polarization")
plt.show()
```
