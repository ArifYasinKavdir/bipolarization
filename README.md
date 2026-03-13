# bipolarization

A Python library for detecting and quantifying **polarization** and **consensus** in survey data using weighted kernel scoring and bootstrap inference.

This library is written based on polarization research conducted by **Zübeyir Nişancı**, **Belkıs Yüce** and **Arif Yasin Kavdır**.

## Installation

```bash
pip install bipolarization
```

**Dependencies:** `numpy`, `pandas`, `matplotlib`

---

## What it does

Given two Likert-scale survey variables, this library answers questions like:

- *How polarized are Muslim and Kemalist identity self-ratings among respondents?*
- *Is there more consensus or divergence between left-wing and right-wing self-placement?*
- *Which group contributes more to the observed polarization?*

The core idea: build a joint frequency table of two variables, multiply each cell by a geometrically derived weight, and sum the result into a scalar score. Confidence intervals are produced via non-parametric bootstrap resampling.

---

## Quick Start

```python
import pandas as pd
from polarization import calculate_scores, dashboard_pair

# Polarization score between two variables
result = calculate_scores(df, "idemus", "idekemalist", score_type="polarization")

print(f"Overall polarization: {result['point']['overall']:.4f}")
print(f"95% CI: {result['bootstrap']['overall']['ci']}")
```

### Visual dashboard

```python
from polarization import dashboard_pair

dashboard_pair(df, "idemus", "idekemalist", score_type="polarization", B_boot=2000)
```

This produces a three-panel figure: point estimates with CI error bars, bootstrap distributions, and a heatmap of the weighted probability matrix.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `score_type` | `"polarization"` | `"polarization"` or `"consensus"` |
| `kernel` | `"power"` | `"power"` or `"gaussian"` |
| `p` | `1.0` | Distance exponent (power) or sigma bandwidth (gaussian) |
| `q` | `1.0` | Agreement exponent |
| `start_value` | `0` | Minimum value on the response scale |
| `end_value` | `10` | Maximum value on the response scale |
| `B` | `2000` | Number of bootstrap replications |
| `ci` | `0.95` | Confidence level |

---

## Polarization vs Consensus

**Polarization** — detects opposing camps. High score means respondents split into two groups rating the variables in opposite directions.

**Consensus** — detects shared conviction. High score means respondents tend to rate both variables similarly (both high or both low).

---

## Comparing Multiple Pairs

```python
pairs = [
    ("idemus", "ideathe"),
    ("idemus", "idekemalist"),
    ("ideleft", "ideright"),
]

for x, y in pairs:
    r = calculate_scores(df, x, y, score_type="polarization", B=500)
    ci = r["bootstrap"]["overall"]["ci"]
    print(f"{x} × {y}: {r['point']['overall']:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")
```

---

## Full Documentation

See [polarization/DOCUMENTATION.md](polarization/DOCUMENTATION.md) for the complete API reference, return value schemas, parameter tuning guide, and examples.

---

## License

MIT
