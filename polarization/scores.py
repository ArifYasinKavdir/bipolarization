"""
Score calculation: point estimates and bootstrap confidence intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .weights import weight_matrix_generator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_by_masks(A: np.ndarray, masks: dict) -> np.ndarray:
    """Normalise each region of *A* independently by its own sum."""
    out = np.zeros_like(A, dtype=float)
    for m in masks.values():
        s = A[m].sum()
        if s > 0:
            out[m] = A[m] / s
    return out


def _single_run_score(
    df: pd.DataFrame,
    x: str,
    y: str,
    start_value: int,
    end_value: int,
    p: float,
    q: float,
    score_type: str,
    dropna: bool,
    kernel: str = "power",
) -> dict:
    """
    Compute a point-estimate score for the pair (*x*, *y*).

    Returns a dict whose keys depend on *score_type*:

    Polarization keys
    -----------------
    ``type``, ``<x>``, ``<y>``, ``overall``,
    ``_n_above``, ``_n_below``, ``_sparsity``, ``_wp_matrix``, ``_x_labels``

    Consensus keys
    --------------
    ``type``, ``negatives``, ``positives``, ``overall``,
    ``_n_anti_above``, ``_n_anti_below``, ``_sparsity``, ``_wp_matrix``, ``_x_labels``

    Keys prefixed with ``_`` are diagnostic / internal.
    """
    full_scale = range(start_value, end_value + 1)

    col_x = df[x].dropna().astype(int)
    col_y = df[y].dropna().astype(int)
    col_x, col_y = col_x.align(col_y, join="inner")

    count_matrix = (
        pd.crosstab(col_x, col_y)
        .reindex(index=full_scale, columns=full_scale, fill_value=0)
    )

    W = np.asarray(
        weight_matrix_generator(
            start_value=start_value, end_value=end_value,
            p=p, q=q, type=score_type, kernel=kernel,
        ),
        dtype=float,
    )
    A = count_matrix.to_numpy()

    if W.shape != A.shape:
        raise ValueError(f"weights shape {W.shape} must match table shape {A.shape}")

    I, J = np.indices(A.shape)
    n = A.shape[0]

    if score_type == "polarization":
        masks = {"above": I < J, "below": I > J, "diag": I == J}
        P = _normalize_by_masks(A, masks)
        WP = P * W
        return {
            "type": "polarization",
            x: float(WP[masks["below"]].sum()),
            y: float(WP[masks["above"]].sum()),
            "overall": float(WP.sum()),
            "_n_above": float(A[masks["above"]].sum()),
            "_n_below": float(A[masks["below"]].sum()),
            "_sparsity": float(np.mean(A == 0)),
            "_wp_matrix": WP,
            "_x_labels": list(full_scale),
        }

    elif score_type == "consensus":
        s = I + J
        masks = {
            "anti_above": s < (n - 1),
            "anti_below": s > (n - 1),
            "anti_diag":  s == (n - 1),
        }
        P = _normalize_by_masks(A, masks)
        WP = P * W
        return {
            "type": "consensus",
            "negatives": float(WP[masks["anti_above"]].sum()),
            "positives": float(WP[masks["anti_below"]].sum()),
            "overall": float(WP.sum()),
            "_n_anti_above": float(A[masks["anti_above"]].sum()),
            "_n_anti_below": float(A[masks["anti_below"]].sum()),
            "_sparsity": float(np.mean(A == 0)),
            "_wp_matrix": WP,
            "_x_labels": list(full_scale),
        }

    else:
        raise ValueError("score_type must be 'polarization' or 'consensus'")


def _bootstrap(
    df: pd.DataFrame,
    x: str,
    y: str,
    start_value: int,
    end_value: int,
    p: float,
    q: float,
    score_type: str,
    dropna: bool,
    B: int,
    ci: float,
    rng: np.random.Generator,
    kernel: str = "power",
) -> tuple[dict, dict]:
    """
    Run a non-parametric bootstrap around the point estimate.

    Returns
    -------
    point : dict
        Point-estimate result from :func:`_single_run_score`.
    summary : dict
        Per-key dict with keys ``mean``, ``se``, ``ci``, ``dist``.
    """
    point = _single_run_score(df, x, y, start_value, end_value, p, q, score_type, dropna, kernel)
    public_keys = [k for k in point if k != "type" and not k.startswith("_")]

    n = len(df)
    idx = np.arange(n)
    dist = {k: np.empty(B, dtype=float) for k in public_keys}

    for b in range(B):
        sample_idx = rng.choice(idx, size=n, replace=True)
        s = _single_run_score(df.iloc[sample_idx], x, y, start_value, end_value, p, q, score_type, dropna, kernel)
        for k in public_keys:
            dist[k][b] = s[k]

    alpha = (1 - ci) / 2
    lo_q, hi_q = 100 * alpha, 100 * (1 - alpha)

    summary = {}
    for k in public_keys:
        a = dist[k]
        summary[k] = {
            "mean": float(a.mean()),
            "se":   float(a.std(ddof=1)),
            "ci":   [float(np.percentile(a, lo_q)), float(np.percentile(a, hi_q))],
            "dist": a,
        }
    return point, summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_scores(
    df: pd.DataFrame,
    first_variable: str,
    second_variable: str,
    *,
    start_value: int = 0,
    end_value: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    score_type: str = "polarization",
    kernel: str = "power",
    dropna: bool = False,
    B: int = 2000,
    ci: float = 0.95,
    random_state: int | None = 42,
    keep_dists: bool = False,
) -> dict:
    """
    Calculate polarization or consensus scores for a pair of survey variables.

    The method builds a joint frequency table for (*first_variable*,
    *second_variable*), multiplies it by a weighted kernel matrix, and
    summarises the result as a scalar score per triangle (or per half of the
    anti-diagonal).  Bootstrap resampling provides standard errors and
    confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Survey data.  Each row is one respondent.
    first_variable : str
        Column name of the first variable (row axis of the crosstab).
    second_variable : str
        Column name of the second variable (column axis).
    start_value : int, default 0
        Minimum value on the response scale.
    end_value : int, default 10
        Maximum value on the response scale.
    p : float, default 1.0
        Distance kernel parameter (exponent for ``'power'``, sigma for
        ``'gaussian'``).
    q : float, default 1.0
        Agreement exponent.
    score_type : {'polarization', 'consensus'}, default 'polarization'
        Which scoring geometry to apply.
    kernel : {'power', 'gaussian'}, default 'power'
        Kernel function used for the distance term.
    dropna : bool, default False
        Whether to drop NaN values before scoring (currently informational;
        both columns are always dropna'd via alignment).
    B : int, default 2000
        Number of bootstrap replications.
    ci : float, default 0.95
        Confidence level for the bootstrap interval.
    random_state : int or None, default 42
        Seed for reproducible bootstrap samples.
    keep_dists : bool, default False
        If ``True``, include the raw bootstrap distribution arrays in the
        returned ``bootstrap`` dict under the ``'dist'`` key.

    Returns
    -------
    dict with keys:
        ``'type'`` : str
            Echo of *score_type*.
        ``'point'`` : dict
            Point estimates.  For ``'polarization'``: keys are
            *first_variable*, *second_variable*, and ``'overall'``.
            For ``'consensus'``: ``'negatives'``, ``'positives'``, ``'overall'``.
            Diagnostic keys (prefixed ``_``) are also present.
        ``'bootstrap'`` : dict
            Per-key bootstrap summary with ``'mean'``, ``'se'``, ``'ci'``
            (and ``'dist'`` if *keep_dists* is True).
        ``'meta'`` : dict
            ``'B'``, ``'ci'``, ``'kernel'``.

    Examples
    --------
    >>> result = calculate_scores(df, "idemus", "idekemalist",
    ...                           score_type="polarization", B=2000)
    >>> result["point"]["overall"]
    0.6215
    >>> result["bootstrap"]["overall"]["ci"]
    [0.5665, 0.6848]
    """
    rng = np.random.default_rng(random_state)
    point, boot = _bootstrap(
        df, first_variable, second_variable,
        start_value, end_value, p, q,
        score_type, dropna, B, ci, rng, kernel=kernel,
    )
    if not keep_dists:
        for k in boot:
            boot[k].pop("dist", None)
    return {
        "type":      point["type"],
        "point":     point,
        "bootstrap": boot,
        "meta":      {"B": B, "ci": ci, "kernel": kernel},
    }
