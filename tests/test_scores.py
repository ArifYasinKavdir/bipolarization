"""Tests for polarization.scores."""

import numpy as np
import pandas as pd
import pytest

from polarization import scores
from polarization.scores import calculate_scores

# Rows i = X (0..4), columns j = Y (0..4); each cell counts respondents.
TOY_TABLE = np.array([
    [3, 0, 1, 1, 4],
    [0, 1, 0, 1, 0],
    [1, 0, 2, 0, 0],
    [1, 1, 0, 0, 0],
    [3, 0, 0, 0, 1],
])


def _toy_frame() -> pd.DataFrame:
    """
    One row per respondent in TOY_TABLE (floats, so answers can be blanked).

    Rows are shuffled, keeping their labels, so row order differs from table
    order and the index is not monotonic.
    """
    i, j = np.indices(TOY_TABLE.shape)
    counts = TOY_TABLE.ravel()
    df = pd.DataFrame({
        "X": np.repeat(i.ravel(), counts).astype(float),
        "Y": np.repeat(j.ravel(), counts).astype(float),
    })
    return df.sample(frac=1, random_state=0)


def _toy_frame_missing_y() -> pd.DataFrame:
    """The toy respondents, three of whom answered X but not Y."""
    df = _toy_frame()
    df.loc[[2, 7, 16], "Y"] = np.nan
    return df


@pytest.mark.parametrize(
    "data",
    [
        {"X": [0, 0, 1, 2, 4, 3, 0, 4], "Y": [4, 0, 1, 2, 0, 3, 4, 4]},
        {"X": [0, 0, 1, 2, 4, 3, 0, 4], "Y": [4, 0, np.nan, 2, 0, 3, np.nan, 4]},
    ],
    ids=["complete", "nan-in-one-column"],
)
def test_bootstrap_runs(data):
    result = calculate_scores(
        pd.DataFrame(data), "X", "Y",
        start_value=0, end_value=4, B=50, keep_dists=True,
    )

    assert result["bootstrap"].keys() == {"X", "Y", "overall", "X_per_person", "Y_per_person"}
    for summary in result["bootstrap"].values():
        assert summary["dist"].shape == (50,)
        assert np.isfinite(summary["dist"]).all()
        assert summary["ci"][0] <= summary["ci"][1]


def test_point_estimates_match_hand_computed_values():
    result = calculate_scores(
        _toy_frame(), "X", "Y",
        start_value=0, end_value=4, p=1.0, q=1.0,
        score_type="polarization", kernel="power", formula="global", B=50,
    )

    # W[i, j] = |i - j| / 4 * (1 - |i + j - 4| / 4); each score is the sum of
    # W * counts over its triangle, divided by the 20 respondents.
    point = result["point"]
    assert point["X"] == pytest.approx(0.215625)  # below the diagonal: 4.3125 / 20
    assert point["Y"] == pytest.approx(0.265625)  # above the diagonal: 5.3125 / 20
    assert point["overall"] == pytest.approx(0.48125)


@pytest.mark.parametrize(
    "score_type, formula",
    [("polarization", "global"), ("polarization", "per_triangle"), ("consensus", "global")],
)
def test_bootstrap_replicates_match_direct_row_resample(score_type, formula):
    seed, B = 7, 50
    df = _toy_frame_missing_y()
    result = calculate_scores(
        df, "X", "Y",
        start_value=0, end_value=4, score_type=score_type, formula=formula,
        B=B, random_state=seed, keep_dists=True,
    )

    rows = df.to_numpy()
    rows = rows[~np.isnan(rows).any(axis=1)]  # listwise deletion
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(B):
        resample = rng.choice(rows, size=len(rows), replace=True)  # whole rows
        replicates.append(scores._single_run_score(
            pd.DataFrame(resample, columns=["X", "Y"]), x="X", y="Y",
            start_value=0, end_value=4, p=1.0, q=1.0, score_type=score_type,
            dropna=False, kernel="power", formula=formula,
        ))

    for key, summary in result["bootstrap"].items():
        assert summary["dist"] == pytest.approx([r[key] for r in replicates])


def test_every_replicate_counts_the_complete_pairs(monkeypatch):
    df = _toy_frame_missing_y()
    df.loc[0, "X"] = 99.0  # off-scale code (e.g. "don't know"): dropped like a missing answer
    n = 20 - 3 - 1

    table_totals = []
    score_counts = scores._score_counts

    def recording_score_counts(A, *args, **kwargs):
        table_totals.append(int(A.sum()))
        return score_counts(A, *args, **kwargs)

    monkeypatch.setattr(scores, "_score_counts", recording_score_counts)
    calculate_scores(df, "X", "Y", start_value=0, end_value=4, B=50)

    # The point-estimate table, then one table per replicate.
    assert table_totals == [n] * 51
