"""Tests for polarization.weights."""

import numpy as np
import pytest

from polarization.weights import weight_matrix_generator


@pytest.mark.parametrize("kernel", ["power", "gaussian"])
@pytest.mark.parametrize("p", [0.5, 1.0, 2.0])
def test_consensus_weights_take_the_sign_of_their_half(kernel, p):
    W = weight_matrix_generator(0, 4, p=p, q=1.0, type="consensus", kernel=kernel)
    i, j = np.indices(W.shape)

    assert np.isfinite(W).all()
    assert (W[i + j < 4] < 0).all()  # both answers low
    assert (W[i + j > 4] > 0).all()  # both answers high
    np.testing.assert_allclose(W[::-1, ::-1], -W)  # mirroring both answers flips the sign


@pytest.mark.parametrize("score_type", ["polarization", "consensus"])
@pytest.mark.parametrize("kernel", ["power", "gaussian"])
def test_weights_depend_only_on_position_within_the_scale(score_type, kernel):
    expected = weight_matrix_generator(0, 6, p=1.0, q=1.0, type=score_type, kernel=kernel)
    for start in (1, -3):  # a 1..7 Likert scale and a -3..3 bipolar scale
        W = weight_matrix_generator(start, start + 6, p=1.0, q=1.0, type=score_type, kernel=kernel)
        np.testing.assert_array_equal(W, expected)
