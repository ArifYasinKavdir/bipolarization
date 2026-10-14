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
