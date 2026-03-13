"""
Weight matrix construction for polarization and consensus scoring.
"""

import numpy as np


def weight_matrix_generator(
    start_value: int = 0,
    end_value: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    type: str = "polarization",
    kernel: str = "power",
) -> np.ndarray:
    """
    Generate a normalised weight matrix for polarization or consensus scoring.

    Parameters
    ----------
    start_value : int
        Lower bound of the response scale.
    end_value : int
        Upper bound of the response scale.
    p : float
        Distance exponent (power kernel) or sigma bandwidth (gaussian kernel).
    q : float
        Agreement exponent — controls steepness of the agreement term.
    type : {'polarization', 'consensus'}
        Which scoring geometry to use.

        * ``'polarization'``:
          - *d* = normalised absolute distance between the two responses.
          - *a* = how far the pair is from both extremes (highest at midpoints).

        * ``'consensus'``:
          - *d* = how far above the scale midpoint the pair sum lies.
          - *a* = normalised agreement (inverse of absolute difference).

    kernel : {'power', 'gaussian'}
        How *d* is mapped to a weight.

        * ``'power'``    : ``M = (d**p) * (a**q)``
        * ``'gaussian'`` : ``M = (1 - exp(-d² / 2p²)) * sign(d) * (a**q)``

    Returns
    -------
    np.ndarray
        Square matrix of shape ``(N+1, N+1)`` where ``N = end_value - start_value``,
        normalised so that the maximum absolute value equals 1.

    Raises
    ------
    ValueError
        If *type* or *kernel* is not recognised.
    """
    N = end_value - start_value
    i = np.arange(start_value, end_value + 1)
    j = np.arange(start_value, end_value + 1)
    I, J = np.meshgrid(i, j)

    if type == "polarization":
        d = np.abs(I - J) / N
        a = 1 - (np.abs((I + J) - N) / N)
    elif type == "consensus":
        d = ((I + J) / N) - 1
        a = (N - np.abs(I - J)) / N
    else:
        raise ValueError("type must be 'polarization' or 'consensus'")

    if kernel == "power":
        M = (d ** p) * (a ** q)
    elif kernel == "gaussian":
        g = 1 - np.exp(-(d ** 2) / (2 * p ** 2))
        g = g * np.sign(d)
        M = g * (a ** q)
    else:
        raise ValueError("kernel must be 'power' or 'gaussian'")

    max_val = np.abs(M).max()
    if max_val > 0:
        M = M / max_val
    return M
