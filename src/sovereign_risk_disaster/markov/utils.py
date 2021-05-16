from math import erfc, sqrt

from numba import njit


@njit
def std_norm_cdf(x):
    return 0.5 * erfc(-x / sqrt(2))
