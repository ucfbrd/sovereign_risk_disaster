import numpy as np
from numba import njit
from quantecon.markov.core import MarkovChain

from .utils import std_norm_cdf


def tauchen(rho, sigma_u, h=0, m=3, n=7):
    r"""
    Computes a Markov chain associated with a discretized version of
    the linear Gaussian AR(1) process

    Adapted from: https://quanteconpy.readthedocs.io/en/latest/_modules/quantecon/markov/approximation.html#tauchen

    .. math::

        y_{t+1} = \rho y_t - \ksi h + u_{t+1}

    using Tauchen's method. Here :math:`{u_t}` is an i.i.d. Gaussian process
    with zero mea, where :math:`\ksi` is an indicator function that is equal to one when the economy is hit by a disatser

    Parameters
    ----------
    h : scalar(float)
        The probability that a disaster happens
    rho : scalar(float)
        The autocorrelation coefficient
    sigma_u : scalar(float)
        The standard deviation of the random process
    m : scalar(int), optional(default=3)
        The number of standard deviations to approximate out to
    n : scalar(int), optional(default=7)
        The number of states to use in the approximation

    Returns
    -------

    mc : MarkovChain
        An instance of the MarkovChain class that stores the transition
        matrix and state values returned by the discretization method

    """

    # standard deviation of non disaster y_t
    std_y = np.sqrt(sigma_u ** 2 / (1 - rho ** 2))

    # top of discrete state space for non disaster y_t
    x_max = m * std_y

    # bottom of discrete state space for non disaster y_t
    x_min = -x_max

    # discretized state space for non disaster y_t
    x = np.linspace(x_min, x_max, n)

    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step
    P = np.empty((n, n))

    # approximate Markov transition matrix for non disaster y_t
    _fill_tauchen(x, P, n, rho, sigma_u, half_step)

    # shifts the state values by a probability of disaster h
    mu = -h / (1 - rho)

    mc = MarkovChain(P, state_values=x + mu)

    return mc


@njit
def _fill_tauchen(x, P, n, rho, sigma, half_step):
    for i in range(n):
        P[i, 0] = std_norm_cdf((x[0] - rho * x[i] + half_step) / sigma)
        P[i, n - 1] = 1 - std_norm_cdf((x[n - 1] - rho * x[i] - half_step) / sigma)
        for j in range(1, n - 1):
            z = x[j] - rho * x[i]
            P[i, j] = std_norm_cdf((z + half_step) / sigma) - std_norm_cdf((z - half_step) / sigma)
