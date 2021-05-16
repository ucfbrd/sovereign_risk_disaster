from numba import jit


@jit(nopython=True)
def u_crra(c, gamma):
    return c ** (1 - gamma) / (1 - gamma)
