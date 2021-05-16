import numpy as np

from numba import jit, int64


@jit(nopython=True)
def solve(model, tol=1e-8, maxiter=10000):
    """
    This function computes the optimal policy and value functions
    Adapted from: https://python-advanced.quantecon.org/arellano.html
    """
    # Unpack certain parameters for simplification
    beta, gamma, r, theta = model.beta, model.gamma, model.r, model.theta
    B = np.ascontiguousarray(model.B)
    P, y = np.ascontiguousarray(model.P), np.ascontiguousarray(model.y)
    nB, ny = B.size, y.size

    # Allocate space
    iBstar = np.zeros((ny, nB), int64)
    default_prob = np.zeros((ny, nB))
    default_states = np.zeros((ny, nB))
    q = np.ones((ny, nB)) * 0.95
    Vd = np.zeros(ny)
    Vnd, V, Vupd = np.zeros((ny, nB)), np.zeros((ny, nB)), np.zeros((ny, nB))

    it = 0
    dist = 10.0
    while (it < maxiter) and (dist > tol):

        # Compute expectations used for this iteration
        EV = P @ V
        EVd = P @ Vd

        for iy in range(ny):
            # Update value function for default state
            Vd[iy] = model.bellman_default(iy, EVd, EV)

            for iB in range(nB):
                # Update value function for non-default state
                iBstar[iy, iB] = model.compute_savings_policy(iy, iB, q, EV)
                Vnd[iy, iB] = model.bellman_nondefault(iy, iB, q, EV, iBstar[iy, iB])

        # Once value functions are updated, can combine them to get
        # the full value function
        Vd_compat = np.reshape(np.repeat(Vd, nB), (ny, nB))
        Vupd[:, :] = np.maximum(Vnd, Vd_compat)  # TODO: Optimize over a continuous default rather than discrete

        # Can also compute default states and update prices
        default_states[:, :] = 1.0 * (Vd_compat > Vnd)
        default_prob[:, :] = P @ default_states
        q[:, :] = (1 - default_prob) / (1 + r)

        # Check tolerance etc...
        dist = np.max(np.abs(Vupd - V))
        V[:, :] = Vupd[:, :]
        it += 1

    return V, Vnd, Vd, iBstar, default_prob, default_states, q
