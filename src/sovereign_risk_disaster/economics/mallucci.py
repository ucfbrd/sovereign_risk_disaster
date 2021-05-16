import numpy as np
from numba import float64
from numba.experimental import jitclass

from sovereign_risk_disaster.economics.utility import u_crra

mallucci_data = [
    ("B", float64[:]),
    ("P", float64[:, :]),
    ("y", float64[:]),
    ("beta", float64),
    ("gamma", float64),
    ("r", float64),
    ("rho", float64),
    ("eta", float64),
    ("theta", float64),
    ("def_y", float64[:]),
]


@jitclass(mallucci_data)
class MallucciModel:
    """
    Adapted from https://python-advanced.quantecon.org/arellano.html
    Parameters
    ----------
    B : vector(float64)
        A grid for bond holdings
    P : matrix(float64)
        The transition matrix for a country's output
    y : vector(float64)
        The possible output states
    beta : float
        Time discounting parameter
    gamma : float
        Risk-aversion parameter
    r : float
        int lending rate
    rho : float
        Persistence in the income process
    eta : float
        Standard deviation of the income process
    theta : float
        Probability of re-entering financial markets in each period, corresponds to lambda in the paper
    """

    def __init__(self, B, P, y, beta=0.9, gamma=2.0, r=0.0451, rho=0.945, eta=0.025, theta=0.33, u=u_crra):

        # Save parameters
        self.B, self.P, self.y = B, P, y
        self.beta, self.gamma, self.r, = (
            beta,
            gamma,
            r,
        )
        self.rho, self.eta, self.theta = rho, eta, theta

        # define Utility Function
        self.u = u

        # Compute the mean output
        self.def_y = np.minimum(0.969 * np.mean(y), y)

    def bellman_default(self, iy, EVd, EV):
        """
        The RHS of the Bellman equation when the country is in a
        defaulted state on their debt
        """

        # Compute continuation value
        zero_ind = len(self.B) // 2
        cont_value = self.theta * EV[iy, zero_ind] + (1 - self.theta) * EVd[iy]

        return self.u(self.def_y[iy], self.gamma) + self.gamma * cont_value

    def bellman_non_default(self, iy, iB, q, EV, iB_tp1_star=-1):
        """
        The RHS of the Bellman equation when the country is not in a
        defaulted state on their debt
        """

        # Compute the RHS of Bellman equation
        if iB_tp1_star < 0:
            iB_tp1_star = self.compute_savings_policy(iy, iB, q, EV)
        c = max(self.y[iy] - q[iy, iB_tp1_star] * self.B[iB_tp1_star] + self.B[iB], 1e-14)

        return self.u(c, self.gamma) + self.beta * EV[iy, iB_tp1_star]

    def compute_savings_policy(self, iy, iB, q, EV):
        """
        Finds the debt/savings that maximizes the value function
        for a particular state given prices and a value function
        """

        # Compute the RHS of Bellman equation
        current_max = -1e14
        iB_tp1_star = 0
        for iB_tp1, B_tp1 in enumerate(self.B):
            c = max(self.y[iy] - q[iy, iB_tp1] * self.B[iB_tp1] + self.B[iB], 1e-14)
            m = self.u(c, self.gamma) + self.beta * EV[iy, iB_tp1]

            if m > current_max:
                iB_tp1_star = iB_tp1
                current_max = m

        return iB_tp1_star
