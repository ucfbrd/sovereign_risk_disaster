import numpy as np
import quantecon as qe


def simulate(model, T, default_states, iBstar, q, y_init=None, B_init=None):
    """
    Simulates a sovereign debt model
    Adapted from: https://python-advanced.quantecon.org/arellano.html

    Parameters
    ----------
    model: MallucciModel
        An instance of the Arellano model with the corresponding parameters
    T: integer
        The number of periods that the model should be simulated
    default_states: array(float64, 2)
        A matrix of 0s and 1s that denotes whether the country was in
        default on their debt in that period (default = 1)
    iBstar: array(float64, 2)
        A matrix which specifies the debt/savings level that a country holds
        during a given state
    q: array(float64, 2)
        A matrix that specifies the price at which a country can borrow/save
        for a given state
    y_init: integer
        Specifies which state the income process should start in
    B_init: integer
        Specifies which state the debt/savings state should start

    Returns
    -------
    y_sim: array(float64, 1)
        A simulation of the country's income
    B_sim: array(float64, 1)
        A simulation of the country's debt/savings
    q_sim: array(float64, 1)
        A simulation of the price required to have an extra unit of
        consumption in the following period
    default_sim: array(bool, 1)
        A simulation of whether the country was in default or not
    """
    # Find index i such that Bgrid[i] is approximately 0
    zero_B_index = np.searchsorted(model.B, 0.0)

    # Set initial conditions
    in_default = False
    max_y_default = 0.969 * np.mean(model.y)
    if y_init == None:
        y_init = np.searchsorted(model.y, model.y.mean())
    if B_init == None:
        B_init = zero_B_index

    # Create Markov chain and simulate income process
    mc = qe.MarkovChain(model.P, model.y)
    y_sim_indices = mc.simulate_indices(T + 1, init=y_init)

    # Allocate memory for remaining outputs
    Bi = B_init
    B_sim = np.empty(T)
    y_sim = np.empty(T)
    q_sim = np.empty(T)
    default_sim = np.empty(T, dtype=bool)

    # Perform simulation
    for t in range(T):
        yi = y_sim_indices[t]

        # Fill y/B for today
        if not in_default:
            y_sim[t] = model.y[yi]
        else:
            y_sim[t] = np.minimum(model.y[yi], max_y_default)
        B_sim[t] = model.B[Bi]
        default_sim[t] = in_default

        # Check whether in default and branch depending on that state
        if not in_default:
            if default_states[yi, Bi] > 1e-4:
                in_default = True
                Bi_next = zero_B_index
            else:
                Bi_next = iBstar[yi, Bi]
        else:
            Bi_next = zero_B_index
            if np.random.rand() < model.θ:
                in_default = False

        # Fill in states
        q_sim[t] = q[yi, Bi_next]
        Bi = Bi_next

    return y_sim, B_sim, q_sim, default_sim
