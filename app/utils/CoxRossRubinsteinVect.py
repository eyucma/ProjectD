"""
This module contains Binomial pricing function using CoxRossRubinstein vectorized
(no-jump forward adjusted)
"""

import numpy as np

from app.utils.types import ArrayLike
from app.modules.dividend_riskfree import Dividend, RiskFree
from app.utils.convert import convert_to_numpy

small = 1e-6


def crr(
    S: ArrayLike,  # pylint: disable=invalid-name
    K: ArrayLike,  # pylint: disable=invalid-name
    T: ArrayLike,  # pylint: disable=invalid-name
    r: ArrayLike | RiskFree,
    q: ArrayLike | Dividend,
    vol: ArrayLike,
    n: int,
    call: bool = True,
    american: bool = False,
) -> np.ndarray:
    """
    Compute moment matching binomial options price, vectorized. Crucially inputs must all be arrays except
    n,call,american and r/q if using specialt class. Units in years time

    Parameters:
        price: observed option price
        S: underlying spot price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        q: dividend yield
        vol: volatility
        n: number to discretize time interval
        call: True for call False for put
        american: True for American option, False for European

    Returns:
        Option price
    """

    S = convert_to_numpy(S)
    K = convert_to_numpy(K)
    T = convert_to_numpy(T)

    vol = convert_to_numpy(vol)

    num_options = len(S)
    dt = T / n

    # case of non Dividend class q
    if not (isinstance(q, Dividend) or isinstance(r, RiskFree)):
        q_rate = convert_to_numpy(q)
        r = convert_to_numpy(r)

        # --- 2. Calculate Parameters (Vectorized across num_options) ---

        # Risk-neutral drift (log-return mean)
        m = r - q_rate - 0.5 * vol**2

        # Log-return magnitude 'g' (the up/down jump size)
        g = np.sqrt(dt * (m**2 * dt + vol**2))  # Simplified g expression

        # Risk-neutral probability (vectorized)
        p = 0.5 * (m * dt / g + 1.0)

        # Pre-calculate discount factor and probability
        discount_factor = np.exp(-r * dt)
        prob_down = 1.0 - p

        # --- 3. Initialize Lattice and Terminal Payoff ---
        # The lattice V is 3D: (num_options, n+1 steps, n+1 nodes)
        # We only need (num_options, n+1) for backward induction, so we optimize V's size.
        # To use slicing, v must be large enough for the largest row (n+1)

        v = np.zeros((num_options, n + 1), dtype=float)

        # Array representing the stock price at the final time step (T=n)
        # S_T is (num_options, n+1)
        # S_T[:, j] = S * exp(g * (2 * j - n))

        # Use array broadcasting to compute all final stock prices at once
        j_arr = np.arange(n + 1)
        terminal_s = S[:, None] * np.exp(g[:, None] * (2 * j_arr[None, :] - n))

        # Terminal Payoff (Vectorized)
        if call:
            v = np.clip(terminal_s - K[:, None], 0, None)
        else:
            v = np.clip(K[:, None] - terminal_s, 0, None)

        # --- 4. Backward Induction (Outer Python loop, Inner NumPy Vectorization) ---

        for i in reversed(range(n)):
            # Calculate Continuation Value (Expectation, discounted)
            # V[:, i + 1] contains the (i+2) values at time i+1 (from 0 to i+1)

            # We need the values V[:, i+1] but only the first i+1 elements (down path) and
            # the last i+1 elements (up path) are relevant for the current step i.

            # Slicing the lattice for the relevant part of the next step (i+1)
            # Note: We must reduce the size of the array used for the next step
            # to ensure V is not accessed out of bounds later on.

            # 1. Expected Value (Dot product style operation)
            # This calculates p*V_up + (1-p)*V_down for all num_options
            continuation_value = (
                p[:, None] * v[:, 1 : i + 2] + prob_down[:, None] * v[:, 0 : i + 1]
            )

            # 2. Discounting
            continuation_value = discount_factor[:, None] * continuation_value

            # American Option Check
            if american:
                # Calculate Intrinsic Value at time i (Vectorized across num_options AND nodes j)

                # Stock price at time i (S_i is num_options x i+1)
                j_arr_i = np.arange(i + 1)
                current_s = S[:, None] * np.exp(g[:, None] * (2 * j_arr_i[None, :] - i))

                if call:
                    intrinsic_value = np.clip(current_s - K[:, None], 0, None)
                else:  # Put
                    intrinsic_value = np.clip(K[:, None] - current_s, 0, None)

                # Maximize Intrinsic Value vs. Continuation Value
                v = np.maximum(intrinsic_value, continuation_value)
            else:
                # European: Assign the continuation value directly
                v = continuation_value

        # --- 5. Return Final Price ---
        # The final prices for all options are in the first column of the
        # last computed v (which now effectively represents v[:, 0])

        # v is now (num_options, 1) or (num_options,)
        return v[:, 0]

    assert isinstance(q, Dividend) and isinstance(r, RiskFree)
    if not q.fitted:
        q.fit(T, n)
    if not r.fitted:
        r.fit(T, n)

    g = vol * np.sqrt(T / n)
    u = np.exp(g)
    d = np.exp(-g)
    div_payments = q.div
    discount_factors = r.discount_factors  # (len(T),n+1)
    principal_div = (div_payments * discount_factors).sum(axis=1)  # (len(T))
    inter_disc = r.intermediate_discount  # (len(T),n)
    S_adj = np.clip(S - principal_div, small, None)  # pylint: disable=invalid-name
    p = (1 / inter_disc - d[:, None]) / (u[:, None] - d[:, None])
    v = np.zeros(
        (len(T), n + 1, n + 1)
    )  # stores the tree as (option_no,ith_time,jth_price)
    if call:
        v[:, -1, :] = np.clip(
            S_adj[:, None] * np.exp(g[:,None] * (2 * np.arange(n + 1)[None, :] - n))
            - K[:, None],
            0,
            None,
        )
    else:
        v[:, -1, :] = np.clip(
            K[:, None]
            - S_adj[:, None] * np.exp(g[:,None]  * (2 * np.arange(n + 1)[None, :] - n)),
            0,
            None,
        )

    for i in reversed(range(n)):
        candidate = inter_disc[:, i, None] * (
            p[:, i, None] * v[:, i + 1, 1 : i + 2]
            + (1 - p[:, i, None]) * v[:, i + 1, : i + 1]
        )
        if american:
            if call:
                v[:, i, : i + 1] = np.maximum(
                    S_adj[:, None] * np.exp(g[:,None] * (2 * np.arange(i + 1)[None, :] - i))
                    - K[:, None],
                    candidate,
                )
            else:
                v[:, i, : i + 1] = np.maximum(
                    K[:, None]
                    - S_adj[:, None]
                    * np.exp(g[:,None] * (2 * np.arange(i + 1)[None, :] - i)),
                    candidate,
                )
        else:
            v[:, i, : i + 1] = candidate

    return v
