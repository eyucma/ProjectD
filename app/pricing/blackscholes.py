"""
This module contains Black-Scholes pricing function
"""

import numpy as np

from scipy.stats import norm


from app.utils.convert import convert_to_numpy
from app.utils.types import ArrayLike


def _a(i: int) -> float:
    """
    Auxilary function to for phi
    """
    r = [
        -1.26551223,
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]
    return r[i]


def _phi_pos(x_val: float | np.ndarray) -> float | np.ndarray:
    """
    Auxilary function for phi
    """
    t = 1 / (1 + x_val / np.sqrt(8))
    sum_terms = sum(_a(i) * t**i for i in range(10))
    return 1 - t / 2 * np.exp(-(x_val**2) / 2 + sum_terms)


def phi(x: float | np.ndarray) -> float | np.ndarray:
    """
    Approximation of the standard normal cdf for fast computation
    """
    if isinstance(x, np.ndarray):
        # Create a boolean mask for positive and negative values
        is_pos = x > 0
        is_neg = x < 0
        is_zero = x == 0

        # Initialize result array
        result = np.zeros_like(x, dtype=float)

        # 1. Handle positive values
        if np.any(is_pos):
            result[is_pos] = _phi_pos(x[is_pos])

        # 2. Handle zero values
        result[is_zero] = 0.5

        # 3. Handle negative values (1 - N(-x) where -x is positive)
        if np.any(is_neg):
            # Apply phi_pos to the ABSOLUTE value of the negative inputs
            result[is_neg] = 1 - _phi_pos(np.abs(x[is_neg]))

        return result

    if x == 0:
        return 0.5
    if x > 0:
        t = 1 / (1 + x / np.sqrt(8))
        return 1 - t / 2 * np.exp(-(x**2) / 2 + sum(_a(i) * t**i for i in range(10)))
    return 1 - phi(-x)


def bs(
    S: ArrayLike | float | int,  # pylint: disable=invalid-name
    K: ArrayLike | float | int,  # pylint: disable=invalid-name
    T: ArrayLike | float | int,  # pylint: disable=invalid-name
    r: ArrayLike | float | int,
    q: ArrayLike | float | int,
    vol: ArrayLike | float | int,
    call: bool = True,
    approx: bool = True,
) -> np.ndarray:
    """
    Compute BlackScholes options price. Units in years time

    Parameters:
        price: observed option price
        S: underlying spot price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        q: dividend yield
        vol: volatility
        call: True for call False for put

    Returns:
        Option price
    """

    # --- Input Conversion ---
    S = convert_to_numpy(S)
    K = convert_to_numpy(K)
    T = convert_to_numpy(T)
    r = convert_to_numpy(r)
    q = convert_to_numpy(q)
    vol = convert_to_numpy(vol)

    cdf = phi if approx else norm.cdf

    dp = (np.log(S / K) + (r - q + vol**2 / 2) * T) / vol / np.sqrt(T)
    dm = (np.log(S / K) + (r - q - vol**2 / 2) * T) / vol / np.sqrt(T)
    if call:
        return S * np.exp(-q * T) * cdf(dp) - K * np.exp(-r * T) * cdf(dm)
    return K * np.exp(-r * T) * cdf(-dm) - S * np.exp(-q * T) * cdf(-dp)


def bs76(
    F: ArrayLike | float | int,  # pylint: disable=invalid-name
    k: ArrayLike | float | int,  # pylint: disable=invalid-name
    T: ArrayLike | float | int,  # pylint: disable=invalid-name
    DF: ArrayLike | float | int,
    vol: ArrayLike | float | int,
    call: bool = True,
    approx: bool = True,
) -> np.ndarray:
    """
    Compute BlackScholes options price. Units in years time

    Parameters:
        price: observed option price
        S: underlying spot price
        K: strike price
        T: time to maturity (in years)
        r: risk-free rate
        q: dividend yield
        vol: volatility
        call: True for call False for put

    Returns:
        Option price
    """

    # --- Input Conversion ---
    S = convert_to_numpy(F)
    k = convert_to_numpy(k)
    T = convert_to_numpy(T)
    DF = convert_to_numpy(DF)
    vol = convert_to_numpy(vol)

    cdf = phi if approx else norm.cdf

    dp = (k + vol**2 / 2 * T) / vol / np.sqrt(T)
    dm = (k - vol**2 / 2 * T) / vol / np.sqrt(T)
    if call:
        return DF*F*(cdf(dp) -  np.exp(-k) * cdf(dm))
    return DF * F*(np.exp(-k) *cdf(-dm) - cdf(-dp))
