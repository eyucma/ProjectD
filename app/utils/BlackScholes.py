import numpy as np
import pandas as pd

from scipy.stats import norm


def a(i: int) -> float:
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


def convert_to_numpy(
    data: float | list | np.ndarray | pd.Series | pd.DataFrame,
) -> float | int | np.ndarray:
    """Auxiliary function to ensure input is a NumPy array or float."""
    if isinstance(data, (float, int)):
        return data
    elif isinstance(data, (list, pd.Series, pd.DataFrame)):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")


def phi_pos(x_val: float | np.ndarray) -> float | np.ndarray:
    """
    Auxilary function for phi
    """
    t = 1 / (1 + x_val / np.sqrt(8))
    sum_terms = sum(a(i) * t**i for i in range(10))
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
            result[is_pos] = phi_pos(x[is_pos])

        # 2. Handle zero values
        result[is_zero] = 0.5

        # 3. Handle negative values (1 - N(-x) where -x is positive)
        if np.any(is_neg):
            # Apply phi_pos to the ABSOLUTE value of the negative inputs
            result[is_neg] = 1 - phi_pos(np.abs(x[is_neg]))

        return result

    if x == 0:
        return 0.5
    elif x > 0:
        t = 1 / (1 + x / np.sqrt(8))
        return 1 - t / 2 * np.exp(-(x**2) / 2 + sum(a(i) * t**i for i in range(10)))
    else:
        return 1 - phi(-x)


def BlackScholes(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    vol: float | np.ndarray,
    call=True,
    approx=True,
) -> float | np.ndarray:
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
        option_type: 'call' or 'put'

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
    else:
        return K * np.exp(-r * T) * cdf(-dm) - S * np.exp(-q * T) * cdf(-dp)
