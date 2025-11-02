from typing import Type
from tqdm import trange

import numpy as np

from app.pricing.coxrossrubinstein import CRR
from app.modules.dividend_riskfree import Pairqr
from app.utils.convert import convert_to_numpy
from app.utils.types import ArrayLike


class VolCalibrator:
    """
    Implied Volatility calibrator
    """

    def __init__(self, model: Type[CRR], method: str = "CRR", **kwargs) -> None:  # type: ignore
        self.method = method
        self.model = model(**kwargs)  # type: ignore

    def bisection(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        qr: Pairqr,
        market_prices: ArrayLike,
        n: int | None = None,
        vol_low: float = 0.001,
        vol_high: float = 1.5,
        fidelity: float = 0.001,
        use_extrapolation: bool = False,  # Set to True to use extr
        **extr_kwargs,  # type: ignore
    ):
        """
        Calculates implied volatility for a vector of options using
        a vectorized bisection algorithm.
        """
        # Convert all inputs to numpy arrays
        S_np = convert_to_numpy(S)
        K_np = convert_to_numpy(K)
        T_np = convert_to_numpy(T)
        P_market = convert_to_numpy(market_prices)

        if not use_extrapolation:
            assert not n is None

        # Initialize volatility guesses as arrays
        num_options = len(S_np)
        v_low = np.full(num_options, vol_low)
        v_high = np.full(num_options, vol_high)
        v_mid = (v_low - v_high) / 2

        def _price_batch(vol_guess: np.ndarray):
            if use_extrapolation:
                return self.model.extr(S=S_np, K=K_np, T=T_np, qr=qr, vol=vol_guess, **extr_kwargs)  # type: ignore
            else:
                return self.model(S=S_np, K=K_np, T=T_np, qr=qr, vol=vol_guess, n=n)  # type: ignore

        # --- Pre-calculate prices at the boundaries ---
        # Note: qr.fit will be called inside price_batch on the first run
        price_low = _price_batch(v_low)
        price_high = _price_batch(v_high)
        price_mid = _price_batch(v_mid)

        # --- Error checking (optional but recommended) ---
        # Find options where the market price is outside the bracket
        failed_bracket = (P_market < price_low) | (P_market > price_high)
        if np.any(failed_bracket):
            print(
                f"Warning: {np.sum(failed_bracket)} options are outside the initial vol bracket."
            )
            # Clamp market prices to be within the bracket for stability
            P_market = np.clip(P_market, price_low + 1e-10, price_high - 1e-10)

        iterations = int(np.ceil(np.log((vol_high - vol_low) / fidelity) / np.log(2)))
        err = price_mid - P_market
        # --- Vectorized Bisection Loop ---
        for _ in trange(iterations, mininterval=0.5):
            v_mid = (v_low + v_high) / 2.0

            # This is the key: one call prices ALL options
            price_mid = _price_batch(v_mid)

            # Calculate error for all options
            err = price_mid - P_market

            # Vectorized update:
            # Where err > 0 (price is too high), our new v_high is v_mid
            # Where err <= 0 (price is too low), our new v_low is v_mid
            is_too_high = err > 0
            v_high = np.where(is_too_high, v_mid, v_high)
            v_low = np.where(is_too_high, v_low, v_mid)

        # The result is the midpoint of the final bracket
        v_final = (v_low + v_high) / 2.0

        # Return NaN for options that failed the initial bracketing
        # v_final[failed_bracket] = np.nan

        return v_final
