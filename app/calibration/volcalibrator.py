# pylint: disable=C0103, R0913, R0914, R0915, R0917
"""
Class for calibrating volatility surfaces
"""
from typing import Type, Dict, Literal
from tqdm import trange

import inspect
import numpy as np

from app.pricing.coxrossrubinstein import CRR
from app.modules.dividend_riskfree import Pairqr
from app.utils.convert import convert_to_numpy
from app.utils.types import ArrayLike


class VolCalibrator:
    """
    Implied Volatility calibrator
    """

    data: Dict[str, np.ndarray]

    def __init__(self, model: Type[CRR], method: str = "CRR", **kwargs) -> None:  # type: ignore
        self.method = method
        self.model = model(**kwargs)  # type: ignore
        self.data = {}
        self.eval = False
        self.parameters = None
        self.qr = None

    def bisection(
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        qr: Pairqr,
        market_prices: ArrayLike,
        n: int = 100,
        vol_low: float = 0.05,
        vol_high: float = 5,
        fidelity: float = 0.001,
        use_extrapolation: bool = False,  # Set to True to use extr
        get_vega: bool = False,
        **extr_kwargs,  # type: ignore
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
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

        if use_extrapolation:
            price_low = self.model.extr(
                S=S_np, K=K_np, T=T_np, qr=qr, vol=v_low, **extr_kwargs  # type: ignore
            )
            price_high = self.model.extr(
                S=S_np, K=K_np, T=T_np, qr=qr, vol=v_high, **extr_kwargs  # type: ignore
            )
        else:
            price_low = self.model(S=S_np, K=K_np, T=T_np, qr=qr, vol=v_low, n=n)  # type: ignore
            price_high = self.model(S=S_np, K=K_np, T=T_np, qr=qr, vol=v_high, n=n)  # type: ignore

        if not self.eval:
            assert not qr.r.discount_factors is None
            self.data["DF"] = qr.r.discount_factors[:, -1]
            self.data["F_T"] = qr.form_forward()
            self.qr=qr
            if qr.q.procent:
                self.data["F_T"] *= S_np

        # --- Error checking (optional but recommended) ---
        # Find options where the market price is outside the bracket
        too_low = P_market < price_low
        too_high = P_market > price_high
        failed_bracket = too_low | too_high
        if np.any(failed_bracket):
            print(
                f"Warning: {np.sum(failed_bracket)} options are outside the initial vol bracket."
            )
            # Clamp market prices to be within the bracket for stability
            P_market = np.clip(P_market, price_low + 1e-10, price_high - 1e-10)
        idx_valid = ~failed_bracket
        S_valid = S_np[idx_valid]
        K_valid = K_np[idx_valid]
        T_valid = T_np[idx_valid]
        P_market_valid = P_market[idx_valid]

        v_low = np.full(len(S_valid), vol_low)
        v_high = np.full(len(S_valid), vol_high)
        v_mid = (v_low + v_high) / 2.0

        def _price_batch(vol_guess: np.ndarray):
            if use_extrapolation:
                return self.model.extr(
                    S=S_valid,
                    K=K_valid,
                    T=T_valid,
                    qr=qr,
                    vol=vol_guess,
                    **extr_kwargs,  # type: ignore
                )
            return self.model(
                S=S_valid, K=K_valid, T=T_valid, qr=qr, vol=vol_guess, n=n
            )

        iterations = int(np.ceil(np.log((vol_high - vol_low) / fidelity) / np.log(2)))
        # --- Vectorized Bisection Loop ---
        for _ in trange(iterations, mininterval=0.5):

            # This is the key: one call prices ALL options
            price_mid = _price_batch(v_mid)

            # Calculate error for all options
            err = price_mid - P_market_valid

            # Vectorized update:
            # Where err > 0 (price is too high), our new v_high is v_mid
            # Where err <= 0 (price is too low), our new v_low is v_mid
            is_too_high = err > 0
            v_high = np.where(is_too_high, v_mid, v_high)
            v_low = np.where(is_too_high, v_low, v_mid)
            np.add(v_low, v_high, out=v_mid)
            v_mid *= 0.5
        # The result is the midpoint of the final bracket
        v_final = np.full_like(S_np, vol_high)

        v_final[too_low] = vol_low
        v_final[idx_valid] = v_mid

        if not self.eval:
            self.data["k"] = np.log(K_np) - np.log(
                S_np / self.data["DF"] - self.data["F_T"]
            )
            self.data["IV"] = v_final
            self.data["T"] = T_np
            self.data['K'] = K_np
            self.data["S"] = S_np
            self.data["market"] = P_market
            if get_vega:
                vega = np.full_like(S_np, np.nan)
                vega[idx_valid] = (_price_batch(v_high) - _price_batch(v_low)) / (
                    v_high - v_low
                )
                self.data["Vega"] = vega
                return v_final, vega

        if get_vega:
            vega = np.full_like(S_np, np.nan)
            vega[idx_valid] = (_price_batch(v_high) - _price_batch(v_low)) / (
                v_high - v_low
            )
            return v_final, vega
        # Return NaN for options that failed the initial bracketing
        # v_final[failed_bracket] = np.nan\
        return v_final

    def form_grid(
        self, smooth: Literal["gaussian1d", "gaussian2d", None] = None
    ) -> None:
        """
        Method to form the grid for interpolation
        """
        self.data["w"] = self.data["T"] * self.data["IV"] ** 2
        if smooth == "gaussian2d":
            bandwidth_strike = 0.01
            bandwidth_time = 0.1
            dk = (self.data["k"][None, :] - self.data["k"][:, None]) / bandwidth_strike
            dt = (self.data["T"][None, :] - self.data["T"][:, None]) / bandwidth_time
            weights = np.exp(-(dk**2) / 2) * np.exp(-(dt**2) / 2)
            weights /= np.sum(weights, axis=1)[:, None]
            self.data["w_sm"] = np.sum(weights * self.data["w"][None, :], axis=1)
        if smooth == "gaussian1d":
            bandwidth_strike = 0.01
            dk = (self.data["k"][None, :] - self.data["k"][:, None]) / bandwidth_strike
            dt = (self.data["T"][None, :] == self.data["T"][:, None])*1.0
            weights = np.exp(-(dk**2) / 2)*dt
            weights /= np.sum(weights, axis=1)[:, None]
            self.data["w_sm"] = np.sum(weights * self.data["w"][None, :], axis=1)

    def compute_vol(self,vol:ArrayLike, use_extrapolation:bool=True)->None:
        '''
        Computes volatilti from data
        '''
        self.eval=True
        keys = inspect.signature(self.model).parameters.keys()
        kw_args = {k: i for k, i in self.data.items() if k in keys}
        assert not self.qr is None
        if use_extrapolation:
            return self.model.extr(vol=vol, qr=self.qr, **kw_args) # type: ignore
        return self.model(vol=vol, qr=self.qr, **kw_args) # type: ignore