# pylint: disable=R0913, R0914, R0917
"""
Contains class for imp vol surface
"""

from typing import Dict, List, Tuple, cast

import pandas as pd
import numpy as np
import numpy.typing as npt

from scipy.optimize import minimize  # type: ignore

from app.modules.dividend_riskfree import Pairqr
from app.curves.svi import SVI
from app.utils.types import SurfaceSchema


required_columns = {"T", "k", "Vega", "Volume", "IV"}

bdds: List[Tuple[float | None, float | None]] = [
    (None, None),  # a
    (0, None),  # b
    (-1, 1),  # p
    (None, None),  # m
    (0, None),  # s
]


class Surface:
    """
    Class for handling imp vol surfaces
    """

    data: pd.DataFrame
    qr: Pairqr
    params: Dict[float, npt.NDArray[np.float64]] | None

    def __init__(
        self,
        data: pd.DataFrame,
        qr: Pairqr,
        params: Dict[float, npt.NDArray[np.float64]] | None = None,
    ) -> None:
        SurfaceSchema.validate(data, lazy=True)
        self.data = data
        self.qr = qr
        self.params = params

    def _svi_obj(
        self,
        x: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
        vega_w: npt.NDArray[np.float64],
        vol_w: npt.NDArray[np.float64],
        c: float | np.float64,
        p_norm: float | np.float64 = 1.1,
    ) -> float | np.float64:
        a, b, p, m, s = x
        w_svi = SVI(k, a, b, p, m, s)
        res = (
            np.abs(w - w_svi) ** p_norm  # Squared IV residual
            * vega_w  # log(1 + abs(Vega)) weight
            * vol_w  # log(1 + Volume) weight
        )
        return float(np.mean(res) * c)

    def fit_svi(
        self,
        data: None | pd.DataFrame = None,
        p_norm: float = 1.1,
        l_bound: float = 0.001,
        copy: bool = True,
        store_params: bool = True,
    ) -> tuple[Dict[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Fits SVI curve to data, returning a dictionary with key equal to float times,
        and items the array with the 5 params as well as a score
        """
        if data is None:
            options = self.data
        else:
            SurfaceSchema.validate(data, lazy=True)
            options = data

        if not required_columns.issubset(options.columns):
            missing_cols = required_columns - set(options.columns)
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing_cols}"
            )
        optionary = {}

        scores = np.array([])

        theta = np.array([0.1, 0.1, -0.5, 0.5, 1.0])

        for t, smile in options.groupby("T"):  # type: ignore
            vega = smile["Vega"].to_numpy(copy=copy, dtype=np.float64)  # type: ignore
            volume = smile["Volume"].to_numpy(copy=copy, dtype=np.float64)  # type: ignore
            vega[np.isnan(vega)] = 0.0001
            vega = np.abs(vega)
            volume[np.isnan(volume)] = 1

            w = smile["IV"].to_numpy(dtype=np.float64) ** 2 * t  # type: ignore
            k = smile["k"].to_numpy(dtype=np.float64)  # type: ignore
            vega_w = np.log(1 + vega)
            vol_w = np.log(1 + volume)
            c = 1e8 / np.sqrt(t) / np.mean(vega_w * vol_w)  # type: ignore

            args_tuple = (k, w, vega_w, vol_w, c, p_norm)  # type: ignore
            res = minimize(  # type: ignore
                fun=self._svi_obj,  # your loss function
                x0=theta,  # starting guess for parameters
                args=args_tuple,
                method="L-BFGS-B",
                bounds=bdds,  # list of (low, high) for each parameter
                options={"maxiter": 500, "ftol": 1e-12},
            )
            theta = res.x  # type: ignore
            optionary[t] = theta
            residual = np.abs(SVI(k, *theta) - w)  # type: ignore
            med = np.median(residual)
            score = (residual - med) / np.maximum(med, l_bound)
            scores = np.concatenate((scores, score))
        if store_params:
            self.params = cast(Dict[float, npt.NDArray[np.float64]], optionary)
        return cast(Dict[float, npt.NDArray[np.float64]], optionary), scores  # type: ignore

    def prune_svi(
        self,
        data: None | pd.DataFrame = None,
        tol: float = 6.0,
        max_it: int = 10,
        in_place: bool = True,
    ) -> pd.DataFrame:
        """
        Algorithm to prune outliers using the SVI fit and score
        """
        if data is None:
            options = self.data.copy()
        else:
            SurfaceSchema.validate(data, lazy=True)
            options = data.copy()
        optionary: Dict[float, npt.NDArray[np.float64]] | None = None
        for _ in range(max_it):
            optionary, scores = self.fit_svi(data=options, copy=False)
            if len(scores) != len(options):
                raise ValueError(
                    "Score array length does not match the DataFrame length."
                    + " Check if your scores were correctly concatenated."
                )
            options["Score"] = scores
            options = options[options["Score"] < tol]

        if in_place:
            if optionary is None:
                raise RuntimeError("fit_svi was not executed because max_it=0")
            self.data = options.copy()
            self.params = optionary
        return options

    def llr(self, data: None | pd.DataFrame = None) -> pd.DataFrame:
        """
        Performs Local Linear Regression (LLR)
        """
        if self.params is None:
            raise RuntimeError("Surface is missing parameters")

        if data is None:
            options = self.data.copy()
        else:
            SurfaceSchema.validate(data, lazy=True)
            options = data.copy()

        ds = np.array([])

        for t in self.params.keys():
            data = options[(options["T"] == t)]
            if len(data) < 3:
                # Not enough points to smooth, just append raw/zeros
                ds = np.concatenate((ds, np.zeros(len(data))))
                continue

            ks = cast(npt.NDArray[np.float64], data["k"].to_numpy())  # type: ignore
            ws = cast(npt.NDArray[np.float64], data["w"].to_numpy())  # type: ignore

            # SVI Predict
            predict = SVI(ks, *self.params[t])
            residual = ws - predict  # The residual we want to smooth

            # --- FIX 1: Robust Bandwidth ---
            # Don't use min(). Use median spacing.
            # Multiplier 1.5 - 2.0 is usually a good starting point for options data.
            h = (
                8
                * np.median(np.abs(np.diff(ks)))
                # * np.maximum(8 * (np.abs(ks) - 0.1), 1)[:, None]
            )

            # Fallback if h is 0 (e.g. all points identical)
            if h == 0:
                h = 0.04
            # h[h == 0] = 0.05

            # --- FIX 2: Local Linear Regression (Vectorized) ---
            # We solve for a weighted least squares line at every point x_target.
            # This fixes the boundary bias.

            # 1. Compute Weights Matrix (N x N)
            # diffs[i, j] = ks[j] - ks[i]
            diffs = ks[None, :] - ks[:, None]
            weights = np.exp(-0.5 * (diffs / h) ** 2)

            # 2. Compute Weighted Moments
            # sum_w: sum of weights for each point
            sum_w = weights.sum(axis=1)
            # sum_wx: sum of weights * (neighbor_k - current_k)
            sum_wx = (weights * diffs).sum(axis=1)
            # sum_wxx: sum of weights * (neighbor_k - current_k)^2
            sum_wxx = (weights * diffs**2).sum(axis=1)

            # 3. Compute Weighted Y moments
            # For residuals (smoothing d)
            sum_wy_res = (weights * residual[None, :]).sum(axis=1)
            sum_wxy_res = (weights * diffs * residual[None, :]).sum(axis=1)

            # For raw variance (smoothing v)
            sum_wy_raw = (weights * ws[None, :]).sum(axis=1)
            sum_wxy_raw = (weights * diffs * ws[None, :]).sum(axis=1)

            # 4. Solve the linear system for the intercept (the smoothed value)
            # Determinant of the X^T W X matrix
            det = sum_w * sum_wxx - sum_wx**2

            # Handle instability if det is too small (fallback to weighted average)
            mask = det < 1e-12
            det[mask] = 1.0  # Avoid div by zero

            # Smoothed Residuals (a)
            # Formula: (sum_wxx * sum_wy - sum_wx * sum_wxy) / det
            smooth_res = (sum_wxx * sum_wy_res - sum_wx * sum_wxy_res) / det
            # Fallback to standard weighted average where det is unstable
            smooth_res[mask] = sum_wy_res[mask] / sum_w[mask]

            # Smoothed Raw Variance (b)
            smooth_raw = (sum_wxx * sum_wy_raw - sum_wx * sum_wxy_raw) / det
            smooth_raw[mask] = sum_wy_raw[mask] / sum_w[mask]

            # Final values
            a = smooth_res + predict

            ds = np.concatenate((ds, a))

        options["Smoothed"] = ds
        return options
