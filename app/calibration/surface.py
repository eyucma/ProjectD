# pylint: disable=R0913, R0914, R0917
"""
Contains class for imp vol surface
"""

from typing import Dict, List, Tuple

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

    def __init__(self, data: pd.DataFrame, qr: Pairqr) -> None:
        SurfaceSchema.validate(data, lazy=True)
        self.data = data
        self.qr = qr

    def _svi_obj(
        self,
        x: npt.NDArray[np.float64],
        k: npt.NDArray[np.float64],
        w: npt.NDArray[np.float64],
        vega_w: npt.NDArray[np.float64],
        vol_w: npt.NDArray[np.float64],
        c: float | np.float64,
        p: float | np.float64 = 1.1,
    ) -> float | np.float64:
        a, b, p, m, s = x
        w_svi = SVI(k, a, b, p, m, s)
        res = (
            np.abs(w - w_svi) ** p  # Squared IV residual
            * vega_w  # log(1 + abs(Vega)) weight
            * vol_w  # log(1 + Volume) weight
        )
        return float(np.mean(res) * c)

    def fit_svi(
        self,
        data: None | pd.DataFrame = None,
        p: float = 1.1,
        l_bound: float = 0.001,
        copy: bool = True,
    ) -> tuple[Dict[float, float], npt.NDArray[np.float64]]:
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

        for t, dat in options.groupby("T"):  # type: ignore
            vega = dat["Vega"].to_numpy(copy=copy, dtype=np.float64)  # type: ignore
            volume = dat["Volume"].to_numpy(copy=copy, dtype=np.float64)  # type: ignore
            vega[np.isnan(vega)] = 0.0001
            vega = np.abs(vega)
            volume[np.isnan(volume)] = 1

            w = data["IV"].to_numpy(dtype=np.float64) ** 2 * t  # type: ignore
            k = data["k"].to_numpy(dtype=np.float64)  # type: ignore
            vega_w = np.log(1 + vega)
            vol_w = np.log(1 + volume)
            c = 1e8 / np.sqrt(t) / np.mean(vega_w * vol_w)  # type: ignore

            args_tuple = (k, w, vega_w, vol_w, c, p)  # type: ignore
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
            res = np.abs(SVI(k, *theta) - w)  # type: ignore
            med = np.median(res)
            score = (res - med) / np.maximum(med, l_bound)
            scores = np.concatenate((scores, score))

        return optionary, scores  # type: ignore

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

        for _ in range(max_it):
            _, scores = self.fit_svi(data=options, copy=False)
            if len(scores) != len(options):
                raise ValueError(
                    "Score array length does not match the DataFrame length."
                    +" Check if your scores were correctly concatenated."
                )
            options["Score"] = scores
            options = options[options["Score"] < tol]

        if in_place:
            self.data = options.copy()
        return options
