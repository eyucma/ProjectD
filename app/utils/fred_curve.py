# pylint: disable=W0718
"""
Module for handling FRED treasury yield curve and interpolation
"""


from pathlib import Path
from typing import Dict, List, cast
from fredapi import Fred  # type: ignore
from scipy.interpolate import PchipInterpolator
from pandas.tseries.offsets import BDay

from pandera.typing import Series

import pandas as pd
import numpy as np
import numpy.typing as npt

from app.utils.pathfinder import find_project_root
from app.utils.types import RFSchema

KEY = "5e3ecc635dff0f679f018d24310688a4"


class FredCurve:
    """
    Module for Fred treasury yield curves
        ----------
        date : pd.Timestamp
            Target date for risk-free curve.
        api_key : str
            FRED API key.
    """

    def __init__(
        self, date: pd.Timestamp | str, root: Path | None = None, key: str = KEY
    ) -> None:
        self.key = key
        if root is None:
            root = find_project_root(anchor="data")
        dt = pd.Timestamp(date)
        self.date = dt
        year = f"{dt.year:04d}"
        month = f"{dt.month:02d}"
        day = f"{dt.day:02d}"
        date_dir = root / year / month / day
        date_dir.mkdir(parents=True, exist_ok=True)
        curve_path = date_dir / "DGS.feather"
        if curve_path.exists():
            self.data = pd.read_feather(curve_path)
        else:
            self.data = self.get_data(date=self.date)
            self.data.to_feather(curve_path)
        RFSchema.validate(self.data, lazy=True)
        self.interpolator = PchipInterpolator(
            self.data["Maturity"], np.log(1 + self.data["Yield"])
        )

    def get_data(self, date: pd.Timestamp | str, key: str = KEY) -> pd.DataFrame:
        """
        Fetches treasury yields
        Parameters
        ----------
        date : pd.Timestamp
            Target date for risk-free curve.
        api_key : str
            FRED API key.

        Returns
        -------
        f : callable
            Function f(t) giving annualized risk-free rate (decimal) for maturity t (years).
        last_available_date : pd.Timestamp
            The actual date the yields were fetched for.
        """
        fred = Fred(api_key=key)
        date = pd.Timestamp(date)

        # Rollback to last business day (if weekend/holiday)
        target_date = BDay().rollback(date)

        # FRED series IDs for standard maturities
        series_map = {
            "1M": ("DGS1MO", 1 / 12),
            "3M": ("DGS3MO", 3 / 12),
            "6M": ("DGS6MO", 6 / 12),
            "1Y": ("DGS1", 1.0),
            "2Y": ("DGS2", 2.0),
            "3Y": ("DGS3", 5.0),
            "5Y": ("DGS5", 5.0),
            "7Y": ("DGS7", 5.0),
            "10Y": ("DGS10", 10.0),
            "20Y": ("DGS20", 20.0),
            "30Y": ("DGS30", 30.0),
        }

        data_rows: List[Dict[str, float | np.float64]] = []

        for label, (sid, mat) in series_map.items():
            try:
                # Fetch small window
                s = fred.get_series(  # type: ignore
                    sid, target_date - pd.Timedelta(days=7), target_date
                )
                if len(s) == 0 or s.empty:
                    print(f"No data available for {sid} ({label})")
                    continue

                # Extract last available yield
                last_yield = s.iloc[-1] / 100  # type: ignore
                data_rows.append({"Maturity": mat, "Yield": last_yield})

            except (ValueError, IndexError) as e:
                # Catch specific data/indexing errors
                print(f"Data Error fetching {sid}: {e}")
            except Exception as e:  # type: ignore
                # Catch network/unexpected errors but log them clearly
                print(f"Unexpected Error fetching {sid}: {e}")

        df = pd.DataFrame(data_rows)
        return df

    def fit_pchip(self, data: pd.DataFrame | None = None) -> None:
        """
        Fits pchip interpolator to data
        """
        if data is None:
            data = self.data
        RFSchema.validate(data, lazy=True)
        self.interpolator = PchipInterpolator(
            data["Maturity"], np.log(1 + data["Yield"])
        )

    def __call__(
        self, times: float | npt.NDArray[np.float64] | Series[float | int]
    ) -> npt.NDArray[np.float64]:
        """
        Returns interpolated time
        """
        return cast(npt.NDArray[np.float64], self.interpolator(times))
