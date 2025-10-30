"""
This module contains typing
"""

from typing import Union, List, Callable

import numpy as np
import pandas as pd

from app.utils.convert import convert_to_numpy


Numeric = Union[float, int]

ArrayLike = List[Numeric] | np.ndarray | pd.Series | pd.DataFrame

ARRAY_LIKE_RUNTIME_TYPES = (list, np.ndarray, pd.Series, pd.DataFrame)
NUMERIC_RUNTIME_TYPES = (float, int)


class Dividend:
    """
    Class for handling dividend data
    Parameters:
        price: observed option price
        data: has to be a nx2 np.array or list of list in that shape each pair is of form
          (time, dividend paid) notably time is relative, with 0 being the time of considerations

    Submodules:
        fit: fits the Dividend data into a form compatability with binomial models
    """

    def __init__(self, data: ArrayLike) -> None:
        self.data = convert_to_numpy(data)
        self.fitted = False
        self.div = convert_to_numpy(data)

    def fit(self, T: ArrayLike, n: int = 100) -> None:  # pylint: disable=invalid-name
        """
        fits the Dividend data into a form compatability with binomial models
        Parameters:
        T: time to maturity, must be 1d arraylike
        n: integer to discretize it
        """
        T = convert_to_numpy(T)
        assert len(T.shape) < 2
        self.div = np.zeros_like((len(T), n + 1))

        for time, cash in self.data:
            valid_maturities_idx = np.where(T >= time)[0]
            if len(valid_maturities_idx) == 0:
                continue
            index_rate = time * n / T[valid_maturities_idx]
            step_indices = np.floor(index_rate).astype(int)
            np.add.at(self.div, (valid_maturities_idx, step_indices), cash)

        self.fitted = True


class RiskFree:
    """
    Class for handling RiskFree
    Parameters:
        price: observed option price
        data: has to be a nx2 np.array or list of list in that shape each pair is of form
          (time, dividend paid) notably time is relative, with 0 being the time of considerations

    Submodules:
        fit: fits the Dividend data into a form compatability with binomial models
    """

    def __init__(self, data: np.ndarray, f: Callable) -> None:
        self.data = data
        self.fitted = False
        self.discount_factors = np.zeros_like(data) #placeholder
        self.rs = np.zeros_like(data)   #placeholder
        self.intermediate_discount = np.zeros_like(data) #placeholder
        self.f = f

    def fit(self, T: ArrayLike, n: int = 100) -> None:  # pylint: disable=invalid-name
        """
        fits the Dividend data into a form compatability with binomial models
        Parameters:
        T: time to maturity, must be 1d arraylike
        n: integer to discretize it
        """
        T = convert_to_numpy(T)
        times = np.linspace(0, T, n + 1).T  # shape (len(T),n+1)
        assert len(T.shape) < 2
        self.rs = self.f(times)  # shape (len(T),n+1)
        self.discount_factors = np.exp(-self.rs*times)
        self.intermediate_discount = self.discount_factors[:,1:]/self.discount_factors[:,:-1]
        self.fitted = True
