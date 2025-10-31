"""
This module contains class definitions for Dividend and Risk Free
"""

from typing import Callable

import numpy as np

from app.utils.types import ArrayLike, Numeric
from app.utils.convert import convert_to_numpy


def shifted_cumsum(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    cumsum but shifted one to be exclusive
    """
    assert len(x.shape) > 0
    return np.insert(np.cumsum(x, axis=axis)[..., :-1], 0, 0, axis=axis)


def shifted_cumprod(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    cumprod but shifted one to be exclusive
    """
    assert len(x.shape) > 0
    return np.insert(np.cumprod(x, axis=axis)[..., :-1], 0, 1, axis=axis)


class Dividend:
    """
    Class for handling dividend data
    Parameters:
        data: has to be a nx2 np.array or list of list in that shape each pair is of form
          (time, dividend paid) or (time, dividend yield) notably time is relative,
          with 0 being the time of considerations.
           Alternatively, a scalar which is cast in a form compatible
        procent: boolean if inpute rate if procent instead of

    Submodules:
        fit: fits the Dividend data into a form compatability with binomial models
    """

    def __init__(self, data: ArrayLike | Numeric, procent: bool = False) -> None:
        self.is_scalar = False
        if isinstance(data, (float, int)):
            self.is_scalar = True
        self.data = convert_to_numpy(data)
        self.procent = procent
        if procent:
            if self.is_scalar:
                assert self.data >= 0 and self.data <= 1
            else:
                assert np.all(self.data[:, 1] <= 1) and np.all(self.data[:, 1] >= 0)
        self.factors = None
        self.cash_paid = None
        self.n = 0
        self.forward = False

    def _convert(
        self, x: np.ndarray, s: np.ndarray, to_rate: bool = True
    ) -> np.ndarray:
        """
        converts nominal payments to rates
        Parameters:
        s: np.array, base underlying price to convert against
        to_rate: boolean, True if convert to rate, False if to nominal
        """
        if to_rate:
            return x / s
        return s * x

    def fit(
        self,
        T: ArrayLike,  # pylint: disable=invalid-name
        S: ArrayLike,  # pylint: disable=invalid-name
        n: int = 100,
        forward: bool = False,
    ) -> None:
        """
        fits the Dividend data into a form compatability with binomial models,
        produces a dividend matrix self.factors which include cumulative drops
        in underlying

        Parameters:
        T: time to maturity, must be 1d ArrayLike
        S: initial underlying, must be 1d ArrayLike, only necessary if all
            the following are true:
            - dividends are in cash value and not procent
            - forward is False
        n: integer to discretize it
        forward: boolean, True if this class is used for forward adjustments
        Note this assumes that (t_i,D_i) t_i is the x-day
        """
        self.forward = forward
        self.factors = np.ones((len(T), n + 1))

        T = convert_to_numpy(T)
        S = convert_to_numpy(S)
        assert len(T.shape) < 2

        if self.is_scalar:
            div_yields = np.ones((len(T), n + 1)) * self.data * T[:, None] / n
            self.factors = np.exp(-shifted_cumsum(div_yields, axis=1))
        else:
            m = np.zeros((len(T), n + 1))
            for time, cash in self.data:
                valid_maturities_idx = np.where(T >= time)[0]
                if len(valid_maturities_idx) == 0:
                    continue
                index_rate = time * n / T[valid_maturities_idx]
                step_indices = np.floor(index_rate).astype(int)
                np.add.at(m, (valid_maturities_idx, step_indices), cash)
            if forward:
                assert not self.procent
                self.cash_paid = m
            else:
                if not self.procent:
                    m = self._convert(m, S[:, None])
                self.factors = shifted_cumprod(1 - m, axis=1)
            self.n = n


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

    def __init__(self, data: ArrayLike | Numeric, f: Callable | None = None) -> None:
        self.is_scalar = False
        if isinstance(data, (float, int)):
            self.is_scalar = True
        self.data = convert_to_numpy(data)
        self.discount_factors = np.zeros_like(data)  # placeholder
        self.rs = np.zeros_like(data)  # placeholder
        self.intermediate_discount = np.zeros_like(data)  # placeholder
        self.f = f
        self.n = 0

    def fit(self, T: ArrayLike, n: int = 100) -> None:  # pylint: disable=invalid-name
        """
        fits the Dividend data into a form compatability with binomial models
        Parameters:
        T: time to maturity, must be 1d arraylike
        n: integer to discretize it
        """
        T = convert_to_numpy(T)
        times = np.linspace(0, T, n + 1).T
        if self.is_scalar:
            self.rs = self.rs + self.data
        else:
            times = np.linspace(0, T, n + 1).T  # shape (len(T),n+1)
            assert len(T.shape) < 2
            assert not self.f is None
            self.rs = self.f(times)  # shape (len(T),n+1)
        self.intermediate_discount = np.exp(
            (self.rs * times)[:, :-1] - (self.rs * times)[:, 1:]
        )  # BEWARE shape (len(T),n)
        self.discount_factors = np.exp(-self.rs * times)
        self.n = n
