"""
This module contains class definitions for Dividend and Risk Free
"""

from typing import Callable

import numpy as np

from app.utils.types import ArrayLike, Numeric, ArrayLike2
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
        procent: boolean, True if inpute rate if procent instead of amount cash paid
        forward: boolean, True if the forward/hybrid method is to be used.

    Submodules:
        fit: fits the Dividend data into a form compatability with binomial models
        special_fit: fit but allows conversion of cash payments into percentage yield
    """

    def __init__(
        self, data: ArrayLike2 | Numeric, procent: bool = False, forward: bool = False
    ) -> None:
        self.is_scalar = False
        if isinstance(data, (float, int)):
            self.is_scalar = True
        self.data = np.asarray(data)
        self.procent = procent
        if procent:
            if self.is_scalar:
                assert self.data >= 0 and self.data <= 1
            else:
                assert np.all(self.data[:, 1] <= 1) and np.all(self.data[:, 1] >= 0)
        self.factors = None
        self.cash_paid = None
        self.n = 0
        self.forward = forward

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
        n: int = 100,
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
        self.factors = np.ones((len(T), n + 1))

        T = convert_to_numpy(T)  # type: ignore

        assert len(T.shape) < 2
        assert (self.forward) or self.procent or self.is_scalar

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
            self.cash_paid = m
            # beware this carries different interpretation depending on if percent or cash value
            if self.forward:
                self.factors = np.ones((len(T), n + 1))
            else:
                self.factors = shifted_cumprod(1 - m, axis=1)
            self.n = n

    def special_fit(
        self,
        T: ArrayLike,  # pylint: disable=invalid-name
        S: ArrayLike,  # pylint: disable=invalid-name
        vol: ArrayLike,  # pylint: disable=invalid-name
        rs: ArrayLike,
        n: int = 100,
    ):
        """
        Same as above but makes adjusts flat dividend amounts to get a percentage
        yield based on q_t=Div/S_0*exp((vol^2-r)t) based on the expected value of
        1/S_t.
        """
        self.factors = np.ones((len(T), n + 1))
        t = convert_to_numpy(T)
        s = convert_to_numpy(S)
        sig = convert_to_numpy(vol)

        m = np.zeros((len(t), n + 1))
        times = np.linspace(0, t, n + 1).T
        for time, cash in self.data:
            valid_maturities_idx = np.where(t >= time)[0]
            if len(valid_maturities_idx) == 0:
                continue
            index_rate = time * n / t[valid_maturities_idx]
            step_indices = np.floor(index_rate).astype(int)
            np.add.at(m, (valid_maturities_idx, step_indices), cash)
        s_adj = s[:, None] * np.exp((rs - sig[:, None] ** 2) * times)
        self.cash_paid = m
        m = self._convert(m, s=s_adj)
        self.factors = shifted_cumprod(1 - m, axis=1)
        self.n = n


class RiskFree:
    """
    Class for handling RiskFree
    Parameters:
    Has to supply either f which maps the risk free curve or
    data as a numeric with representation a common flat rate.
        data: has to be a numeric (float,int)
        f: function that takes np.ndarray time and returns risk free rate up to t
        dynamic_time: boolean, True if each node in the binomial model requires local risk free rate
        false if a flat rate is applied across the entire model.
    Submodules:
        fit: fits the risk free data into a form compatability with binomial models
    """

    def __init__(
        self,
        data: Numeric | None = None,
        f: Callable[[np.ndarray], np.ndarray] | None = None,
        dynamic_time: bool = False,
    ) -> None:
        self.is_scalar = False
        if isinstance(data, (float, int)):
            self.is_scalar = True
            self.data = np.asarray(data)
        else:
            assert not f is None
        # self.data = np.asarray(data)
        self.discount_factors = None  # placeholder
        self.rs = None  # stores continous rs
        self.intermediate_discount = None  # placeholder
        self.f = f
        self.n = 0
        self.dyn_time = dynamic_time

    def fit(self, T: ArrayLike, n: int = 100) -> None:  # pylint: disable=invalid-name
        """
        fits the Dividend data into a form compatability with binomial models
        Parameters:
        T: time to maturity, must be 1d arraylike
        n: integer to discretize it
        """
        t = convert_to_numpy(T)
        assert len(t.shape) < 2
        times = np.linspace(0, t, n + 1).T
        if self.is_scalar:
            self.rs = np.ones_like(times) * self.data
        else:
            assert not self.f is None
            if self.dyn_time:
                self.rs = self.f(times)  # shape (len(T),n+1)
            else:
                self.rs = np.ones_like(times) * self.f(t)[:, None]
        self.intermediate_discount = np.exp(
            (self.rs * times)[:, :-1] - (self.rs * times)[:, 1:]
        )  # BEWARE shape (len(T),n)
        self.discount_factors = np.exp(-self.rs * times)
        self.n = n


class Pairqr:
    """
    Class for handling dividend, risk free pair
    Parameters:
        data: has to be a nx2 np.array or list of list in that shape each pair is of form
          (time, dividend paid) or (time, dividend yield) notably time is relative,
          with 0 being the time of considerations.
           Alternatively, a scalar which is cast in a form compatible
        procent: boolean if inpute rate if procent instead of

    Submodules:
        fit: fits the Dividend data into a form compatability with binomial models
    """

    def __init__(
        self,
        data_q: ArrayLike2 | Numeric,
        data_r: Numeric | None = None,
        procent: bool = False,
        forward: bool = False,
        f: Callable[..., np.ndarray] | None = None,
    ) -> None:
        self.q = Dividend(data=data_q, procent=procent, forward=forward)
        self.r = RiskFree(data=data_r, f=f)
        self.n = 0

    def fit(
        self,
        T: ArrayLike,  # pylint: disable=invalid-name
        S: ArrayLike | None = None,  # pylint: disable=invalid-name
        vol: ArrayLike | None = None,
        n: int = 100,
    ) -> None:
        '''
        Auxilary fit to deal with the different cases
        '''
        self.r.fit(T=T, n=n)
        if self.q.forward or self.q.procent:
            self.q.fit(T=T, n=n)
        else:
            assert not vol is None
            assert not S is None
            assert isinstance(self.r.rs, np.ndarray)
            self.q.special_fit(T=T, n=n, S=S, rs=self.r.rs, vol=vol)

    def form_forward(self) -> np.ndarray:
        '''
        Computes forward value under this pair
        '''
        assert self.q.forward
        assert not self.q.cash_paid is None
        assert not self.r.discount_factors is None
        return (self.q.cash_paid * self.r.discount_factors).sum(
            axis=1
        ) / self.r.discount_factors[:, -1]
