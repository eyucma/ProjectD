"""
This module contains Binomial pricing function using CoxRossRubinstein defined as class
"""

import numpy as np

from app.utils.types import ArrayLike
from app.modules.dividend_riskfree import Dividend, RiskFree
from app.utils.convert import convert_to_numpy

SMALL = 1e-6


class CRR:
    """
    A fully vectorized Cox-Ross-Rubinstein (CRR) binomial option pricing model.

    This class can price a batch of 'm' options simultaneously.
    It is designed to handle RiskFree and Dividend class inputs for risk free and dividend.

    Parameters
    ----------
    call : bool, True for call, False for put
    american : bool, True for American Option, False for European
    """

    def __init__(
        self,
        call: bool = True,
        american: bool = True,
    ) -> None:
        self.v = None
        self.american = american
        self.call = call

    def _payoff(
        self, s: np.ndarray, k: np.ndarray
    ) -> np.ndarray:  # pylint: disable=invalid-name
        """Vectorized payoff calculation."""
        if self.call:
            return np.maximum(s - k, 0)
        return np.maximum(k - s, 0)

    def build_tree(
        self,
        S: ArrayLike,  # pylint: disable=invalid-name
        K: ArrayLike,  # pylint: disable=invalid-name
        T: ArrayLike,  # pylint: disable=invalid-name
        r: RiskFree,
        q: Dividend,
        vol: ArrayLike,
        n: int,
    ) -> np.ndarray:
        """
        Builds the Binomial Tree and stores the at time option value in self.v
        """
        assert n > 0

        S = convert_to_numpy(S)
        K = convert_to_numpy(K)
        T = convert_to_numpy(T)

        vol = convert_to_numpy(vol)

        if q.n != n:
            q.fit(S, T, n)
        if r.n != n:
            r.fit(T, n)

        if q.forward:
            principal_div = (q.cash_paid * r.discount_factors).sum(axis=1)
            assert isinstance(principal_div, np.ndarray)
            S = S - np.clip(S - principal_div, SMALL, None)

        g = vol * np.sqrt(T / n)
        u = np.exp(g)
        d = np.exp(-g)
        inter_disc = r.intermediate_discount  # (len(T),n)
        p = (1 / inter_disc - d[:, None]) / (
            u[:, None] - d[:, None]
        )  # Assuming dividend is not factored into p
        v = np.zeros(
            (len(T), n + 1, n + 1)
        )  # stores the tree as (option_no,ith_time,jth_price)

        # terminal
        assert isinstance(q.factors, np.ndarray)
        v[:, -1, :] = self._payoff(
            S[:, None]
            * np.exp(g[:, None] * (2 * np.arange(n + 1)[None, :] - n))
            * q.factors[:, -1, None],
            K[:, None],
        )

        for i in reversed(range(n)):
            candidate = inter_disc[:, i, None] * (
                p[:, i, None] * v[:, i + 1, 1 : i + 2]
                + (1 - p[:, i, None]) * v[:, i + 1, : i + 1]
            )
            if self.american:
                v[:, i, : i + 1] = np.maximum(
                    self._payoff(
                        S[:, None]
                        * np.exp(g[:, None] * (2 * np.arange(i + 1)[None, :] - i))
                        * q.factors[:, i, None],
                        K[:, None],
                    ),
                    candidate,
                )
            else:
                v[:, i, : i + 1] = candidate
        self.v = v
        return self.v[:, 0, 0]

    def extrapolate(
        self,
        S: ArrayLike,  # pylint: disable=invalid-name
        K: ArrayLike,  # pylint: disable=invalid-name
        T: ArrayLike,  # pylint: disable=invalid-name
        r: RiskFree,
        q: Dividend,
        vol: ArrayLike,
        n0: int = 10,
        l: int = 2,
        s: int = 7,
    ):
        """
        Richardson-Romberg extrapolation
        """
        n_values = [n0*l**i for i in range(s)]
        num_options = len(convert_to_numpy(T))
        a = np.zeros((num_options,s,s))
        for i in range(s):
            a[:,i,0]=self.build_tree(S,K,T,r,q,vol,n=n_values[i])
        for k in range(1,s):
            factor = 1.0 / (l**k - 1.0)
            a[:,k:s, k] = a[:,k:s, k - 1] + (a[:,k:s, k - 1] - a[:,k - 1:s-1, k - 1]) * factor
        return a[:,s-1,s-1]