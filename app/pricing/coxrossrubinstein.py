"""
This module contains Binomial pricing function using CoxRossRubinstein defined as class
"""

import numpy as np

from typing import Literal

from app.utils.types import ArrayLike
from app.modules.dividend_riskfree import Pairqr
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

    def __call__(
        self,
        S: ArrayLike,  # pylint: disable=invalid-name
        K: ArrayLike,  # pylint: disable=invalid-name
        T: ArrayLike,  # pylint: disable=invalid-name
        qr: Pairqr,
        vol: ArrayLike,
        n: int,
        store_tree: bool = False,
    ) -> np.ndarray:
        """
        Builds the Binomial Tree and stores the at time option value in self.v
        """
        assert n > 0

        S = convert_to_numpy(S)  # type: ignore
        K = convert_to_numpy(K)  # type: ignore
        T = convert_to_numpy(T)  # type: ignore
        vol = convert_to_numpy(vol)

        if qr.n != n:
            qr.fit(S=S, T=T, n=n, vol=vol)

        if qr.q.forward:
            assert not qr.q.cash_paid is None
            assert not qr.r.discount_factors is None
            if qr.q.procent:
                qr.q.cash_paid = qr.q.cash_paid * S[:, None]
            principal_div = (qr.q.cash_paid * qr.r.discount_factors).sum(axis=1)
            assert isinstance(principal_div, np.ndarray)
            S = np.clip(S - principal_div, SMALL, None)  # type: ignore

        g = vol * np.sqrt(T / n)
        u = np.exp(g)
        d = np.exp(-g)
        inter_disc = qr.r.intermediate_discount  # (len(T),n)
        assert not inter_disc is None
        p = (1 / inter_disc - d[:, None]) / (
            u[:, None] - d[:, None]
        )  # Assuming dividend is not factored into p

        # cached
        exp_cache = [
            np.exp(g[:, None] * (2 * np.arange(i + 1)[None, :] - i))
            for i in range(n + 1)
        ]

        assert isinstance(qr.q.factors, np.ndarray)
        v_last = self._payoff(
            S[:, None] * exp_cache[-1] * qr.q.factors[:, -1, None], K[:, None]
        )

        if store_tree:
            self.v = np.zeros(
                (len(T), n + 1, n + 1)
            )  # stores the tree as (option_no,ith_time,jth_price)
            self.v[:, -1, :] = v_last

        # arange_cache = [np.exp(g[:, None]*(2*np.arange(i+1)[None,:]-i)) for i in range(n+1)]
        for i in reversed(range(n)):

            # continuation value: inter_disc[:, i, None] has shape (m,1)
            v_next = inter_disc[:, i, None] * (
                p[:, i, None] * v_last[:, 1 : i + 2]
                + (1 - p[:, i, None]) * v_last[:, : i + 1]
            )  # v_next has shape (m, i+1)
            if self.american:
                np.maximum(
                    self._payoff(
                        S[:, None]
                        * exp_cache[i]
                        * qr.q.factors[:, i, None],
                        K[:, None],
                    ),
                    v_next,
                    out=v_next,
                )
            if store_tree:
                self.v[:, i, : i + 1] = v_next  # type: ignore
                # shape (m, i+1)
            v_last = v_next
        return v_last[:,0]

    def extr(
        self,
        S: ArrayLike,  # pylint: disable=invalid-name
        K: ArrayLike,  # pylint: disable=invalid-name
        T: ArrayLike,  # pylint: disable=invalid-name
        qr: Pairqr,
        vol: ArrayLike,
        n0: int = 10,
        l: int = 2,
        s: int = 6,
        tol_rel: float = 0.1,
    ):
        """
        Richardson-Romberg extrapolation
        """
        n_values = [n0 * l**i for i in range(s)]
        num_opts = len(convert_to_numpy(T))
        a = np.zeros((num_opts, s, s))
        for i in range(s):
            a[:, i, 0] = self(S=S, K=K, T=T, qr=qr, vol=vol, n=n_values[i])
        for k in range(1, s):
            factor = 1.0 / (l**k - 1.0)
            a[:, k:s, k] = (
                a[:, k:s, k - 1]
                + (a[:, k:s, k - 1] - a[:, k - 1 : s - 1, k - 1]) * factor
            )
        # check deviation:
        fn_max = a[:, -1, 0]
        extrplt = np.clip(a[:, s - 1, s - 1], 0, None)
        rel_diff = np.abs(extrplt - fn_max) / np.maximum(np.abs(fn_max), 1e-20)
        invalid_mask = (extrplt < 0) | (rel_diff > tol_rel) | ~np.isfinite(extrplt)
        extrplt[invalid_mask] = fn_max[invalid_mask]
        return extrplt

    def set_method(self, method: Literal["call", "put"]) -> None:
        if method == "call":
            self.call = True
        else:
            self.call = False

    def set_type(self, options_type: Literal["American", "European"]) -> None:
        if options_type == "American":
            self.american = True
        else:
            self.american = False

    def rr_extr(  # type: ignore
        self,
        S: ArrayLike,
        K: ArrayLike,
        T: ArrayLike,
        qr: Pairqr,
        vol: ArrayLike,
        n0: int = 10,
        l: int = 2,
        s: int = 6,
        tol_rel: float = 0.1,  # max relative diff allowed between extrapolated and f_nmax
    ):
        """
        Robust vectorized Richardson–Romberg extrapolation with convergence-order estimation.
        Falls back to f_nmax when extrapolated values are negative or deviate too much.

        Parameters
        ----------
        n0 : int
            Base number of tree steps.
        l : int
            Step multiplier (2 = doubling sequence).
        s : int
            Number of levels (>=3 recommended).
        tol_rel : float
            Relative tolerance for accepting extrapolated values.
        """
        # Vectorize inputs

        num_opts = len(convert_to_numpy(T))
        n_values = np.array([n0 * l**i for i in range(s)], dtype=int)

        # Step 1: compute prices at all tree depths
        a = np.zeros((num_opts, s))
        for i in range(s):
            a[:, i] = self(S=S, K=K, T=T, qr=qr, vol=vol, n=n_values[i])

        # Step 2: estimate convergence order p for each option
        # Use last 3 points to reduce noise
        Pn, P2n, P4n = a[:, -3], a[:, -2], a[:, -1]
        d1 = Pn - P2n
        d2 = P2n - P4n
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((d1 * d2 > 0) & (np.abs(d2) > 1e-14), d1 / d2, np.nan)
            p_est = np.log(np.abs(ratio)) / np.log(l)
        # Clamp unrealistic values
        p_est = np.clip(np.nan_to_num(p_est, nan=1.0, posinf=1.0, neginf=1.0), 0.5, 4.0)

        # Step 3: extrapolate assuming asymptotic error ∼ C * n^{-p}
        two_p = l**p_est
        extrapolated = (two_p * P2n - Pn) / (two_p - 1.0)

        # Step 4: stability checks
        fn_max = a[:, -1]  # highest n result
        rel_diff = np.abs(extrapolated - fn_max) / np.maximum(np.abs(fn_max), 1e-12)

        # Criteria for fallback
        invalid_mask = (
            (extrapolated < 0) | (rel_diff > tol_rel) | ~np.isfinite(extrapolated)
        )

        # Replace bad extrapolations with last finite value
        extrapolated[invalid_mask] = fn_max[invalid_mask]

        return extrapolated, {
            "n_values": n_values,
            "raw_prices": a,
            "p_est": p_est,
            "fallback_mask": invalid_mask,
            "rel_diff": rel_diff,
        }  # type: ignore
