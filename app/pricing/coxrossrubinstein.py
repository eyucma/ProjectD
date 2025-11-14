"""
This module contains Binomial pricing function using CoxRossRubinstein defined as class
"""

import numpy as np

from typing import Literal
from numba import njit, prange # type: ignore

from app.utils.types import ArrayLike
from app.modules.dividend_riskfree import Pairqr
from app.utils.convert import convert_to_numpy

SMALL = 1e-6

@njit(parallel=True) # type: ignore
def loop(p: np.ndarray, inter: np.ndarray, pay: np.ndarray, n: int, american: bool = True) -> np.ndarray:
    m = pay.shape[0]  # number of options
    v_last = pay[:, -1, :].copy()  # shape (m, n+1)
    
    for i in range(n-1, -1, -1):
        v_next = np.zeros((m, i+1))
        
        # vectorized over options
        for j in prange(i+1): # type: ignore
            v_next[:, j] = inter[:, i] * (p[:, i] * v_last[:, j+1] + (1 - p[:, i]) * v_last[:, j])
        
        if american:
            # ensure shapes match exactly
            v_next = np.maximum(v_next, pay[:, i, :i+1])
        
        v_last = v_next  # next step
    
    return v_last[:, 0]


@njit(parallel=True) # type: ignore
def loop_error(
    p: np.ndarray,
    inter: np.ndarray,
    pay: np.ndarray,
    n: int,
    #    store_tree: bool = False,
    american: bool = True,
) -> np.ndarray:
    # if store_tree:
    #    v = np.zeros(
    #        (len(T), n + 1, n + 1)
    #    )  # stores the tree as (option_no,ith_time,jth_price)
    #    v[:, -1, :] = v_last
    v_last = pay[:, -1, :]
    for i in range(n - 1, -1, -1):
        # continuation value: inter_disc[:, i, None] has shape (m,1)
        v_next = inter[:, i, None] * (
            p[:, i,None] * v_last[:, 1 : i + 2] + (1 - p[:, i, None]) * v_last[: ,0: i + 1]
        )  # v_next has shape (m, i+1)
        if american:
            v_next = np.maximum( pay[:, i, 0:i+1], v_next)
        # if store_tree:
        #    v[:, i, : i + 1] = v_next  # type: ignore
        #    # shape (m, i+1)
        v_last = v_next
    # if store_tree:
    #    return v  # type: ignore
    return v_last[:, 0]  # type: ignore


@njit(parallel=True) # type: ignore
def build_upper_triangular(
    S: np.ndarray,
    K: np.ndarray,
    g: np.ndarray,
    factors: np.ndarray,
    n: int,
    call: bool,
) -> np.ndarray:
    # preallocate triangular array
    # shape (num_options, n+1, n+1), but we will only fill j <= k
    M = np.zeros((len(S), n + 1, n + 1))
    for j in range(n + 1):
        for k in range(j + 1):  # only k <= j
            M[:, j, k] = S[:] * factors[:, j] * np.exp(g[:] * (2 * k - j)) - K[:]
    
    if call:
        M= np.maximum(M, 0) # type: ignore
    else:
        M= np.maximum(-M, 0) # type: ignore

    return M

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
    

    def _build_upper_triangular(
        self,
        S: np.ndarray,
        K: np.ndarray,
        g: np.ndarray,
        factors: np.ndarray,
        n: int,
        call: bool = False,
    ):
        j = np.arange(n + 1)
        k = np.arange(n + 1)
        exp_term = np.exp(g[:, None, None] * (2 * k[None,None,:] - j[None,:,None]))  # shape (num_opt, n+1, n+1)
        base = S[:, None, None] * factors[:, :, None] * exp_term - K[:, None, None]
        M = np.tril(base, 0)  # upper triangular
        return np.maximum(M, 0) if call else np.maximum(-M, 0)

    def _build_payoff(
        self, S: np.ndarray, K: np.ndarray, g: np.ndarray, factors: np.ndarray, n: int
    ) -> np.ndarray:
        n_nodes = n + 1
        kj_matrix = 2 * np.arange(n_nodes).reshape(1, n_nodes) - np.arange(
            n_nodes
        ).reshape(n_nodes, 1)
        exp_tensor = np.exp(g[:, None, None] * kj_matrix[None, :, :])
        M = S[:, None, None] * exp_tensor * factors[:, None, :]
        return self._payoff(M, K[:, None, None])

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
                cash_paid = qr.q.cash_paid * S[:, None]
            else:
                cash_paid = qr.q.cash_paid
            principal_div = (cash_paid * qr.r.discount_factors).sum(axis=1)
            assert isinstance(principal_div, np.ndarray)
            S = np.clip(S - principal_div, SMALL, None)  # type: ignore

        g = vol * np.sqrt(T / n)
        u = np.exp(g)
        d = np.exp(-g)
        inter_disc = qr.r.intermediate_discount  # (len(T),n)
        assert not inter_disc is None
        p = (1 / inter_disc - d[:, None]) / (
            u[:, None] - d[:, None]
        )
        assert np.all(p>0) and np.all(p<1)
          # Assuming dividend is not factored into p

        # cached
        #exp_cache = [
        #    np.exp(g[:, None] * (2 * np.arange(i + 1)[None, :] - i))
        #    for i in range(n + 1)
        #]

        assert isinstance(qr.q.factors, np.ndarray)

        # payoff_matrix = self._build_payoff(S=S, K=K, g=g, factors=qr.q.factors, n=n)
        payoff_matrix = build_upper_triangular(
            S=S, K=K, g=g, factors=qr.q.factors, n=n, call=self.call
        )

        # if store_tree:
        #    self.v = loop(
        #        T=T,
        #        p=p,
        #        inter=inter_disc,
        #        pay=payoff_matrix,
        #        n=n,
        #        store_tree=store_tree,
        #        american=self.american,
        #    )
        #    return self.v[:, 0, 0]
        return loop(
            p=p,
            inter=inter_disc,
            pay=payoff_matrix,
            n=n,
            american=self.american,
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
                        S[:, None] * exp_cache[i] * qr.q.factors[:, i, None],
                        K[:, None],
                    ),
                    v_next,
                    out=v_next,
                )
            if store_tree:
                self.v[:, i, : i + 1] = v_next  # type: ignore
                # shape (m, i+1)
            v_last = v_next
        return v_last[:, 0]

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
