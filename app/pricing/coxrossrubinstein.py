# pylint: disable=C0103
"""
This module contains Binomial pricing function using CoxRossRubinstein defined as class
"""

from typing import Literal
from numba import njit, prange  # type: ignore

import numpy as np

from app.utils.types import ArrayLike
from app.modules.dividend_riskfree import Pairqr
from app.utils.convert import convert_to_numpy

SMALL = 1e-6


@njit(parallel=True)  # type: ignore
def loop(
    p: np.ndarray, inter: np.ndarray, pay: np.ndarray, n: int, american: bool = True
) -> np.ndarray:
    """
    Function to compute american backward iteration in the binomial tree
    """
    m = pay.shape[0]  # number of options
    v_last = pay[:, -1, :].copy()  # shape (m, n+1)

    for i in range(n - 1, -1, -1):
        v_next = np.zeros((m, i + 1))

        # vectorized over options
        for j in prange(i + 1):  # pylint: disable=E1133 # type: ignore
            v_next[:, j] = inter[:, i] * (
                p[:, i] * v_last[:, j + 1] + (1 - p[:, i]) * v_last[:, j]
            )

        if american:
            # ensure shapes match exactly
            v_next = np.maximum(v_next, pay[:, i, : i + 1])

        v_last = v_next  # next step

    return v_last[:, 0]


@njit(parallel=True)  # type: ignore
def build_upper_triangular(
    S: np.ndarray,
    K: np.ndarray,
    g: np.ndarray,
    factors: np.ndarray,
    n: int,
    call: bool,
) -> np.ndarray:
    """
    Constructs uppertriangular array for looping in the binomial tree
    """
    # preallocate triangular array
    # shape (num_options, n+1, n+1), but we will only fill j <= k
    M = np.zeros((len(S), n + 1, n + 1))
    for j in range(n + 1):
        for k in range(j + 1):  # only k <= j
            M[:, j, k] = S[:] * factors[:, j] * np.exp(g[:] * (2 * k - j)) - K[:]

    if call:
        M = np.maximum(M, 0)  # type: ignore
    else:
        M = np.maximum(-M, 0)  # type: ignore

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
        exp_term = np.exp(
            g[:, None, None] * (2 * k[None, None, :] - j[None, :, None])
        )  # shape (num_opt, n+1, n+1)
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
    ) -> np.ndarray:
        """
        Computes the option value as per the Binomial Tree
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
        p = (1 / inter_disc - d[:, None]) / (u[:, None] - d[:, None])
        assert np.all(p > 0) and np.all(p < 1)
        # Assuming dividend is not factored into p

        # cached
        # exp_cache = [
        #    np.exp(g[:, None] * (2 * np.arange(i + 1)[None, :] - i))
        #    for i in range(n + 1)
        # ]

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
    )->np.ndarray:
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
        """
        safe method to shuffle calls/puts
        """
        if method == "call":
            self.call = True
        else:
            self.call = False

    def set_type(self, options_type: Literal["American", "European"]) -> None:
        """
        safe method to shuffle american/european
        """
        if options_type == "American":
            self.american = True
        else:
            self.american = False
