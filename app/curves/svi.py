# pylint: disable=C0103, R0913, R0914, R0915, R0917
"""
Defines utility functions for surfaces and smiles
"""
from typing import overload, Any, cast, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from pandera.typing import Series


ArrayF: TypeAlias = npt.NDArray[np.float64]
SeriesF: TypeAlias = Series[float]
NumberLike: TypeAlias = float | np.floating
Numeric: TypeAlias = ArrayF | SeriesF | NumberLike


@overload
def SVI(
    k: SeriesF, a: Numeric, b: Numeric, p: Numeric, m: Numeric, s: Numeric
) -> SeriesF: ...

@overload
def SVI(
    k: ArrayF, a: Numeric, b: Numeric, p: Numeric, m: Numeric, s: Numeric
) -> ArrayF: ...

@overload
def SVI(
    k: NumberLike,
    a: NumberLike,
    b: NumberLike,
    p: NumberLike,
    m: NumberLike,
    s: NumberLike,
) -> NumberLike: ...



def SVI(k: Any, a: Any, b: Any, p: Any, m: Any, s: Any) -> Numeric:
    """
    SVI computations
    """

    result = a + b * (p * (k - m) + np.sqrt((k - m) ** 2 + s**2))

    if isinstance(k, pd.Series):
        return cast(SeriesF, result)

    if isinstance(k, np.ndarray):
        return cast(ArrayF, result)

    return cast(NumberLike, result)
