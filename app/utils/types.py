"""
This module contains typing
"""

from typing import Union, List
from pandera.typing import Series

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa


Numeric = Union[float, int]

ArrayLike = List[Numeric] | np.ndarray | Series[Numeric]
ArrayLike2 = List[List[Numeric]] | np.ndarray | pd.DataFrame
NP_SER_FLOAT = npt.NDArray[np.float64] | Series[float]
ARRAY_LIKE_RUNTIME_TYPES = (list, np.ndarray, pd.Series, pd.DataFrame)
NUMERIC_RUNTIME_TYPES = (float, int)
Dates = List[str] | List[pd.Timestamp] | Series[pd.Timestamp] | Series[str] | pd.DatetimeIndex
ndarray = np.typing.NDArray  # pylint: disable=C0103


class SurfaceSchema(pa.DataFrameModel):
    '''
    Class for handling options data needed for Surface Class in app.curves
    '''
    T: Series[float] = pa.Field(ge=0)
    k: Series[float] = pa.Field() # Ensures 'k' is a float column
    Vega: Series[float] = pa.Field(nullable=True)
    Volume: Series[float] = pa.Field(nullable=True)
    w: Series[float] = pa.Field(gt=0)
