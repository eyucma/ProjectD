"""
This module contains typing
"""

from typing import Union, List
from pandera.typing import Series

import numpy as np
import pandas as pd


Numeric = Union[float, int]

ArrayLike = List[Numeric] | np.ndarray | Series[Numeric]
ArrayLike2 = List[List[Numeric]] | np.ndarray | pd.DataFrame
ARRAY_LIKE_RUNTIME_TYPES = (list, np.ndarray, pd.Series, pd.DataFrame)
NUMERIC_RUNTIME_TYPES = (float, int)
Dates = List[str]|List[pd.Timestamp]|Series[pd.Timestamp]|Series[str]
ndarray = np.typing.NDArray #pylint: disable=C0103
