"""
This module contains typing
"""

from typing import Union, List, Callable

import numpy as np
import pandas as pd


Numeric = Union[float, int]

ArrayLike = List[Numeric] | np.ndarray | pd.Series | pd.DataFrame

ARRAY_LIKE_RUNTIME_TYPES = (list, np.ndarray, pd.Series, pd.DataFrame)
NUMERIC_RUNTIME_TYPES = (float, int)



