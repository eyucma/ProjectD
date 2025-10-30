"""
This module contains utility functions converting types
"""

import numpy as np
import pandas as pd

from app.utils.types import ArrayLike, ARRAY_LIKE_RUNTIME_TYPES, NUMERIC_RUNTIME_TYPES


def convert_to_numpy(data: ArrayLike | float | int) -> np.ndarray:
    """
    Converts supported inputs (scalars and containers) to a standardized np.ndarray.
    Performs deep validation on lists to ensure only numerics are present.
    """

    # 1. Handle scalar inputs
    if isinstance(data, NUMERIC_RUNTIME_TYPES):
        return np.array(data)

    # 2. Handle supported container types
    if isinstance(data, ARRAY_LIKE_RUNTIME_TYPES):

        # Deep validation for lists
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, NUMERIC_RUNTIME_TYPES):
                    raise TypeError(
                        f"List input contains unsupported element type: {type(item).__name__}"
                    )
            return np.asarray(data)

        # Handle Pandas conversion (avoids double copy if input is already np.ndarray)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return np.asarray(data.values)
        return data

        # np.asarray handles the final conversion efficiently

    # 3. Handle unsupported types
    raise TypeError(f"Unsupported input type: {type(data).__name__}")
