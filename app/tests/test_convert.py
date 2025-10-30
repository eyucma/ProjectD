"""
Testing for convert.py
"""

import pytest
from app.utils.convert import convert_to_numpy


def test_convert_to_numpy_raises_error():
    """Test that _convert_to_numpy raises TypeError for unsupported types."""
    with pytest.raises(TypeError):
        convert_to_numpy("not a number")  # type: ignore

    # Test for set
    with pytest.raises(TypeError):
        convert_to_numpy({"unsupported", "data"})  # type: ignore

    with pytest.raises(TypeError):
        convert_to_numpy({"S": 100})  # type: ignore
