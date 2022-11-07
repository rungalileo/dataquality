import pytest

from dataquality.utils.torch import (
    convert_fancy_idx_str_to_slice,
    validate_fancy_index_str,
)


def test_fancy_indexing_validation() -> None:
    """Test string to slice conversion validation"""
    assert validate_fancy_index_str("1,2,3")
    assert not validate_fancy_index_str("[1,2,3")
    assert not validate_fancy_index_str("1,2,3]")
    assert not validate_fancy_index_str("1.,..2,3]")
    assert validate_fancy_index_str("[1,2,3]")
    assert not validate_fancy_index_str("[1,2,3];import sys")


def test_fancy_indexing_() -> None:
    """Test string to slice conversion"""
    assert convert_fancy_idx_str_to_slice("[:, 1:, :]") == (
        slice(None, None, None),
        slice(1, None, None),
        slice(None, None, None),
    )

    # catch pytest.raises
    with pytest.raises(Exception):
        convert_fancy_idx_str_to_slice("[:, 1:, :];import sys;[0][0")
