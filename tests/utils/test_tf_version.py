from unittest import mock

from dataquality.utils.tf import is_tf_2


@mock.patch("tensorflow.__version__", "2.0.1")
def test_is_tf_2() -> None:
    assert is_tf_2() is True


@mock.patch("tensorflow.__version__", "1.15.0")
def test_is_tf_2_not_2() -> None:
    assert is_tf_2() is False
