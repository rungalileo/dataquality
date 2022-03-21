from typing import Callable

import pytest

import dataquality


def test_set_split_inference(cleanup_after_use: Callable) -> None:
    assert not dataquality.get_data_logger().logger_config.inference_logged
    dataquality.set_split("inference", "all-customers")
    assert dataquality.get_data_logger().logger_config.cur_split == "inference"
    assert (
        dataquality.get_data_logger().logger_config.cur_inference_name
        == "all-customers"
    )


def test_set_split_inference_missing_inference_name() -> None:
    with pytest.raises(AssertionError):
        dataquality.set_split("inference")
