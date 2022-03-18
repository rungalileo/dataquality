from typing import Callable, Type

import pytest

import dataquality
from dataquality.loggers.base_logger import T
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.model_logger.text_classification import (
    TextClassificationModelLogger,
)


def test_inference_e2e():
    pass


def test_set_split_inference(cleanup_after_use: Callable) -> None:
    assert not dataquality.get_data_logger().logger_config.inference_logged
    dataquality.set_split("inference", "all-customers")
    assert dataquality.get_data_logger().logger_config.cur_split == "inference"
    assert (
        dataquality.get_data_logger().logger_config.cur_inference_name
        == "all-customers"
    )


@pytest.mark.parametrize(
    "Logger", [TextClassificationDataLogger, TextClassificationModelLogger]
)
def test_log_input_data_sets_logged_config(
    set_test_config: Callable, Logger: Type[T]
) -> None:
    logger = Logger(split="inference")
    assert logger.logger_config.inference_logged
