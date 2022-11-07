from typing import Callable
from unittest import mock

import datasets
import pandas as pd
import pytest
import tensorflow as tf
import vaex

import dataquality
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.schemas.split import Split


class TestTextClassificationDataLoggerInference:
    def _setup(self, **kwargs) -> TextClassificationDataLogger:
        logger = TextClassificationDataLogger(split=Split.inference, **kwargs)
        logger.logger_config.reset()
        return logger

    def test_validate_inference(self):
        logger = self._setup(**{"inference_name": "animals"})
        logger.validate()

    def test_validate_inference_with_labels(self):
        logger = self._setup(**{"labels": ["dog", "cat"], "inference_name": "animals"})
        with pytest.raises(AssertionError) as e:
            logger.validate()

        assert e.value.args[0] == "You cannot have labels in your inference split!"

    def test_validate_inference_missing_inference_name(self):
        logger = self._setup()
        with pytest.raises(AssertionError) as e:
            logger.validate()

        assert e.value.args[0] == (
            "Inference name must be set when logging an inference split. Use "
            "set_split('inference', inference_name) to set inference name"
        )

    @pytest.mark.parametrize(
        "dataset",
        [
            pd.DataFrame(
                {
                    "my_text": ["sample1", "sample2", "sample3"],
                    "my_id": [1, 2, 3],
                }
            ),
            vaex.from_dict(
                {
                    "my_text": ["sample1", "sample2", "sample3"],
                    "my_id": [1, 2, 3],
                }
            ),
            [
                {"my_text": "sample1", "my_id": 1},
                {"my_text": "sample2", "my_id": 2},
                {"my_text": "sample3", "my_id": 3},
            ],
            tf.data.Dataset.from_tensor_slices(
                {
                    "my_text": ["sample1", "sample2", "sample3"],
                    "my_id": [1, 2, 3],
                }
            ),
            datasets.Dataset.from_dict(
                dict(
                    my_text=["sample1", "sample2", "sample3"],
                    my_id=[1, 2, 3],
                )
            ),
        ],
    )
    def test_log_dataset(
        self, dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
    ) -> None:
        set_test_config(split="inference")
        logger = TextClassificationDataLogger()

        with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
            mock_method.return_value = logger
            dataquality.log_dataset(
                dataset,
                text="my_text",
                id="my_id",
                split="inference",
                inference_name="foo",
            )

            assert logger.texts == ["sample1", "sample2", "sample3"]
            assert logger.ids == [1, 2, 3]
            assert logger.split == Split.inference
            assert logger.inference_name == "foo"
