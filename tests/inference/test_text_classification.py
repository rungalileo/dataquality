from pathlib import Path
from typing import Callable, List, Type
from unittest import mock

import datasets
import pandas as pd
import pytest
import tensorflow as tf
import vaex
from pydantic import ValidationError

import dataquality
from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseGalileoLogger, T
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split


class TestSetSplitInference:
    def test_set_split_inference(self) -> None:
        assert not dataquality.get_data_logger().logger_config.inference_logged
        dataquality.set_split("inference", "all-customers")
        assert dataquality.get_data_logger().logger_config.cur_split == "inference"
        assert (
            dataquality.get_data_logger().logger_config.cur_inference_name
            == "all-customers"
        )

    def test_set_split_inference_missing_inference_name(self) -> None:
        with pytest.raises(ValidationError) as e:
            dataquality.set_split("inference")

        assert (
            e.value.errors()[0]["msg"]
            == "Please specify inference_name when setting split to inference"
        )


class TestBaseLoggersInference:
    # Use _setup since setup is a protected pytest var
    def _setup(self, logger_cls: Type[T]) -> Type[T]:
        logger = logger_cls()
        logger.logger_config.reset()
        return logger

    def test_validate_inference(self) -> None:
        base_logger = self._setup(logger_cls=BaseGalileoLogger)

        assert not base_logger.split
        assert not base_logger.logger_config.inference_logged

        base_logger.logger_config.cur_inference_name = "customers"
        base_logger.logger_config.cur_split = Split.inference
        base_logger.set_split_epoch()

        assert base_logger.split
        assert base_logger.logger_config.inference_logged

    def test_base_model_logger_validate_inference(self) -> None:
        base_model_logger = self._setup(BaseGalileoModelLogger)

        assert not base_model_logger.split
        assert not base_model_logger.logger_config.inference_logged

        base_model_logger.logger_config.cur_inference_name = "customers"
        base_model_logger.logger_config.cur_split = Split.inference

        base_model_logger.set_split_epoch()
        assert base_model_logger.split
        assert base_model_logger.logger_config.inference_logged

    def test_base_model_logger_validate_inference_missing_inference_name(self) -> None:
        base_model_logger = self._setup(BaseGalileoModelLogger)

        assert not base_model_logger.split
        assert not base_model_logger.logger_config.inference_logged

        base_model_logger.split = Split.inference
        with pytest.raises(GalileoException) as e:
            base_model_logger.set_split_epoch()

        assert e.value.args[0] == (
            "For inference split you must either log an inference name or set "
            "it before logging. Use `dataquality.set_split` to setinference_name"
        )

    @mock.patch(
        "dataquality.loggers.model_logger.base_model_logger._save_hdf5_file",
        return_value="1234-abcd-5678",
    )
    @mock.patch(
        "dataquality.loggers.model_logger.base_model_logger.uuid4",
        return_value="1234-abcd-5678",
    )
    def test_write_model_output_inference(
        self, mock_uuid: mock.MagicMock, mock_save_file: mock.MagicMock
    ) -> None:
        inference_data = {
            "epoch": [None, None, None],
            "split": ["inference", "inference", "inference"],
            "inference_name": ["customers", "customers", "customers"],
        }
        logger = BaseGalileoModelLogger()
        logger.write_model_output(inference_data)

        local_file = (
            f"{Path.home()}/.galileo/logs/{dataquality.config.current_project_id}/"
            f"{dataquality.config.current_run_id}/inference/customers"
        )
        # Assert _save_hdf5_file is called with correct args
        mock_save_file.assert_called_once_with(
            local_file, "1234abcd5678.hdf5", inference_data
        )

    @pytest.mark.parametrize(
        "splits,expected",
        [
            (["training_logged"], True),
            (["test_logged"], True),
            (["validation_logged"], True),
            (["inference_logged"], False),
            (["training_logged", "inference_logged"], True),
            (["test_logged", "validation_logged"], True),
        ],
    )
    def test_non_inference_logged(self, splits: List[str], expected: bool) -> None:
        base_logger = self._setup(logger_cls=BaseGalileoLogger)
        for split in splits:
            setattr(base_logger.logger_config, split, True)

        assert base_logger.non_inference_logged() is expected


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
