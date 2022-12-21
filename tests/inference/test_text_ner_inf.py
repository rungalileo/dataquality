from typing import Callable
from unittest import mock

import datasets
import pandas as pd
import pytest
import vaex

import dataquality
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_ner import TextNERDataLogger
from dataquality.schemas.split import Split

TEST_LABELS = ["[PAD]", "[CLS]", "[SEP]", "O", "B-ACTOR", "I-ACTOR"]


class TestTextNERDataLoggerInference:
    def _setup(self, **kwargs) -> TextNERDataLogger:
        logger = TextNERDataLogger(split=Split.inference, **kwargs)
        logger.logger_config.reset()
        logger.logger_config.labels = TEST_LABELS
        logger.logger_config.tagging_schema = "BIO"
        return logger

    def test_validate_inference(self):
        logger = self._setup(**{"inference_name": "animals"})
        logger.validate()

    def test_validate_inference_with_labels(self):
        gold_spans = [
            {"start": 0, "end": 4, "label": "YEAR"},
            {"start": 17, "end": 29, "label": "ACTOR"},
        ]
        logger = self._setup(**{"gold_spans": gold_spans, "inference_name": "animals"})

        with pytest.raises(AssertionError) as e:
            logger.validate()

        assert e.value.args[0] == "You cannot have gold spans in your inference split!"

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
                    "text_token_indices": [[(1, 4), (5, 8)], [(0, 4)], [(4, 9)]],
                }
            ),
            vaex.from_dict(
                {
                    "my_text": ["sample1", "sample2", "sample3"],
                    "my_id": [1, 2, 3],
                    "text_token_indices": [[(1, 4), (5, 8)], [(0, 4)], [(4, 9)]],
                }
            ),
            [
                {
                    "my_text": "sample1",
                    "my_id": 1,
                    "text_token_indices": [(1, 4), (5, 8)],
                },
                {"my_text": "sample2", "my_id": 2, "text_token_indices": [(0, 4)]},
                {"my_text": "sample3", "my_id": 3, "text_token_indices": [(4, 9)]},
            ],
            datasets.Dataset.from_dict(
                dict(
                    my_text=["sample1", "sample2", "sample3"],
                    my_id=[1, 2, 3],
                    text_token_indices=[[(1, 4), (5, 8)], [(0, 4)], [(4, 9)]],
                )
            ),
        ],
    )
    def test_log_dataset(
        self, dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
    ) -> None:
        set_test_config(split="inference")
        logger = self._setup(**{"inference_name": "animals"})

        with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
            mock_method.return_value = logger
            dataquality.log_dataset(
                dataset,
                text="my_text",
                id="my_id",
                split="inference",
                inference_name="animals",
            )

            assert logger.texts == ["sample1", "sample2", "sample3"]
            assert logger.ids == [1, 2, 3]
            assert logger.text_token_indices_flat == [[1, 4, 5, 8], [0, 4], [4, 9]]
            assert logger.split == Split.inference
            assert logger.inference_name == "animals"
