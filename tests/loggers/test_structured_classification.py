from typing import Callable, Optional

import pytest

from dataquality.loggers.data_logger.structured_classification import (
    StructuredClassificationDataLogger,
)


class TestStructuredClassificationDataLogger:
    @pytest.mark.parametrize(
        "split, inference_name",
        [
            ("training", None),
            ("test", None),
            ("inference", "inf1"),
            ("inference", "inf2"),
        ],
    )
    def test_validate(
        self, split: str, inference_name: Optional[str], sc_data_logger: Callable
    ) -> None:
        logger: StructuredClassificationDataLogger = sc_data_logger(
            split=split, inference_name=inference_name
        )
        logger.validate()
    
    def test_create_dataset_from_samples(self) -> None:
        pass

    def test_set_probs(self) -> None:
        pass

    def test_log_samples(self) -> None:
        pass

    def test_log_structured_dataset(self) -> None:
        pass

    def test_log(self) -> None:
        pass

    def test_get_dfs(self) -> None:
        pass



class TestStructuredClassificationValidationErrors:
    def test_validate_missing_split(self) -> None:
        pass

    def test_validate_inference_missing_inference_name(self) -> None:
        pass

    def test_validate_data_labels_probs_different_dims(self) -> None:
        """Test error is raised if X, y, and probs have different dimensions"""

    def test_validate_data_and_features_names_mismatch(self) -> None:
        """Test error is raised if X and features_names have different lengths"""


class TestStructuredClassification:
    def test_log_structured_samples_e2e(self) -> None:
        """Test logging structured samples for training, validation and test splits"""

    def test_log_structured_dataset_e2e(self) -> None:
        """Test logging structured dataset for training, validation and test splits"""

    def test_log_structured_samples_e2e_inference(self) -> None:
        """Test logging structured samples for training and inference splits"""

    def test_log_structured_dataset_e2e_inference(self) -> None:
        """Test logging structured dataset for training and inference splits"""

    def test_log_structured_samples_e2e_inference_only(self) -> None:
        """Test logging structured samples for inference splits"""

    def test_log_structured_dataset_e2e_inference_only(self) -> None:
        """Test logging structured dataset for inference splits"""
