class TestStructuredClassificationDataLogger:
    def test_validate(self) -> None:
        pass

    def test_log_samples(self) -> None:
        pass

    def test_log_structured_dataset(self) -> None:
        pass

    def test_log(self) -> None:
        pass

    def test_get_data_dict(self) -> None:
        pass

    def test_upload(self) -> None:
        pass

    def test_upload_split_from_path(self) -> None:
        pass

    def test_separate_dataframe(self) -> None:
        pass

    def test_get_split_upload_folders(self) -> None:
        pass


class TestStructuredClassificationValidationErrors:
    def test_validate_missing_split(self) -> None:
        pass

    def test_validate_inference_missing_inference_name(self) -> None:
        pass

    def test_validate_data_labels_probs_different_dims(self) -> None:
        """Teset error is raised if X, y, and probs have different dimensions"""

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
