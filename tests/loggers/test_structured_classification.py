from typing import Callable, Dict, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from vaex.dataframe import DataFrame

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.structured_classification import (
    StructuredClassificationDataLogger,
)
from dataquality.schemas.job import JobName
from dataquality.schemas.request_type import RequestType
from dataquality.schemas.task_type import TaskType
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID


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
        logger.set_probs()
        logger.validate()

    def test_create_dataset_from_samples(self, sc_data: Dict) -> None:
        dataset = StructuredClassificationDataLogger.create_dataset_from_samples(
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"].copy(),
        )
        assert isinstance(dataset, pd.DataFrame)
        n_rows = sc_data["training"]["X"].shape[0]
        n_features = sc_data["training"]["X"].shape[1]
        assert dataset.shape == (n_rows, n_features + 1)
        assert "gold" in dataset.columns

        # Test that we can create a dataset without labels for inference
        dataset = StructuredClassificationDataLogger.create_dataset_from_samples(
            X=sc_data["inf1"]["X"],
            y=None,
            feature_names=sc_data["feature_names"].copy(),
        )
        assert isinstance(dataset, pd.DataFrame)
        n_rows = sc_data["inf1"]["X"].shape[0]
        n_features = sc_data["inf1"]["X"].shape[1]
        assert dataset.shape == (n_rows, n_features)
        assert "gold" not in dataset.columns

        # Test that inference works with empty y numpy array as well
        dataset = StructuredClassificationDataLogger.create_dataset_from_samples(
            X=sc_data["inf1"]["X"],
            y=np.array([]),
            feature_names=sc_data["feature_names"].copy(),
        )
        assert isinstance(dataset, pd.DataFrame)
        assert dataset.shape == (n_rows, n_features)
        assert "gold" not in dataset.columns

    def test_set_probs(self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict) -> None:
        logger = StructuredClassificationDataLogger()
        logger.model = fit_xgboost
        logger.dataset = pd.DataFrame(
            sc_data["training"]["X"], columns=sc_data["feature_names"].copy()
        )

        assert len(logger.probs) == 0
        logger.set_probs()
        # 3 since wine dataset has 3 classes
        assert logger.probs.shape == (len(logger.dataset), 3)

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_log_samples(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """Test log_samples sets dataset and split attributes and calls log method"""
        logger = StructuredClassificationDataLogger()
        assert not hasattr(logger, "dataset")
        assert logger.split is None
        logger.log_samples(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"].copy(),
            split="training",
        )
        assert hasattr(logger, "dataset")
        assert logger.split == "training"
        assert logger.dataset.shape == sc_data["training"]["dataset"].shape
        mock_log.assert_called_once()

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_log_structured_dataset(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """Test log_structured_dataset

        Test that this fn:
          - sets dataset gold col
          - sets split attributes
          - calls log method
        """
        logger = StructuredClassificationDataLogger()
        assert not hasattr(logger, "dataset")
        assert logger.split is None

        dataset = sc_data["training"]["dataset"].copy()
        assert "gold" not in dataset.columns

        logger.log_structured_dataset(
            model=fit_xgboost,
            dataset=dataset,
            label="my-label",
            split="training",
        )
        assert logger.split == "training"
        assert logger.dataset.shape == sc_data["training"]["dataset"].shape
        assert "gold" in logger.dataset.columns
        mock_log.assert_called_once()

    @mock.patch.object(ObjectStore, "create_project_run_object_from_df")
    @mock.patch.object(StructuredClassificationDataLogger, "_get_dfs")
    @mock.patch.object(StructuredClassificationDataLogger, "validate")
    @mock.patch.object(StructuredClassificationDataLogger, "set_probs")
    def test_log(
        self,
        mock_set_probs: mock.MagicMock,
        mock_validate: mock.MagicMock,
        mock_get_dfs: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
    ) -> None:
        """Test log method

        Very simple test that mocks all helper fns and makes sure they are called
        """
        set_test_config()
        mock_get_dfs.return_value = (mock.MagicMock(), mock.MagicMock())

        logger = StructuredClassificationDataLogger()
        logger.split = "training"
        logger.log()

        mock_set_probs.assert_called_once_with()
        mock_validate.assert_called_once_with()

        mock_get_dfs.assert_called_once_with()
        assert mock_upload_df_to_minio.call_count == 2
        mock_upload_df_to_minio.assert_any_call(
            mock.ANY, f"{DEFAULT_PROJECT_ID}/{DEFAULT_RUN_ID}/training/data/data.hdf5"
        )
        mock_upload_df_to_minio.assert_any_call(
            mock.ANY, f"{DEFAULT_PROJECT_ID}/{DEFAULT_RUN_ID}/training/prob/prob.hdf5"
        )

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_get_dfs(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        # Set up logger with dataset, probs and split
        logger = StructuredClassificationDataLogger()
        dataset = sc_data["training"]["dataset"].copy()
        logger.log_structured_dataset(
            model=fit_xgboost,
            dataset=dataset,
            label="my-label",
            split="training",
        )
        logger.set_probs()

        # Assertions on returned dfs
        df, probs_df = logger._get_dfs()
        assert isinstance(df, DataFrame)
        expected_cols = sc_data["feature_names"].copy()
        expected_cols += ["id", "split", "data_schema_version"]
        assert sorted(df.get_column_names()) == sorted(expected_cols)

        assert isinstance(probs_df, DataFrame)
        expected_cols = ["prob", "id", "gold"]
        assert sorted(probs_df.get_column_names()) == sorted(expected_cols)

        assert len(df) == len(probs_df)


class TestStructuredClassificationValidationErrors:
    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_validate_missing_split(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        # Set up logger with dataset, probs and split
        logger = StructuredClassificationDataLogger()
        logger.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
        )
        with pytest.raises(GalileoException) as e:
            logger.validate()

        assert str(e.value) == (
            "You didn't log a split and did not set a split. Use "
            "'dataquality.set_split' to set the split"
        )

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_validate_missing_inference_name(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        # Set up logger with dataset, probs and split
        logger = StructuredClassificationDataLogger()
        logger.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
            split="inference",
        )
        with pytest.raises(GalileoException) as e:
            logger.validate()

        assert str(e.value) == (
            "For inference split you must either log an inference name "
            "or set it before logging. Use `dataquality.set_split` to set "
            "inference_name"
        )

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_validate_data_probs_different_dims(
        self, mock_log: mock.MagicMock, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """Test error is raised if X, y, and probs have different dimensions"""
        # Set up logger with dataset, probs and split
        logger = StructuredClassificationDataLogger()
        logger.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
            split="training",
        )
        logger.set_probs()
        logger.probs = logger.probs[1:]
        with pytest.raises(AssertionError) as e:
            logger.validate()

        assert str(e.value) == (
            "Data and probs are not the same length. " "Data: 142, Probs: 141"
        )

    def test_validate_data_and_features_names_mismatch(self) -> None:
        """Test error is raised if X and features_names have different lengths"""

    @mock.patch.object(StructuredClassificationDataLogger, "log")
    def test_set_probs_missing_model(
        self, mock_log: mock.MagicMock, sc_data: Dict
    ) -> None:
        """If logger model is not set, raise error"""
        logger = StructuredClassificationDataLogger()
        logger.log_structured_dataset(
            model=None,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.set_probs()

        assert str(e.value) == (
            "Model must be set before setting probs. Try calling "
            "`log_structured_dataset`"
        )

    def test_set_probs_missing_dataset(self) -> None:
        """If logger dataset is not set, raise error"""
        logger = StructuredClassificationDataLogger()
        logger.model = mock.MagicMock()
        logger.dataset = None
        with pytest.raises(AssertionError) as e:
            logger.set_probs()

        assert str(e.value) == (
            "Dataset must be set before setting probs. Try calling "
            "`log_structured_dataset`"
        )


@mock.patch.object(ObjectStore, "create_project_run_object_from_df")
@mock.patch("dataquality.core.finish._version_check")
@mock.patch("dataquality.core.finish._reset_run")
@mock.patch("dataquality.core.finish.upload_dq_log_file")
@mock.patch.object(
    ApiClient, "make_request", return_value={"link": "link", "job_name": "job_name"}
)
class TestStructuredClassificationE2E:
    def _assert_mocks(
        self,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        inf_only: bool = False,
    ) -> None:
        mock_version_check.assert_called_once_with()
        mock_upload_dq_log_file.assert_called_once_with()
        if inf_only:
            mock_reset_run.assert_not_called()
        else:
            mock_reset_run.assert_called_once_with(
                DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, TaskType.structured_classification
            )

    def test_log_structured_samples_e2e(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured samples for training, validation and test splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            y=sc_data["training"]["y"],
            split="training",
        )
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["test"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            y=sc_data["test"]["y"],
            split="test",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (training and test)
        assert mock_upload_df_to_minio.call_count == 4
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
            },
        )
        self._assert_mocks(mock_upload_dq_log_file, mock_reset_run, mock_version_check)

    def test_log_structured_dataset_e2e(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured dataset for training, validation and test splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
            split="training",
        )
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["test"]["dataset"].copy(),
            label="my-label",
            split="test",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (training and test)
        assert mock_upload_df_to_minio.call_count == 4
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
            },
        )
        self._assert_mocks(mock_upload_dq_log_file, mock_reset_run, mock_version_check)

    def test_log_structured_samples_e2e_inference(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured samples for training and inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            y=sc_data["training"]["y"],
            split="training",
        )
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["inf1"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            split="inference",
            inference_name="inf1",
        )
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["inf2"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            split="inference",
            inference_name="inf2",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (training and 2 inf)
        assert mock_upload_df_to_minio.call_count == 6
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
                "job_name": JobName.inference,
                "non_inference_logged": True,
            },
        )
        self._assert_mocks(mock_upload_dq_log_file, mock_reset_run, mock_version_check)

    def test_log_structured_dataset_e2e_inference(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured dataset for training and inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["training"]["dataset"].copy(),
            label="my-label",
            split="training",
        )
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["inf1"]["dataset"].copy(),
            split="inference",
            inference_name="inf1",
        )
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["inf2"]["dataset"].copy(),
            split="inference",
            inference_name="inf2",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (training and 2 inf)
        assert mock_upload_df_to_minio.call_count == 6
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
                "job_name": JobName.inference,
                "non_inference_logged": True,
            },
        )
        self._assert_mocks(mock_upload_dq_log_file, mock_reset_run, mock_version_check)

    def test_log_structured_samples_e2e_inference_only(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured samples for inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["inf1"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            split="inference",
            inference_name="inf1",
        )
        dq.log_structured_samples(
            model=fit_xgboost,
            X=sc_data["inf2"]["X"],
            feature_names=sc_data["feature_names"].copy(),
            split="inference",
            inference_name="inf2",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (2 inf)
        assert mock_upload_df_to_minio.call_count == 4
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
                "job_name": JobName.inference,
                "non_inference_logged": False,
            },
        )
        self._assert_mocks(
            mock_upload_dq_log_file, mock_reset_run, mock_version_check, True
        )

    def test_log_structured_dataset_e2e_inference_only(
        self,
        mock_create_job: mock.MagicMock,
        mock_upload_dq_log_file: mock.MagicMock,
        mock_reset_run: mock.MagicMock,
        mock_version_check: mock.MagicMock,
        mock_upload_df_to_minio: mock.MagicMock,
        set_test_config: Callable,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test logging structured dataset for inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["inf1"]["dataset"].copy(),
            split="inference",
            inference_name="inf1",
        )
        dq.log_structured_dataset(
            model=fit_xgboost,
            dataset=sc_data["inf2"]["dataset"].copy(),
            split="inference",
            inference_name="inf2",
        )
        dq.finish(wait=False)

        # We upload df and probs_df for each split (2 inf)
        assert mock_upload_df_to_minio.call_count == 4
        mock_create_job.assert_called_once_with(
            RequestType.POST,
            url="http://localhost:8088/jobs",
            body={
                "project_id": str(DEFAULT_PROJECT_ID),
                "run_id": str(DEFAULT_RUN_ID),
                "labels": ["class_0", "class_1", "class_2"],
                "task_type": "structured_classification",
                "tasks": None,
                "ner_labels": [],
                "job_name": JobName.inference,
                "non_inference_logged": False,
            },
        )
        self._assert_mocks(
            mock_upload_dq_log_file, mock_reset_run, mock_version_check, True
        )
