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
    def test_init(
        self,
        fit_xgboost: xgb.XGBClassifier,
        sc_data: Dict,
    ) -> None:
        """Test that validate_inputs is called on init"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"],
            split="training",
        )
        for attr in [
            "model",
            "X",
            "y",
            "feature_names",
            "split",
            "inference_name",
            "probs",
        ]:
            assert hasattr(logger, attr)

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
        self, split: str, inference_name: Optional[str], create_logger: Callable
    ) -> None:
        """Test that validation works for all splits"""
        logger: StructuredClassificationDataLogger = create_logger(
            split=split, inference_name=inference_name
        )
        logger.validate()

    def validate_inputs(self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict) -> None:
        """Test that validate_inputs casts X and y to the correct types"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=list(sc_data["training"]["y"]),
            feature_names=sc_data["feature_names"],
            split="training",
        )
        logger.validate_and_prepare_logger()
        assert isinstance(logger.X, pd.DataFrame)
        assert isinstance(logger.y, np.ndarray)

    def test_set_probs(self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict) -> None:
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"],
            split="training",
        )
        assert logger.probs is None
        logger.set_probs()
        # 3 since wine dataset has 3 classes
        assert logger.probs.shape == (len(logger.X), 3)

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
        create_logger: Callable,
    ) -> None:
        """Test log method

        Very simple test that mocks all helper fns and makes sure they are called
        """
        set_test_config()
        mock_get_dfs.return_value = (mock.MagicMock(), mock.MagicMock())

        logger: StructuredClassificationDataLogger = create_logger(split="training")
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

    def test_get_dfs(self, create_logger: Callable, sc_data: Dict) -> None:
        # Set up logger with dataset, probs and split
        logger: StructuredClassificationDataLogger = create_logger(split="training")
        logger.validate_and_prepare_logger()

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
    def test_validate_missing_split(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"],
        )
        with pytest.raises(GalileoException) as e:
            logger.validate()

        assert str(e.value) == (
            "You didn't log a split and did not set a split. Use "
            "'dataquality.set_split' to set the split"
        )

    def test_validate_missing_inference_name(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"],
            split="inference",
        )
        with pytest.raises(GalileoException) as e:
            logger.validate()

        assert str(e.value) == (
            "For inference split you must either log an inference name "
            "or set it before logging. Use `dataquality.set_split` to set "
            "inference_name"
        )

    def test_validate_inputs_model_model_missing_predict_proba(self) -> None:
        """Test error is raised if model does not have a preodict_proba method"""
        logger = StructuredClassificationDataLogger(
            model=None,
            X=None,
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "Model must have a predict_proba method. "
            "If you are using a custom model, please implement a predict_proba method."
        )

    def test_validate_inputs_model_not_fitted(self) -> None:
        """Test error is raised if model is not already fit"""
        model = xgb.XGBClassifier()
        logger = StructuredClassificationDataLogger(
            model=model,
            X=None,
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == ("Model must be fit before logging data.")

    def test_validate_inputs_invalid_X_type(
        self, fit_xgboost: xgb.XGBClassifier
    ) -> None:
        """X is not a DataFrame or numpy array"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=3,
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "X must be a pandas DataFrame or numpy array, not <class 'int'>"
        )

    def test_validate_inputs_invalid_y_type(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """y is not a list or numpy array"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=3,
            feature_names=sc_data["feature_names"],
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "y must be a pandas Series, List, or numpy array of labels, "
            "not <class 'int'>"
        )

    def test_validate_inputs_X_and_y_different_lengths(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """X and y have different lengths"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"][:2],
            feature_names=sc_data["feature_names"],
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "X and y must be the same length. X has 142 rows, y has 2 rows"
        )

    def test_validate_inputs_data_feature_names_set(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """Feature names are not set and X is numpy array"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "feature_names must be provided when logging X as a numpy array. "
            "If X is a pandas DataFrame, feature_names will be inferred from the "
            "column names."
        )

    def test_validate_inputs_data_and_features_names_mismatch(
        self, fit_xgboost: xgb.XGBClassifier, sc_data: Dict
    ) -> None:
        """Test error is raised if X and features_names have different lengths"""
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"][1:],
            split="training",
        )
        with pytest.raises(AssertionError) as e:
            logger.validate_and_prepare_logger()

        assert str(e.value) == (
            "X and feature_names must have the same number of features"
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

    def test_log_pandas_e2e(
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
        """Test logging input pandas dfs for training, validation and test splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["training"]["df"],
            y=sc_data["training"]["y"],
            split="training",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["test"]["df"],
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

    def test_log_arrays_e2e(
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
        """Test logging input numpy arrays for training, validation and test splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            feature_names=sc_data["feature_names"],
            y=sc_data["training"]["y"],
            split="training",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["test"]["X"],
            feature_names=sc_data["feature_names"],
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

    def test_log_pandas_e2e_inference(
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
        """Test logging input pandas dfs for training and inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["training"]["df"],
            y=sc_data["training"]["y"],
            split="training",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf1"]["df"],
            split="inference",
            inference_name="inf1",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf2"]["df"],
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

    def test_log_arrays_e2e_inference(
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
        """Test logging input numpy arrays for training and inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["training"]["X"],
            y=sc_data["training"]["y"],
            feature_names=sc_data["feature_names"],
            split="training",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf1"]["X"],
            feature_names=sc_data["feature_names"],
            split="inference",
            inference_name="inf1",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf2"]["X"],
            feature_names=sc_data["feature_names"],
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

    def test_log_pandas_e2e_inference_only(
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
        """Test logging input data as pandas dfs for inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf1"]["df"],
            split="inference",
            inference_name="inf1",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf2"]["df"],
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

    def test_log_arrays_e2e_inference_only(
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
        """Test logging input data as numpy arrays for inference splits"""
        set_test_config(task_type="structured_classification")
        dq.set_labels_for_run(sc_data["labels"])
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf1"]["X"],
            feature_names=sc_data["feature_names"],
            split="inference",
            inference_name="inf1",
        )
        dq.log_xgboost(
            model=fit_xgboost,
            X=sc_data["inf2"]["X"],
            feature_names=sc_data["feature_names"],
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
