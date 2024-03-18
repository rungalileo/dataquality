import os
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import vaex
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.setfit import auto, watch
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCAL_MODEL_PATH, TestSessionVariables


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_setfit_watch(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    dataset = Dataset.from_dict(
        {"text": ["hello", "world", "foo", "bar"], "label": [0, 1] * 2}
    )
    model = SetFitModel.from_pretrained(LOCAL_MODEL_PATH)

    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        num_iterations=1,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()

    labels = ["nocat", "cat"]
    dq.set_labels_for_run(labels)
    split = "training"
    batch_size = 3
    ds_len = len(dataset)
    dq_evaluate = watch(
        model,
        validate_before_training=False,
    )
    dataset = dataset.map(
        lambda x, idx: {"id": idx, "meta_col": "meta"}, with_indices=True
    )
    for i in range(0, ds_len, batch_size):
        batch = dataset[i : i + batch_size]
        dq_evaluate(
            batch,
            split=split,
            meta=["meta_col"],
        )
    ThreadPoolManager.wait_for_threads()
    dq.get_data_logger().upload()
    train_data = vaex.open(f"{test_session_vars.TEST_PATH}/training/0/data/data.hdf5")
    assert train_data["meta_col"].unique() == ["meta"]
    dq.finish()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_log_dataset(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    dataset = Dataset.from_dict(
        {"text": ["hello", "world", "foo", "bar"], "label": [0, 1] * 2}
    )
    model = SetFitModel.from_pretrained(LOCAL_MODEL_PATH)
    set_test_config(task_type="text_classification")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        num_iterations=1,
        column_mapping={"text": "text", "label": "label"},
    )
    trainer.train()
    labels = ["nocat", "cat"]
    dq.set_labels_for_run(labels)
    split = "training"
    batch_size = 3
    ds_len = len(dataset)
    dataset = dataset.map(
        lambda x, idx: {"id": idx, "meta_col": "meta"}, with_indices=True
    )
    dq.log_dataset(dataset, split=split, meta=["meta_col"])
    dq_evaluate = watch(
        model,
        validate_before_training=False,
    )
    for i in range(0, ds_len, batch_size):
        batch = dataset[i : i + batch_size]
        dq_evaluate(batch, split)

    ThreadPoolManager.wait_for_threads()
    dq.get_data_logger().upload()
    train_data = vaex.open(f"{test_session_vars.TEST_PATH}/training/0/data/data.hdf5")
    assert train_data["meta_col"].unique() == ["meta"]
    dq.finish()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_setfit_trainer(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    dataset = Dataset.from_dict(
        {"text": ["hello", "world", "foo", "bar"], "label": [0, 1] * 2}
    )

    set_test_config(task_type="text_classification")

    # 🔭🌕 Galileo logging
    from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitModel, SetFitTrainer

    import dataquality as dq
    from dataquality.integrations.setfit import watch

    model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SetFitModel.from_pretrained(model_id, use_differentiable_head=True)
    column_mapping = {"text": "text", "label": "label"}
    dataset = dataset.map(
        lambda x, idx: {"id": idx, "meta_col": "meta"}, with_indices=True
    )
    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        loss_class=CosineSimilarityLoss,
        num_iterations=1,
        column_mapping=column_mapping,
    )
    labels = ["nocat", "cat"]

    # 🔭🌕 Galileo logging
    watch(
        trainer,
        labels=labels,
        batch_size=512,  # Speed up prediction
        # 🔭🌕 Set finish to False to add test
        finish=False,
        validate_before_training=False,
    )
    trainer.freeze()
    trainer.train()
    trainer.unfreeze(keep_body_frozen=True)
    dq.set_epoch(1)
    trainer.train()
    dq_evaluate = watch(
        model,
        batch_size=512,
        validate_before_training=False,
    )
    dq_evaluate(
        dataset,
        split="test",
        column_mapping=column_mapping,
        # for inference set the split to inference
        # and pass an inference_name="inference_run_1"
    )
    dq.finish()


@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dq.clients.api.ApiClient,
    "get_healthcheck_dq",
    return_value={
        "bucket_names": {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        },
        "minio_fqdn": "127.0.0.1:9000",
    },
)
@patch.object(ApiClient, "get_current_user", return_value={"email": "hi@example.com"})
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_auto(
    mock_valid_user: MagicMock,
    mock_get_current_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    mock_get_project_by_name.return_value = {"id": test_session_vars.DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": test_session_vars.DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    example_data = {"text": ["hello", "world", "foo", "bar"] * 2, "label": [0, 1] * 4}
    dataset = Dataset.from_dict(example_data)
    dq.utils.setfit.BATCH_LOG_SIZE = 1
    set_test_config(
        task_type="text_classification",
        project_name="test_project",
        run_name="test_run",
    )
    df = pd.DataFrame(example_data)

    dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True)
    column_mapping = {"text": "text", "label": "label"}

    labels = ["nocat", "cat"]
    eval_ds = dataset.remove_columns("label").map(lambda x: {"meta_col": "meta"})
    os.environ["DQ_SKIP_FINISH"] = "1"
    model = auto(
        train_data=df,
        val_data=dataset,
        test_data=dataset,
        inference_data={"inftest": eval_ds},
        project_name="project_name",
        run_name="labels",
        training_args={"num_epochs": 1, "num_iterations": 1},
        labels=labels,
        column_mapping=column_mapping,
    )

    dq_evaluate = watch(
        model,
        project_name="project_name",
        run_name="labels",
        labels=labels,
        finish=False,
        batch_size=2,
    )

    ThreadPoolManager.wait_for_threads()

    # print(os.listdir(f"{test_session_vars.TEST_PATH}/inference/"))
    dq.get_data_logger().upload()

    dq_evaluate(
        eval_ds,
        split="inference",
        meta=["meta_col"],
        # for inference set the split to inference
        inference_name="inference_run_1",
        batch_size=2,
    )
    dq.get_data_logger().upload()

    inf_data_2 = vaex.open(
        f"{test_session_vars.TEST_PATH}/inference/inference_run_1/data/data.hdf5"
    )
    assert inf_data_2["meta_col"].unique() == ["meta"]
