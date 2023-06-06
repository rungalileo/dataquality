from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import vaex
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.setfit import watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import (
    DEFAULT_PROJECT_ID,
    DEFAULT_RUN_ID,
    LOCAL_MODEL_PATH,
    TEST_PATH,
)

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


@patch.object(ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
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
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_setfitwatch(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    global dataset
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)

    dq.init(task_type=TaskType.text_classification)
    labels = ["nocat", "cat"]
    dq.set_labels_for_run(labels)
    split = "training"
    batch_size = 3
    ds_len = len(dataset)
    dq_evaluate = watch(model)
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
    train_data = vaex.open(f"{TEST_PATH}/training/0/data/data.hdf5")
    assert train_data["meta_col"].unique() == ["meta"]
    dq.finish()


@patch.object(ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
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
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_log_dataset(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    global dataset
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)

    dq.init(task_type=TaskType.text_classification)
    labels = ["nocat", "cat"]
    dq.set_labels_for_run(labels)
    split = "training"
    batch_size = 3
    ds_len = len(dataset)
    dataset = dataset.map(
        lambda x, idx: {"id": idx, "meta_col": "meta"}, with_indices=True
    )
    dq.log_dataset(dataset, split=split, meta=["meta_col"])
    dq_evaluate = watch(model)
    for i in range(0, ds_len, batch_size):
        batch = dataset[i : i + batch_size]
        dq_evaluate(batch, split)

    ThreadPoolManager.wait_for_threads()
    dq.get_data_logger().upload()
    train_data = vaex.open(f"{TEST_PATH}/training/0/data/data.hdf5")
    assert train_data["meta_col"].unique() == ["meta"]
    dq.finish()


@patch.object(ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
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
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end_to_end(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    global dataset
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)

    # 🔭🌕 Galileo logging
    from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitModel, SetFitTrainer

    import dataquality as dq
    from dataquality.integrations.setfit import watch

    model_id = "sentence-transformers/paraphrase-mpnet-base-v2"
    model = SetFitModel.from_pretrained(model_id, use_differentiable_head=True)
    column_mapping = {"text": "text", "label": "label"}
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
        # project_name=project_name, run_name=run_name,
        batch_size=512,  # Speed up prediction
        # 🔭🌕 Set finish to False to add test
        finish=False,
    )
    trainer.freeze()
    trainer.train()
    trainer.unfreeze(keep_body_frozen=True)
    dq.set_epoch(1)
    trainer.train()

    # model.save_pretrained("./trained_model")

    # trainer.evaluate()
    # unwatch(trainer)
    dq_evaluate = watch(model, batch_size=512)
    dq_evaluate(
        dataset,
        split="test",
        column_mapping=column_mapping
        # for inference set the split to inference
        # and pass an inference_name="inference_run_1"
    )
    dq.finish()
