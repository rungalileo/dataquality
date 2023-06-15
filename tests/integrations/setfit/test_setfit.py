from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import vaex
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer

import dataquality as dq
from dataquality.integrations.setfit import auto, watch
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCAL_MODEL_PATH, TestSessionVariables


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_setfit_watch(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
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
    dq_evaluate = watch(model, validate_before_training=False)
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
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_log_dataset(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    dataset = Dataset.from_dict(
        {"text": ["hello", "world", "foo", "bar"], "label": [0, 1] * 2}
    )
    model = SetFitModel.from_pretrained(LOCAL_MODEL_PATH)
    set_test_config(
        task_type="text_classification",
        project_name="test_project",
        run_name="test_run",
    )

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
    dq_evaluate = watch(model, validate_before_training=False)
    for i in range(0, ds_len, batch_size):
        batch = dataset[i : i + batch_size]
        dq_evaluate(batch, split)

    ThreadPoolManager.wait_for_threads()
    dq.get_data_logger().upload()
    train_data = vaex.open(f"{test_session_vars.TEST_PATH}/training/0/data/data.hdf5")
    assert train_data["meta_col"].unique() == ["meta"]
    dq.finish()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_setfit_trainer(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
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
        project_name="project_name",
        run_name="run_name",
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

    # model.save_pretrained("./trained_model")

    # trainer.evaluate()
    # unwatch(trainer)
    dq_evaluate = watch(model, batch_size=512, validate_before_training=False)
    dq_evaluate(
        dataset,
        split="test",
        column_mapping=column_mapping
        # for inference set the split to inference
        # and pass an inference_name="inference_run_1"
    )
    dq.finish()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_setfit_auto(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    dataset = Dataset.from_dict(
        {"text": ["hello", "world", "foo", "bar"], "label": [0, 1] * 2}
    )
    set_test_config(
        task_type="text_classification",
        project_name="test_project",
        run_name="test_run",
    )

    column_mapping = {"text": "text", "label": "label"}

    labels = ["nocat", "cat"]
    auto(
        train_data=dataset,
        val_data=dataset,
        test_data=dataset,
        inference_data={"eval": dataset},
        run_name="labels",
        training_args={"num_epochs": 2},
        labels=labels,
        column_mapping=column_mapping,
    )
