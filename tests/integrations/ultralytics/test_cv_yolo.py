import os
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import vaex

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.ultralytics import watch
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.ultralytics import temporary_cfg_for_val
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION, TEST_PATH


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
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
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
def test_end2end_yolov8(
    mock_dq_healthcheck: MagicMock,
    mock_valid_user: MagicMock,
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
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    from ultralytics import YOLO

    dq.init(TaskType.object_detection)
    # TODO: Make this path better using the current file location
    ds_path = "tests/integrations/ultralytics/coco.yaml"
    model = YOLO("./tests/integrations/ultralytics/yolov8n.pt")
    dq.set_split(Split.validation)
    watch(model)
    for split in [Split.training, Split.validation]:
        dq.set_split(Split.validation)
        tmp_cfg_path = temporary_cfg_for_val(ds_path, split)
        if not tmp_cfg_path:
            continue
        dq.set_epoch(0)
        dq.set_split(split)
        model = YOLO("./tests/integrations/ultralytics/yolov8n.pt")
        watch(model)
        model.val(data=tmp_cfg_path)
        os.remove(tmp_cfg_path)

    ThreadPoolManager.wait_for_threads()
    image_df = vaex.open(f"{LOCATION}/input_data/validation/data_0.arrow")
    box_df = vaex.open(f"{LOCATION}/validation/0/*.hdf5")
    # Need to make sure that all image_ids in the box df exist in image df.
    # It's possible image_df has more, when an image has no GT and no pred boxes
    assert set(box_df["image_id"].unique()).issubset(image_df["id"].tolist())
    dq.get_data_logger().upload()

    # TODO: @franz need to get training data logged
    for split in [Split.training, Split.validation]:
        dq.set_split(split)
        vaex.open(f"{TEST_PATH}/{split}/0/data/data.hdf5")
        prob_df = vaex.open(f"{TEST_PATH}/{split}/0/prob/prob.hdf5")
        emb_df = vaex.open(f"{TEST_PATH}/{split}/0/emb/emb.hdf5")
        assert sorted(emb_df.get_column_names()) == ["emb_pca", "id"]
        prob_cols = [
            "bbox",
            "epoch",
            "gold",
            "image_id",
            "is_gold",
            "is_pred",
            "prob",
            "split",
        ]
        assert sorted(prob_df.get_column_names()) == sorted(prob_cols)
        assert prob_df.bbox.shape == (len(prob_df), 4)
        assert prob_df.bbox.dtype == "float32"
        assert prob_df.prob.shape == (len(prob_df), 80)
        assert prob_df.bbox.dtype == "float32"
        # data_cols = ["id", "image", "split", "data_schema_version"]
        # assert sorted(data_df.get_column_names()) == sorted(data_cols)
    dq.finish()
