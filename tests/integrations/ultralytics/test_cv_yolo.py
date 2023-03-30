from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.integrations.ultralytics import watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION


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
def test_end2end_yolov8(
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
    model = YOLO("./tests/integrations/ultralytics/yolov8n.pt")
    watch(model)
    preds = model.val(data="tests/integrations/ultralytics/coco.yaml")

    # df_lookup = df.reset_index().set_index("im_file")["index"].to_dict()

    ThreadPoolManager.wait_for_threads()

    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
    # validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")

    import vaex

    # print(os.popen(f"tree {LOCATION}").read())
    image_df = vaex.open(f"{LOCATION}/input_data/validation/data_0.arrow")
    box_df = vaex.open(f"{LOCATION}/validation/0/*.hdf5")
    # Need to make sure that all image_ids in the box df exist in image df.
    # It's possible image_df has more, when an image has no GT and no pred boxes
    assert set(box_df["image_id"].unique()).issubset(image_df["id"].tolist())
    # dq.finish()
