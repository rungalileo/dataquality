import time
from typing import Callable, Dict, List
from unittest import mock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import dataquality.clients.api
from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType
from dataquality.schemas.task_type import TaskType
from tests.test_utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    FAKE_NEW_RUN,
    MockResponse,
    mocked_delete_project_not_found,
    mocked_delete_project_run,
    mocked_get_project_run,
    mocked_missing_project_name,
)

api_client = ApiClient()


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_project(
    mock_delete_run: MagicMock, mock_get_run: MagicMock, set_test_config: Callable
) -> None:
    """Base case: Tests creating a new project and run"""
    api_client.delete_project(uuid4())


@mock.patch("requests.get", side_effect=mocked_delete_project_not_found)
@mock.patch("requests.delete", side_effect=mocked_delete_project_not_found)
def test_delete_project_not_found(
    mock_delete_run: MagicMock,
    mock_get_run: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    with pytest.raises(GalileoException):
        api_client.delete_project(uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_project_by_name(
    mock_delete_run: MagicMock,
    mock_get_run: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    api_client.delete_project_by_name(EXISTING_PROJECT)


@mock.patch("requests.get", side_effect=mocked_missing_project_name)
def test_delete_project_by_name_not_found(
    mock_get_run: MagicMock, set_test_config: Callable
) -> None:
    """Base case: Tests creating a new project and run"""
    with pytest.raises(GalileoException):
        api_client.delete_project_by_name("some_proj")


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_run(
    mock_delete_run: MagicMock,
    mock_get_run: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    api_client.delete_run(uuid4(), uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_not_found)
def test_delete_run_missing_run(
    mock_delete_run: MagicMock,
    mock_get_run: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    with pytest.raises(GalileoException):
        api_client.delete_run(uuid4(), uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_run_by_name(
    mock_delete_run: MagicMock,
    mock_get_run: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    api_client.delete_run_by_name(EXISTING_PROJECT, EXISTING_RUN)


@mock.patch("requests.get")
def test_delete_run_by_name_missing_run(
    mock_get_run: MagicMock, set_test_config: Callable
) -> None:
    """tests deleting a project/run with a run name that doesn't exist"""
    mock_get_run.side_effect = [
        MockResponse(json_data=[{"id": "uuid"}], status_code=200),
        MockResponse(json_data=[], status_code=200),
    ]
    with pytest.raises(GalileoException) as e:
        api_client.delete_run_by_name("some_proj", "some_run")

    assert str(e.value) == "No project/run found with name some_proj/some_run"


@pytest.mark.parametrize(
    "project_name, run_name",
    [("carrots", None), (None, "Rhubarb")],
)
def test_get_run_status_project_or_run_fails(project_name: str, run_name: str) -> None:
    """Raises error when exactly one of project_name and run_name is passed in"""
    with pytest.raises(GalileoException):
        api_client.get_run_status(project_name, run_name)


@mock.patch.object(ApiClient, "make_request")
def test_get_run_status_project_default_current_run(
    mock_make_request: MagicMock,
    set_test_config: Callable,
    statuses_response: Dict[str, List],
) -> None:
    """Happy path: Empty args return status for current run"""
    mock_make_request.return_value = statuses_response
    project_id = uuid4()
    run_id = uuid4()
    set_test_config(
        current_project_id=project_id,
        current_run_id=run_id,
        api_url="https://api.fake.com",
    )
    status = api_client.get_run_status()
    assert status["status"] == "completed"
    mock_make_request.assert_called_once_with(
        "get", f"https://api.fake.com/projects/{project_id}/runs/{run_id}/jobs/latest"
    )


@mock.patch.object(ApiClient, "get_project_run_by_name")
@mock.patch.object(ApiClient, "make_request")
def test_get_run_status_project_specified_run(
    mock_make_request: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    set_test_config: Callable,
    statuses_response: Dict[str, str],
) -> None:
    """Happy path: Specifying a project, run different from current run"""
    mock_make_request.return_value = statuses_response
    config_project_id = uuid4()
    config_run_id = uuid4()
    project_id = uuid4()
    run_id = uuid4()
    mock_get_project_run_by_name.return_value = {"project_id": project_id, "id": run_id}

    set_test_config(
        current_project_id=project_id,
        current_run_id=run_id,
        api_url="https://api.fake.com",
    )
    status = api_client.get_run_status("Carrot", "Rhubarb")
    assert status["status"] == "completed"
    assert config_project_id != project_id
    assert config_run_id != run_id
    mock_make_request.assert_called_once_with(
        "get", f"https://api.fake.com/projects/{project_id}/runs/{run_id}/jobs/latest"
    )


@mock.patch.object(ApiClient, "get_project_run_by_name", return_value={})
def test_get_run_status_project_nonexistent_run_fails(
    mock_get_project_run_by_name: MagicMock, set_test_config: Callable
) -> None:
    """Specifying a project, run that doesn't exist raises error"""
    set_test_config(api_url="https://api.fake.com")
    with pytest.raises(GalileoException):
        api_client.get_run_status("Carrot", "Fake-Run")


@mock.patch.object(
    ApiClient,
    "get_run_status",
    side_effect=[{"status": "unstarted"}, {"status": "completed"}],
)
def test_wait_for_run_unstarted_completed(mock_get_run_status: MagicMock) -> None:
    """Happy path: Returns after transitioning from unstarted to completed"""
    t_start = time.time()
    api_client.wait_for_run("some_proj", "some_run")
    assert mock_get_run_status.call_count == 2
    assert time.time() - t_start > 2


@mock.patch.object(ApiClient, "get_run_status", return_value={"status": "errored"})
def test_wait_for_run_errored(mock_get_run_status: MagicMock) -> None:
    """Waiting for run with errored status raises error"""
    with pytest.raises(GalileoException):
        api_client.wait_for_run("some_proj", "some_run")


@mock.patch.object(ApiClient, "get_run_status", return_value={"status": "unknown"})
def test_wait_for_run_unknown(mock_get_run_status: MagicMock) -> None:
    """Waiting for run with unknown status raises error"""
    with pytest.raises(GalileoException):
        api_client.wait_for_run("some_proj", "some_run")


@patch.object(
    ApiClient,
    "delete_run",
)
@patch.object(
    ApiClient,
    "create_run",
    return_value={"name": "my_run", "id": FAKE_NEW_RUN},
)
@patch.object(
    ApiClient,
    "get_project",
    return_value={"name": "my_project"},
)
@patch.object(
    ApiClient,
    "get_project_run",
    return_value={"name": "my_run", "task_type": 0},
)
def test_reset(
    mock_get_run: MagicMock,
    mock_get_project: MagicMock,
    mock_create: MagicMock,
    mock_delete: MagicMock,
    set_test_config: Callable,
) -> None:
    """
    Tests that reset run changes the run ID saved and updates the config
    """
    old_pid = config.current_project_id
    assert config.current_run_id != FAKE_NEW_RUN
    old_rid = config.current_run_id
    api_client.reset_run(old_pid, old_rid)

    mock_get_run.assert_called_once_with(old_pid, old_rid)
    mock_get_project.assert_called_once_with(old_pid)
    mock_create.assert_called_once_with(
        "my_project", "my_run", TaskType.text_classification
    )
    mock_delete.assert_called_once_with(old_pid, old_rid)

    assert config.current_run_id == FAKE_NEW_RUN


@patch.object(
    ApiClient,
    "get_run_status",
    side_effect=[{}, {"status": "completed", "timestamp": 1}],
)
def test_get_run_status_no_status(
    mock_get_status: MagicMock, set_test_config: Callable
) -> None:
    """Asserts that wait_for_run with an empty status doens't crash"""
    api_client.wait_for_run()
    assert mock_get_status.call_count == 2


@patch.object(dataquality.clients.api.requests, "post")
@patch.object(dataquality.clients.api.ApiClient, "get_task_type")
@patch.object(dataquality.clients.api.ApiClient, "_get_project_run_id")
def test_export_run_no_data(
    mock_get_run: MagicMock, mock_get_task_type: MagicMock, mock_post: MagicMock
) -> None:
    mock_get_run.return_value = uuid4(), uuid4()
    mock_get_task_type.return_value = TaskType.text_classification
    # In export_run we use requests.post as a context manager (with requests.post(...))
    # so we need to mock the `__enter__` return value
    mock_post.return_value.__enter__.return_value = MockResponse(
        status_code=200, json_data={}, headers={"Galileo-No-Data": "true"}
    )
    with pytest.raises(GalileoException) as e:
        api_client.export_run("project", "run", "training", "file.csv")
    assert str(e.value).startswith("It seems there is no data for this request")


@mock.patch.object(ApiClient, "make_request")
def test_notify_email(mock_make_request: MagicMock, set_test_config: Callable) -> None:
    api_client.notify_email({"foo": "bar"}, "template", ["foo@bar.com"])
    mock_make_request.assert_called_once_with(
        RequestType.POST,
        url="http://localhost:8088/notify/email",
        body={
            "data": {"foo": "bar"},
            "template": "template",
            "emails": ["foo@bar.com"],
        },
    )
