import time
from typing import Callable, Dict, List
from unittest import mock
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_delete_project_not_found,
    mocked_delete_project_run,
    mocked_get_project_run,
    mocked_missing_project_name,
    mocked_missing_run,
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


@mock.patch("requests.get", side_effect=mocked_missing_run)
def test_delete_run_by_name_missing_run(
    mock_get_run: MagicMock, set_test_config: Callable
) -> None:
    """Base case: Tests creating a new project and run"""
    with pytest.raises(GalileoException):
        api_client.delete_run_by_name("some_proj", "some_run")


@pytest.mark.parametrize(
    "api_url",
    ["http://localhost:8088", "http://127.0.0.1:8088"],
)
def test_get_run_status_localhost_fails(
    set_test_config: Callable, api_url: str
) -> None:
    """Raises error if api url is set to localhost"""
    set_test_config(default_api_url=api_url)
    with pytest.raises(GalileoException):
        api_client.get_run_status()


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
    assert status["status"] == "finished"
    mock_make_request.assert_called_once_with(
        "get", f"https://api.fake.com/projects/{project_id}/runs/{run_id}/jobs/status"
    )


@mock.patch.object(ApiClient, "get_project_run_by_name")
@mock.patch.object(ApiClient, "make_request")
def test_get_run_status_project_specified_run(
    mock_make_request: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    set_test_config: Callable,
    statuses_response: Dict[str, List],
) -> None:
    # TODO
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
    assert status["status"] == "finished"
    assert config_project_id != project_id
    assert config_run_id != run_id
    mock_make_request.assert_called_once_with(
        "get", f"https://api.fake.com/projects/{project_id}/runs/{run_id}/jobs/status"
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
    side_effect=[{"status": "started"}, {"status": "finished"}],
)
def test_wait_for_run_started_finished(mock_get_run_status: MagicMock) -> None:
    """Happy path: Returns after transitioning from started to finished"""
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
