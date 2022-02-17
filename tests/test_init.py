import os
from unittest.mock import MagicMock, patch

import pytest

import dataquality
from dataquality import config
from dataquality.exceptions import GalileoException
from tests.exceptions import LoginInvoked
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_create_project_run,
    mocked_get_project_run,
    mocked_login,
    mocked_missing_project_run,
    mocked_missing_run,
)


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Base case: Tests creating a new project and run"""
    dataquality.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Tests calling init passing in an existing project"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", project_name=EXISTING_PROJECT)
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Tests calling init passing in a new project"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", project_name="new_proj")
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_new_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Tests calling init with an existing project but a new run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name="new_run",
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_get_project_run)
@patch("requests.post", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Tests calling init with an existing project and existing run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name=EXISTING_RUN,
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch("requests.post", side_effect=mocked_create_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    """Tests calling init with a new project and new run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification", project_name="new_proj", run_name="new_run"
    )
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_missing_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_only_run(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
) -> None:
    """Tests calling init only passing in a run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", run_name="a_run")
    assert not config.current_run_id
    assert not config.current_project_id


@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_no_token_login(mock_login: MagicMock) -> None:
    config.token = None
    with pytest.raises(LoginInvoked):
        # When no token is passed in we should call login
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_no_token_login_full(
    mock_login: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    config.token = None
    # When no token is passed in we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_expired_token_login(
    mock_login: MagicMock, mock_current_user: MagicMock
) -> None:
    config.token = "sometoken"
    # When a token is passed in but user auth fails we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_expired_token_login_full(
    mock_login: MagicMock,
    mock_current_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    config.token = "sometoken"
    # When a token is passed in but user auth fails we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_invalid_user_login(
    mock_login: MagicMock, mock_valid_user: MagicMock
) -> None:
    # When current user is not valid we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_invalid_user_login_full(
    mock_login: MagicMock,
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    # When current user is not valid we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id
    assert config.current_project_id


@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_bad_task(
    mock_valid_user: MagicMock, mock_requests_get: MagicMock
) -> None:
    with pytest.raises(GalileoException):
        dataquality.init(task_type="not_text_classification")


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_reconfigure(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    dataquality.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id
    old_url = config.minio_url
    test_url = f"{old_url}_TEST"
    os.environ["GALILEO_MINIO_URL"] = test_url
    dataquality.configure()
    assert dataquality.config.minio_url == config.minio_url == test_url
    os.environ["GALILEO_MINIO_URL"] = old_url
    dataquality.configure()
    assert dataquality.config.minio_url == config.minio_url == old_url
