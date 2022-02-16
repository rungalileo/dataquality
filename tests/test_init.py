import os
from unittest import mock

import pytest

import dataquality
from dataquality import config
from dataquality.exceptions import GalileoException
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_create_project_run,
    mocked_get_project_run,
    mocked_missing_project_run,
    mocked_missing_run,
)


def mocked_login() -> None:
    config.token = "sometoken"


class LoginInvoked(Exception):
    pass


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init(*args) -> None:
    """Base case: Tests creating a new project and run"""
    dataquality.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_existing_project(*args) -> None:
    """Tests calling init passing in an existing project"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", project_name=EXISTING_PROJECT)
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_project_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_new_project(*args) -> None:
    """Tests calling init passing in a new project"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", project_name="new_proj")
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_existing_project_new_run(*args) -> None:
    """Tests calling init with an existing project but a new run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name="new_run",
    )
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.post", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_existing_project_run(*args) -> None:
    """Tests calling init with an existing project and existing run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name=EXISTING_RUN,
    )
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_project_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_new_project_run(*args) -> None:
    """Tests calling init with a new project and new run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(
        task_type="text_classification", project_name="new_proj", run_name="new_run"
    )
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_only_run(*args) -> None:
    """Tests calling init only passing in a run"""
    config.current_project_id = config.current_run_id = None
    dataquality.init(task_type="text_classification", run_name="a_run")
    assert not config.current_run_id
    assert not config.current_project_id


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_no_token_succeeds(mock_login: mock.MagicMock, *args) -> None:
    config.token = None
    # When no token is passed in we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_no_token_succeeds_option_2(mock_login: mock.MagicMock, *args) -> None:
    config.token = None
    with pytest.raises(LoginInvoked):
        # When no token is passed in we should call login
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)  # noqa
@mock.patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_expired_token_succeeds(mock_login: mock.MagicMock, *args) -> None:
    config.token = "sometoken"
    # When a token is passed in but user auth fails we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=False
)  # noqa
@mock.patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_invalid_current_user_succeeds(mock_login: mock.MagicMock, *args) -> None:
    # When current user validation fails we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_init_bad_task(*args) -> None:
    with pytest.raises(GalileoException):
        dataquality.init(task_type="not_text_classification")


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch.object(
    dataquality.core.init.ApiClient, "valid_current_user", return_value=True
)  # noqa
def test_reconfigure(*args) -> None:
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
