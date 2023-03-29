import os
from functools import partial
from typing import Callable
from unittest.mock import ANY, MagicMock, patch

import pytest
from tenacity import RetryError

import dataquality
import dataquality as dq
from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.core.auth import GALILEO_AUTH_METHOD
from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID
from tests.exceptions import LoginInvoked
from tests.test_utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    MockResponse,
    mocked_create_project_run,
    mocked_get_project_run,
    mocked_login,
    mocked_login_requests,
)


@pytest.mark.parametrize(
    "task_type",
    TaskType.get_valid_tasks(),
)
@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    task_type: str,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    dataquality.init(task_type=task_type)
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with(ANY)
    mock_create_project.assert_called_once_with(ANY, is_public=True)
    mock_get_project_run_by_name.assert_called_once_with(ANY, ANY)
    # assert run is created with right task type
    mock_create_run.assert_called_once_with(ANY, ANY, task_type=TaskType[task_type])


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "get_project_run_by_name")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_reset_logger_config(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_get_project_run_by_name.return_value = {"id": DEFAULT_RUN_ID}

    dataquality.init(task_type="text_classification")
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID
    dataquality.set_labels_for_run(["a", "b", "c", "d"])
    dataquality.init(task_type="text_classification")
    assert not dataquality.get_data_logger().logger_config.labels


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_private_project(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    dataquality.init(task_type="text_classification", is_public=False)
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with(ANY)
    mock_create_project.assert_called_once_with(ANY, is_public=False)
    mock_get_project_run_by_name.assert_called_once_with(ANY, ANY)
    mock_create_run.assert_called_once_with(
        ANY, ANY, task_type=TaskType["text_classification"]
    )


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init passing in an existing project"""
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(task_type="text_classification", project_name=EXISTING_PROJECT)
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with(EXISTING_PROJECT)
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_called_once_with(EXISTING_PROJECT, ANY)
    mock_create_run.assert_called_once_with(
        EXISTING_PROJECT, ANY, task_type=TaskType["text_classification"]
    )


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init passing in a new project"""
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)

    dataquality.init(task_type="text_classification", project_name="new_proj")
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with("new_proj")
    mock_create_project.assert_called_once_with("new_proj", is_public=True)
    mock_get_project_run_by_name.assert_called_once_with("new_proj", ANY)
    mock_create_run.assert_called_once_with(
        "new_proj", ANY, task_type=TaskType["text_classification"]
    )


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_new_run(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with an existing project but a new run"""
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name="new_run",
    )
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with(EXISTING_PROJECT)
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_called_once_with(EXISTING_PROJECT, ANY)
    mock_create_run.assert_called_once_with(
        EXISTING_PROJECT, ANY, task_type=TaskType["text_classification"]
    )


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name")
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_existing_project_existing_run(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with an existing project and existing run"""
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_get_project_run_by_name.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification",
        project_name=EXISTING_PROJECT,
        run_name=EXISTING_RUN,
    )
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with(EXISTING_PROJECT)
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_called_once_with(EXISTING_PROJECT, EXISTING_RUN)
    mock_create_run.assert_not_called()


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_new_project_run(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init with a new project and new run"""
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(
        task_type="text_classification", project_name="new_proj", run_name="new_run"
    )
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    mock_get_project_by_name.assert_called_once_with("new_proj")
    mock_create_project.assert_called_once_with("new_proj", is_public=True)
    mock_get_project_run_by_name.assert_called_once_with("new_proj", "new_run")
    mock_create_run.assert_called_once_with(
        "new_proj", "new_run", task_type=TaskType["text_classification"]
    )


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name")
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_only_run(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests calling init only passing in a run"""
    set_test_config(current_project_id=None, current_run_id=None)
    dataquality.init(task_type="text_classification", run_name="a_run")
    assert not config.current_run_id
    assert not config.current_project_id

    mock_get_project_by_name.assert_not_called()
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_not_called()
    mock_create_run.assert_not_called()


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name")
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_project_name_collision(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests init with a project name created by another user at same time

    If two users try to create a project with the same name at the same time,
    the second user will return an empty project from `get_project` and then
    run into an error when trying to `create_project` with the same name.

    Using the tenacity `retry` decorator, we should be able to handle this
    gracefully.

    Here we test that despite 2 failed attempts to create a project with the same
    name will result in a successful project creation.
    """
    mock_get_project_by_name.side_effect = [
        GalileoException,
        GalileoException,
        {"id": DEFAULT_PROJECT_ID},
    ]
    mock_get_project_run_by_name.return_value = {"id": DEFAULT_RUN_ID}
    dataquality.init(
        task_type="text_classification", project_name="race-condition-proj"
    )
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID

    assert mock_get_project_by_name.call_count == 3
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_called_once_with("race-condition-proj", ANY)
    mock_create_run.assert_not_called()


@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name")
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_project_name_collision_fails(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    """Tests that the retry decorator will stop after 5 failed attempts"""
    set_test_config(current_project_id=None, current_run_id=None)
    mock_get_project_by_name.side_effect = GalileoException
    with pytest.raises(RetryError):
        dataquality.init(
            task_type="text_classification", project_name="race-condition-proj"
        )

    assert config.current_run_id is None
    assert config.current_project_id is None

    assert mock_get_project_by_name.call_count == 5
    mock_create_project.assert_not_called()
    mock_get_project_run_by_name.assert_not_called()
    mock_create_run.assert_not_called()


@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_failed_login(mock_login: MagicMock, set_test_config: Callable) -> None:
    set_test_config(token=None)
    with pytest.raises(LoginInvoked):
        # When no token is passed in we should call login
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_successful_login(
    mock_login: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(token=None, current_project_id=None, current_run_id=None)
    # When no token is passed in we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()

    # We also test the remaining init flow
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID


@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_expired_token_login(
    mock_login: MagicMock, mock_current_user: MagicMock, set_test_config: Callable
) -> None:
    # When a token is passed in but user auth fails we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dataquality.core.init.ApiClient, "get_current_user", side_effect=GalileoException
)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_expired_token_login_full(
    mock_login: MagicMock,
    mock_current_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    # When a token is passed in but user auth fails we should call login
    dataquality.init(task_type="text_classification")

    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID


@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=LoginInvoked)
def test_init_invalid_user_login(
    mock_login: MagicMock, mock_valid_user: MagicMock
) -> None:
    # When current user is not valid we should call login
    with pytest.raises(LoginInvoked):
        dataquality.init(task_type="text_classification")
        mock_login.assert_called_once()


@patch.object(ApiClient, "get_project_by_name", return_value={})
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=False)
@patch("dataquality.core.init.login", side_effect=mocked_login)
def test_init_invalid_user_login_full(
    mock_login: MagicMock,
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
) -> None:
    mock_create_project.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}

    set_test_config(current_project_id=None, current_run_id=None)
    # When current user is not valid we should call login
    dataquality.init(task_type="text_classification")
    mock_login.assert_called_once()
    # We also test the remaining init flow
    assert config.current_run_id == DEFAULT_RUN_ID
    assert config.current_project_id == DEFAULT_PROJECT_ID


@patch("dataquality.core.init._check_dq_version")
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
def test_init_bad_task(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
) -> None:
    with pytest.raises(GalileoException) as e:
        dataquality.init(task_type="fake_task_type")

    assert "Task type fake_task_type not valid" in str(e.value)


@patch("dataquality.core.auth.login")
def test_reconfigure_sets_env_vars(mock_login: MagicMock) -> None:
    os.environ["DQ_TELEMETRICS"] = "False"
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.fakecompany.io"
    dataquality.configure()
    assert dataquality.config.api_url == config.api_url == "https://api.fakecompany.io"
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.newfake.de"
    dataquality.configure()
    assert dataquality.config.api_url == config.api_url == "https://api.newfake.de"

    assert mock_login.call_count == 2


@patch("requests.post", side_effect=mocked_login_requests)
@patch("requests.get", side_effect=mocked_login_requests)
def test_reconfigure_resets_user_token(
    mock_get_request: MagicMock,
    mock_post_request: MagicMock,
    set_test_config: Callable,
) -> None:
    set_test_config(token="old_token")

    os.environ[GALILEO_AUTH_METHOD] = "email"
    os.environ["GALILEO_USERNAME"] = "user"
    os.environ["GALILEO_PASSWORD"] = "password"
    dataquality.configure()
    assert all([config.token == "mock_token", config.token != "old_token"])


@patch("dataquality.core.auth.login", side_effect=mocked_login)
def test_reconfigure_resets_user_token_login_mocked(
    mock_login: MagicMock, set_test_config: Callable
) -> None:
    set_test_config(token="old_token")
    dataquality.configure()
    assert all([config.token == "mock_token", config.token != "old_token"])
    mock_login.assert_called_once()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
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
@patch.object(dataquality.core.init.ApiClient, "valid_current_user", return_value=True)
@pytest.mark.parametrize(
    "run_name,exc",
    [
        ("my-run-#1", True),
        ("$run!$", True),
        ("a-bad-run##", True),
        ("A good run1", False),
        ("a-good-run-123", False),
        ("MY Run_1-2022", False),
    ],
)
def test_bad_names(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    run_name: str,
    exc: bool,
    set_test_config: Callable,
):
    init = partial(
        dataquality.init, task_type="text_classification", project_name="proj"
    )
    if exc:
        with pytest.raises(GalileoException):
            init(run_name=run_name)
    else:
        init(run_name=run_name)


@patch("requests.get")
def test_init_incompatible_dq_version(mock_get: MagicMock) -> None:
    mock_get.return_value = MockResponse(
        json_data={"minimum_dq_version": "100.0.0"}, status_code=200
    )
    with pytest.raises(GalileoException):
        dataquality.init(task_type="text_classification")


@patch("dataquality.login")
def test_set_console_url_picks_env_vars(mock_login: MagicMock) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.fakecompany.io"
    dataquality.set_console_url()
    assert dataquality.config.api_url == config.api_url == "https://api.fakecompany.io"
    mock_login.assert_not_called()


@patch("dataquality.login")
def test_set_console_url_overwrites_with_param(mock_login: MagicMock) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.newfake.de"
    dataquality.set_console_url(console_url="https://console.newfake2.com")
    assert dataquality.config.api_url == config.api_url == "https://api.newfake2.com"
    mock_login.assert_not_called()
