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


@mock.patch("requests.post", side_effect=mocked_create_project_run)
def test_init(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    dataquality.init()
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.post", side_effect=mocked_create_project_run)
@mock.patch("requests.get", side_effect=mocked_get_project_run)
def test_init_existing_project(*args) -> None:
    """Tests calling init passing in an existing project"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(project_name=EXISTING_PROJECT)
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_project_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
def test_init_new_project(*args) -> None:
    """Tests calling init passing in a new project"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(project_name="new_proj")
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
def test_init_existing_project_new_run(*args) -> None:
    """Tests calling init with an existing project but a new run"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(project_name=EXISTING_PROJECT, run_name="new_run")
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.post", side_effect=mocked_get_project_run)
def test_init_existing_project_run(*args) -> None:
    """Tests calling init with an existing project and existing run"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(project_name=EXISTING_PROJECT, run_name=EXISTING_RUN)
    assert config.current_run_id
    assert config.current_project_id


@mock.patch("requests.get", side_effect=mocked_missing_project_run)
@mock.patch("requests.post", side_effect=mocked_create_project_run)
def test_init_new_project_run(*args) -> None:
    """Tests calling init with a new project and new run"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(project_name="new_proj", run_name="new_run")
    assert config.current_run_id
    assert config.current_project_id


def test_init_only_run(*args) -> None:
    """Tests calling init only passing in a run"""
    config.token = "sometoken"
    config.current_project_id = config.current_run_id = None
    dataquality.init(run_name="a_run")
    assert not config.current_run_id
    assert not config.current_project_id


@mock.patch("requests.get", side_effect=mocked_get_project_run)
def test_init_no_login(*args) -> None:
    config.token = None
    with pytest.raises(GalileoException):
        dataquality.init()
    with pytest.raises(GalileoException):
        dataquality.init(project_name=EXISTING_PROJECT)
