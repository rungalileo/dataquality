from unittest import mock
from uuid import uuid4

import pytest

from dataquality import config
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
def test_delete_project(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    api_client.delete_project(uuid4())


@mock.patch("requests.get", side_effect=mocked_delete_project_not_found)
@mock.patch("requests.delete", side_effect=mocked_delete_project_not_found)
def test_delete_project_not_found(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    with pytest.raises(GalileoException):
        api_client.delete_project(uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_project_by_name(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    api_client.delete_project_by_name(EXISTING_PROJECT)


@mock.patch("requests.get", side_effect=mocked_missing_project_name)
def test_delete_project_by_name_not_found(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    with pytest.raises(GalileoException):
        api_client.delete_project_by_name("some_proj")


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_run(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    api_client.delete_run(uuid4(), uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_not_found)
def test_delete_run_missing_run(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    with pytest.raises(GalileoException):
        api_client.delete_run(uuid4(), uuid4())


@mock.patch("requests.get", side_effect=mocked_get_project_run)
@mock.patch("requests.delete", side_effect=mocked_delete_project_run)
def test_delete_run_by_name(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    api_client.delete_run_by_name(EXISTING_PROJECT, EXISTING_RUN)


@mock.patch("requests.get", side_effect=mocked_missing_run)
def test_delete_run_by_name_missing_run(*args) -> None:
    """Base case: Tests creating a new project and run"""
    config.token = "sometoken"
    with pytest.raises(GalileoException):
        api_client.delete_run_by_name("some_proj", "some_run")
