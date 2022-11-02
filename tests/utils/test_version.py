from unittest import mock
from unittest.mock import MagicMock

import pytest

from dataquality import __version__
from dataquality.exceptions import GalileoException
from dataquality.utils import version
from tests.test_utils.mock_request import (
    mocked_healthcheck_request,
    mocked_healthcheck_request_new_api_version,
)


def test_version() -> None:
    assert __version__ is not None


def test_get_client_version() -> None:
    assert version._get_client_version() == __version__


@mock.patch("requests.get", side_effect=mocked_healthcheck_request)
def test_get_api_version(mock_get_api_version: MagicMock) -> None:
    assert version._get_api_version() == __version__


def test_parse_version_alpha() -> None:
    assert str(version._parse_version("v0.0.1-alpha")) == "0.0.1a0"


def test_parse_version_beta() -> None:
    assert str(version._parse_version("v0.0.2-beta")) == "0.0.2b0"


@mock.patch("requests.get", side_effect=mocked_healthcheck_request_new_api_version)
def test_version_check_fail(mock_get_healthcheck: MagicMock) -> None:
    with pytest.raises(GalileoException):
        version._version_check()


@mock.patch("requests.get", side_effect=mocked_healthcheck_request)
def test_version_check_pass(mock_get_healthcheck: MagicMock) -> None:
    version._version_check()
    assert True
