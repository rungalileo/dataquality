import os
from typing import Callable
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import requests
from packaging import version

import dataquality
from dataquality import __version__ as dq_version
from dataquality.core._config import (
    CLOUD_URL,
    MINIMUM_API_VERSION,
    _check_dq_version,
    set_config,
    url_is_localhost,
)
from dataquality.exceptions import GalileoException
from tests.test_utils.mock_request import MockResponse


def test_console_url(set_test_config: Callable) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest.rungalileo.io"


def test_console_url_dash(set_test_config: Callable) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = "https://console-mytest.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url == "https://api-mytest.rungalileo.io"


@pytest.mark.parametrize(
    "console_url",
    ["http://localhost:3000", "http://127.0.0.1:3000"],
)
def test_console_url_local(set_test_config: Callable, console_url: str) -> None:
    os.environ["GALILEO_CONSOLE_URL"] = console_url
    cfg = set_config()
    assert cfg.api_url == "http://localhost:8088"


def test_bad_console_url(set_test_config: Callable) -> None:
    """If console is not in the console url, dont use it"""
    os.environ["GALILEO_CONSOLE_URL"] = "https://something.mytest2.rungalileo.io"
    cfg = set_config()
    assert cfg.api_url != "https://api.mytest2.rungalileo.io"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://console.domain.com", False),
        ("http://localhost:3000", True),
        ("http://127.0.0.1:8088", True),
    ],
)
def test_url_is_localhost(url: str, expected: bool) -> bool:
    assert url_is_localhost(url) is expected


@patch("dataquality.core._config.requests.get")
@pytest.mark.noautofixt
def test_validate_bad_config_url(get_mock: MagicMock) -> None:
    get_mock.side_effect = requests.ConnectionError("No connection!")
    bad_url = "https://console.mytest2.rungalileo.io/badurl"
    os.environ["GALILEO_CONSOLE_URL"] = bad_url
    with pytest.raises(GalileoException) as e:
        set_config()
    assert e.value.args[0].startswith(f"The provided console URL {bad_url} is invalid")


def test_handle_extra_slash_in_console(set_test_config: Callable) -> None:
    """Removes slashes in an otherwise valid console url"""
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.mytest2.rungalileo.io///"
    cfg = set_config()
    assert cfg.api_url == "https://api.mytest2.rungalileo.io"


@mock.patch("dataquality.core._config.os.path.exists", return_value=False)
@mock.patch.object(dataquality.core._config.Config, "update_file_config")
def test_config_defaults_cloud(
    mock_update_config: mock.MagicMock, set_test_config: Callable
) -> None:
    """Calling set_config without an environment variable should default to CLOUD_URL"""
    if os.getenv("GALILEO_CONSOLE_URL"):
        del os.environ["GALILEO_CONSOLE_URL"]
    if os.getenv("GALILEO_API_URL"):
        del os.environ["GALILEO_API_URL"]
    cfg = set_config()
    assert cfg.api_url == CLOUD_URL.replace("console", "api")
    mock_update_config.assert_called_once()


@pytest.mark.parametrize("min_dq_version", ["0.0.0", dq_version])
@pytest.mark.parametrize("api_version", ["100.0.0", MINIMUM_API_VERSION])
@patch("dataquality.core._config.requests.get")
def test_check_dq_version_happy_path(
    mock_get: mock.MagicMock, api_version: str, min_dq_version: str
) -> None:
    mock_get.return_value = MockResponse(
        json_data={"api_version": api_version, "minimum_dq_version": min_dq_version},
        status_code=200,
    )
    _check_dq_version()


@patch("dataquality.core._config.requests.get")
def test_check_dq_version_fails_major(mock_get: mock.MagicMock) -> None:
    # First test DQ version
    v = version.parse(dq_version)
    new_version = f"{v.major + 1}.{v.minor}.{v.micro}"
    mock_get.return_value = MockResponse(
        json_data={"minimum_dq_version": new_version}, status_code=200
    )
    with pytest.raises(GalileoException):
        _check_dq_version()

    # Then test API version
    v = version.parse(MINIMUM_API_VERSION)
    if v.major > 0:
        new_version = f"{v.major - 1}.{v.minor}.{v.micro}"
        mock_get.return_value = MockResponse(
            json_data={"minimum_dq_version": dq_version, "api_version": new_version},
            status_code=200,
        )
        with pytest.raises(GalileoException):
            _check_dq_version()


@patch("dataquality.core._config.requests.get")
def test_check_dq_version_fails_minor(mock_get: mock.MagicMock) -> None:
    # First test DQ version
    v = version.parse(dq_version)
    new_version = f"{v.major}.{v.minor + 1}.{v.micro}"
    mock_get.return_value = MockResponse(
        json_data={"minimum_dq_version": new_version}, status_code=200
    )
    with pytest.raises(GalileoException):
        _check_dq_version()

    # Then test API version
    v = version.parse(MINIMUM_API_VERSION)
    if v.minor > 0:
        new_version = f"{v.major}.{v.minor - 1}.{v.micro}"
        mock_get.return_value = MockResponse(
            json_data={"minimum_dq_version": dq_version, "api_version": new_version},
            status_code=200,
        )
        with pytest.raises(GalileoException):
            _check_dq_version()


@patch("dataquality.core._config.requests.get")
def test_check_dq_version_fails_micro(mock_get: mock.MagicMock) -> None:
    # First test DQ version
    v = version.parse(dq_version)
    new_version = f"{v.major}.{v.minor}.{v.micro + 1}"
    mock_get.return_value = MockResponse(
        json_data={"minimum_dq_version": new_version}, status_code=200
    )
    with pytest.raises(GalileoException):
        _check_dq_version()

    # Then test API version
    v = version.parse(MINIMUM_API_VERSION)
    if v.micro > 0:
        new_version = f"{v.major}.{v.minor}.{v.micro - 1}"
        mock_get.return_value = MockResponse(
            json_data={"minimum_dq_version": dq_version, "api_version": new_version},
            status_code=200,
        )
        with pytest.raises(GalileoException):
            _check_dq_version()


@patch("dataquality.core._config.requests.get")
def test_check_dq_version_404_silent_fail(mock_get: mock.MagicMock) -> None:
    mock_get.return_value = MockResponse(json_data={}, status_code=404)
    _check_dq_version()


@patch("dataquality.core._config.requests.get")
def test_check_dq_version_api_error(mock_get: mock.MagicMock) -> None:
    mock_get.return_value = MockResponse(json_data={}, status_code=422)
    with pytest.raises(GalileoException):
        _check_dq_version()
