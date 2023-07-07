from unittest import mock
from unittest.mock import MagicMock

from dataquality import __version__
from dataquality.utils import version
from tests.test_utils.mock_request import MockResponse, mocked_healthcheck_request


def test_version() -> None:
    assert __version__ is not None


def test_get_client_version() -> None:
    assert version._get_client_version() == __version__


@mock.patch("requests.get", side_effect=mocked_healthcheck_request)
def test_get_api_version(mock_get_api_version: MagicMock) -> None:
    assert version._get_api_version() == __version__


@mock.patch("requests.get", side_effect=mocked_healthcheck_request)
def test_version_check_pass(mock_get_healthcheck: MagicMock) -> None:
    version.version_check()


@mock.patch(
    "requests.get",
    return_value=MockResponse({"api_version": __version__.replace("v", "")}, 200),
)
def test_version_check_pass_without_v(mock_get_healthcheck: MagicMock) -> None:
    version.version_check()
