import requests
from setuptools.version import pkg_resources

from dataquality import __version__ as dq_client_version
from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.schemas.route import Route


def _version_check() -> None:
    """_version_check.

    Asserts that the dataquality python client has
    matching major and minor versions.

    Returns:
        None
    """
    client_semver = _parse_version(_get_client_version())
    server_semver = _parse_version(_get_api_version())
    try:
        major_match = client_semver.major == server_semver.major
        minor_match = client_semver.minor == server_semver.minor
        assert major_match and minor_match
    except AssertionError:
        raise GalileoException(
            "Major-minor mismatch between client, "
            f"{client_semver}, and server {server_semver}."
        )


def _parse_version(version: str) -> pkg_resources.extern.packaging.version.Version:
    return pkg_resources.parse_version(version)


def _get_client_version() -> str:
    return dq_client_version


def _get_api_version() -> str:
    response = requests.get(f"{config.api_url}/{Route.healthcheck}")
    response_body = response.json()
    return response_body["api_version"]
