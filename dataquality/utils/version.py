import requests
from setuptools.version import pkg_resources  # type: ignore

from dataquality import __version__ as dq_client_version
from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.schemas.route import Route


def _version_check() -> None:
    """_version_check.

    Asserts that the dataquality python client and the api have
    matching major versions.

    https://semver.org/#summary.

    Returns:
        None
    """
    client_semver = _parse_version(_get_client_version())
    server_semver = _parse_version(_get_api_version())
    try:
        assert _major_version(client_semver) == _major_version(server_semver)
    except AssertionError:
        raise GalileoException(
            "Major mismatch between client, "
            f"{client_semver}, and server {server_semver}."
        )


def _major_version(v: pkg_resources.extern.packaging.version.Version) -> str:
    if hasattr(v, "major"):
        return str(v.major)
    else:
        return str(v.base_version).split(".")[0]


def _parse_version(version: str) -> pkg_resources.extern.packaging.version.Version:
    return pkg_resources.parse_version(version)


def _get_client_version() -> str:
    return dq_client_version


def _get_api_version() -> str:
    response = requests.get(f"{config.api_url}/{Route.healthcheck}")
    response_body = response.json()
    return response_body["api_version"]
