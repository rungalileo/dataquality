import requests

from dataquality import __version__ as dq_client_version
from dataquality.core._config import config
from dataquality.schemas.route import Route
from dataquality.utils.dq_logger import get_dq_logger


def version_check() -> None:
    """version_check.

    Asserts that the dataquality python client and the api have
    matching major versions.

    https://semver.org/#summary.

    Returns:
        None
    """
    client_semver = _get_client_version()
    server_semver = _get_api_version()
    if _major_version(client_semver) != _major_version(server_semver):
        get_dq_logger().warning(
            "Major version mismatched between client, "
            f"{client_semver}, and server {server_semver}."
        )


def _major_version(v: str) -> str:
    return str(v).replace("v", "").split(".")[0]


def _get_client_version() -> str:
    return dq_client_version


def _get_api_version() -> str:
    response = requests.get(f"{config.api_url}/{Route.healthcheck}")
    response_body = response.json()
    return response_body["api_version"]
