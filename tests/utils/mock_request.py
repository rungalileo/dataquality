from typing import Dict, List, Union
from uuid import uuid4

import dataquality
from dataquality import __version__

config = dataquality.config

EXISTING_PROJECT = "existing_proj"
EXISTING_RUN = "existing_run"
TMP_CREATE_NEW_PROJ_RUN = None


class MockResponse:
    def __init__(self, json_data: Union[Dict, List], status_code: int) -> None:
        self.json_data = json_data
        self.status_code = status_code
        if status_code in (200, 204):
            self.ok = True
        else:
            self.ok = False
            self.text = json_data

    def json(self) -> Union[Dict, List]:
        return self.json_data


def mocked_healthcheck_request(request_url: str) -> MockResponse:
    if request_url.endswith("healthcheck"):
        return MockResponse({"api_version": __version__}, 200)

    return MockResponse({}, 200)


def mocked_healthcheck_request_new_api_version(request_url: str) -> MockResponse:
    if request_url.endswith("healthcheck"):
        return MockResponse({"api_version": "100.0.0"}, 200)
    return MockResponse({}, 200)


def mocked_login_requests(
    request_url: str,
    json: Dict = {},
    params: Dict = {},
    headers: Dict = {},
    data: Dict = {},
) -> MockResponse:
    if request_url.endswith("login"):
        return MockResponse(
            {
                "access_token": "mock_token",
                "minio_user_secret_access_key": "mock_secret_key",
            },
            200,
        )
    if request_url.endswith("current_user"):
        return MockResponse({"user": "user"}, 200)
    return MockResponse({}, 200)


def mocked_failed_login_requests(
    request_url: str,
    json: Dict = {},
    params: Dict = {},
    headers: Dict = {},
    data: Dict = {},
) -> MockResponse:
    if request_url.endswith("login"):
        return MockResponse({"detail": "Invalid credentials"}, 404)
    return MockResponse({}, 404)


def mocked_get_project_run(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    res = [
        {"id": uuid4(), "name": EXISTING_PROJECT, "project_id": uuid4()},
        {"id": uuid4(), "name": EXISTING_RUN, "project_id": uuid4()},
        {"id": uuid4(), "name": TMP_CREATE_NEW_PROJ_RUN, "project_id": uuid4()},
    ]
    return MockResponse(res, 200)


def mocked_create_project_run(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    global TMP_CREATE_NEW_PROJ_RUN
    TMP_CREATE_NEW_PROJ_RUN = json["name"]
    res = {"id": uuid4(), "name": TMP_CREATE_NEW_PROJ_RUN}
    return MockResponse(res, 200)


def mocked_missing_run(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    # Run does not exist
    if "run_name" in request_url:
        return MockResponse([], 204)
    # Project does exist
    else:
        res = {"id": uuid4(), "name": EXISTING_PROJECT}
        return MockResponse([res], 200)


def mocked_missing_project_run(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    return MockResponse([{"id": uuid4(), "name": TMP_CREATE_NEW_PROJ_RUN}], 200)


def mocked_missing_project_name(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    return MockResponse([], 200)


def mocked_delete_project_run(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    return MockResponse([{"id": uuid4(), "name": TMP_CREATE_NEW_PROJ_RUN}], 200)


def mocked_delete_project_not_found(
    request_url: str, json: Dict, params: Dict, headers: Dict, data: Dict
) -> MockResponse:
    if request_url.endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    return MockResponse({"detail": "project not found"}, 404)


def mocked_login() -> None:
    config.token = "mock_token"
