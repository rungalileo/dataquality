from typing import Any, Dict
from uuid import uuid4

import dataquality

config = dataquality.config

EXISTING_PROJECT = "existing_proj"
EXISTING_RUN = "existing_run"


class MockResponse:
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


def mocked_login_requests(*args: Any, **kwargs: Dict[str, Any]):
    if args[0].endswith("login"):
        return MockResponse({"access_token": "mock_token"}, 200)

    if args[0].endswith("current_user"):
        return MockResponse({"user": "user"}, 200)

    return MockResponse(None, 200)


def mocked_failed_login_requests(*args: Any, **kwargs: Dict[str, Any]):
    if args[0].endswith("login"):
        return MockResponse("Invalid credentials", 404)

    return MockResponse(None, 404)


def mocked_get_project_run(*args: Any, **kwargs: Dict[Any, Any]):
    if args[0].endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    res = [
        {"id": uuid4(), "name": EXISTING_PROJECT},
        {"id": uuid4(), "name": EXISTING_RUN},
    ]
    return MockResponse(res, 200)


def mocked_create_project_run(*args: Any, **kwargs: Dict[Any, Any]):
    res = {"id": uuid4(), "name": "existing"}
    return MockResponse(res, 200)


def mocked_missing_run(*args: Any, **kwargs: Dict[Any, Any]):
    if args[0].endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    # Run does not exist
    if args[0].endswith("runs"):
        return MockResponse({}, 204)
    # Project does exist
    else:
        res = {"id": uuid4(), "name": EXISTING_PROJECT}
        return MockResponse([res], 200)


def mocked_missing_project_run(*args: Any, **kwargs: Dict[Any, Any]):
    if args[0].endswith("current_user"):
        return MockResponse({"id": "user"}, 200)
    return MockResponse({}, 204)
