from typing import Any, Dict

import dataquality

config = dataquality.config


def mocked_requests_post(*args: Any, **kwargs: Dict[str, Any]):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    if args[0].endswith("login"):
        return MockResponse({"access_token": "mock_token"}, 200)

    return MockResponse(None, 404)


def mocked_failed_requests_post(*args: Any, **kwargs: Dict[str, Any]):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    if args[0].endswith("login"):
        return MockResponse({"err": "err"}, 404)

    return MockResponse(None, 404)
