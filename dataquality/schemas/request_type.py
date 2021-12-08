from collections.abc import Callable
from enum import Enum

import requests


class RequestType(str, Enum):
    GET = "get"
    POST = "post"
    PUT = "put"
    DELETE = "delete"

    @staticmethod
    def get_method(request: str) -> Callable:
        return getattr(requests, request)
