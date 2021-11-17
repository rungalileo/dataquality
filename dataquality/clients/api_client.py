from typing import Dict

import requests
from requests.models import Response

from dataquality import config, GalileoException
from dataquality.schemas import Route
from dataquality.schemas.request_type import RequestType


class ApiClient:
    def __init__(self):
        self.api_url = config.api_url

    def make_request(self, request: RequestType, url: str, body: Dict = None, params: Dict = None, headers: Dict = None) -> Dict:
        req = RequestType.get_method(request.value)(url, json=body, params=params, headers=headers)
        if not req.ok:
            msg = (
                "Something didn't go quite right."
                " the api returned a non-ok status code"
                f" {req.status_code} with output: {req.text}"
            )
            raise GalileoException(msg)
        return req.json()
