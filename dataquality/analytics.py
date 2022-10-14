import sys
from platform import platform
from types import ModuleType
from typing import Any

from langcodes import Iterable


class Analytics:
    # Singleton analytics
    def __new__(cls) -> "Analytics":
        if not hasattr(cls, "instance"):
            cls.instance = super(Analytics, cls).__new__(cls)
        return cls.instance

    def __init__(self, config: Any = {}) -> None:
        from dataquality.schemas.request_type import RequestType
        from dataquality.clients.api import ApiClient

        api_client = ApiClient()
        self.api_client = api_client
        self.RequestType = RequestType
        self.config = config
        self.metadata = {
            "platform": self._get_platform(),
            "python_version": self._get_version(),
            "dataquality": config,
        }
        self._is_initialized = False

    def log(self, type: str, message: str) -> None:
        self.api_client.make_request(
            self.RequestType.POST,
            "/analytics/log",
            {"type": type, "message": message, "metadata": self.metadata},
        )

    def _list_sys_modules(self) -> set[str]:
        modules = set()
        for k in set(sys.modules):
            if "." in k:
                modules.add(k[: k.index(".")])
            else:
                modules.add(k)
        return modules

    def _list_global_imports(self) -> Iterable[str]:
        for val in globals().values():
            if isinstance(val, ModuleType):
                yield val.__name__

    def _get_platform(self) -> str:
        return platform()

    def _get_version(self) -> str:
        return sys.version
