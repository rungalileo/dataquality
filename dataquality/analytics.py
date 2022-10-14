import sys
from platform import platform
from types import ModuleType
from typing import Any

from langcodes import Iterable


class Analytics:
    def __init__(self, config: Any = {}) -> None:
        from dataquality.clients.api import ApiClient
        from dataquality.schemas.request_type import RequestType

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

    def log(self, error_message: str, run_task_type: str = "Unknown") -> None:
        self.api_client.send_analytics(
            self.config.current_project_id,
            self.config.current_project_id,
            error_message,
            run_task_type,
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

    def set_config(self, config: Any) -> None:
        self.config = config
