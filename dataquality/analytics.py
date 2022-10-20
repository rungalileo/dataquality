import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from dataquality.utils.ampli import AmpliMetric
from dataquality.utils.profiler import (
    _generate_installed_modules,
    change_function,
    exc_info_from_error,
    get_device_info,
    parse_exception,
    parse_exception_ipython,
)


class Borg:
    _shared_state: Dict[str, str] = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state


class Analytics(Borg):
    def __init__(self, ApiClient: Any, config: Any) -> None:
        super().__init__()
        # initiate the first instance with default state
        if not hasattr(self, "state"):

            self.api_caller = ThreadPoolExecutor(max_workers=5)
            self.api_client = ApiClient()
            self.config = config
            if not getattr(self, "_is_initializing", False):
                self.last_error: Dict = {}
                self.last_log: Dict = {}
                self.user: Dict = self._setup_user()
                self._is_initializing = True
                self._init()

        self._is_initialized = True

    def _init(self) -> None:
        try:
            from IPython import get_ipython

            ip = get_ipython()
            ip.set_custom_exc((Exception,), self.ipython_exception_handler)
            # Alternative
            # InteractiveShell.showtraceback
            # change_function(InteractiveShell.showtraceback)
        except Exception:
            new_hook = change_function(
                sys.excepthook, self.handle_exception, AmpliMetric.dq_general_exception
            )

            sys.excepthook = new_hook

    def _setup_user(self) -> Dict:
        profile: Dict[str, Any] = {"uuid": str(hex(uuid.getnode()))}
        try:
            packages = dict(_generate_installed_modules())
            profile["packages"] = packages
        except Exception:
            pass
        return profile

    def ipython_exception_handler(
        self,
        shell: Any,
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
        tb_offset: Any = None,
    ) -> None:
        from IPython import get_ipython

        lines = "\n".join([h[2] for h in get_ipython().history_manager.get_range()])
        for line in traceback.TracebackException(
            type(evalue), evalue, tb, limit=1024
        ).format(chain=True):
            lines += line
        self.handle_exception_ipython(
            etype, evalue, tb, lines, AmpliMetric.dq_galileo_warning
        )
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    def handle_exception_ipython(
        self,
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
        lines: str = "",
        scope: AmpliMetric = AmpliMetric.dq_log_batch_error,
    ) -> None:
        data = parse_exception_ipython(etype, evalue, tb, lines)
        self.last_error = data
        self.track_dq_error(data, scope)

    def handle_exception(
        self,
        etype: Optional[Type[BaseException]],
        evalue: Optional[BaseException],
        tb: Optional[TracebackType],
        scope: AmpliMetric = AmpliMetric.dq_log_batch_error,
    ) -> None:
        if etype is not None and evalue is not None and tb is not None:
            data = parse_exception(etype, evalue, tb)
            self.last_error = data
            self.api_client.send_analytics(
                self.config.current_project_id,
                self.config.current_run_id,
                self.config.task_type,
                data,
                scope,
            )

    def log(self, error_message: str, run_task_type: str = "Unknown") -> None:
        self.api_client.send_analytics(
            self.config.current_project_id,
            self.config.current_run_id,
            run_task_type,
            error_message,
        )

    def capture_exception(
        self,
        error: Union[Exception, None],
        scope: AmpliMetric = AmpliMetric.dq_general_exception,
    ) -> None:
        if error is not None:
            exc_info = exc_info_from_error(error)
        else:
            exc_info = sys.exc_info()
        self.handle_exception(*exc_info, scope)

    def log_import(self, module: str) -> None:
        data = get_device_info()
        data["method"] = "import"
        data["value"] = module
        data["arguments"] = ""
        self.last_log = data
        self.track_dq_error(data, AmpliMetric.dq_import)

    def log_function(self, function: str) -> None:
        data = get_device_info()
        data["method"] = "function"
        data["value"] = function
        data["arguments"] = ""
        self.last_log = data
        self.track_dq_error(data, AmpliMetric.dq_function_call)

    def track_dq_error(self, data: Dict, scope: AmpliMetric) -> None:
        # print("LOGGING DQ ERROR")
        try:
            self.api_caller.submit(
                self.api_client.send_analytics,
                self.config.current_project_id,
                self.config.current_run_id,
                self.config.task_type,
                data,
                scope,
            )
        except Exception as e:
            print(e)

    def set_config(self, config: Any) -> None:
        self.config = config
