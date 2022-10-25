import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union

from dataquality.utils.ampli import AmpliMetric
from dataquality.utils.profiler import (
    _installed_modules,
    change_function,
    exception_from_error,
    get_device_info,
    parse_exception,
    parse_exception_ipython,
)


class Borg:
    # We use a borg pattern to share state across all instances of this class.
    # Due to submitting some errors in a thread,
    # we want to share the thread pool executor
    _shared_state: Dict[str, str] = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state


class Analytics(Borg):
    """Analytics is used to track errors and logs in the background"""

    def __init__(self, ApiClient: Any, config: Any) -> None:
        """To initialize the Analytics class you need to pass in an ApiClient and the dq config.
        :param ApiClient: The ApiClient class
        :param config: The dq config
        """

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
        """This function is used to initialize the Exception hook.
        We use a different hook for ipython then normal python.
        """
        try:
            from IPython import get_ipython

            ip = get_ipython()
            ip.set_custom_exc((Exception,), self.ipython_exception_handler)
        except Exception:
            new_hook = change_function(
                sys.excepthook, self.handle_exception, AmpliMetric.dq_general_exception
            )

            sys.excepthook = new_hook

    def _setup_user(self) -> Dict:
        """This function is used to setup the user information.
        This includes all installed packages.
        """
        profile: Dict[str, Any] = {"uuid": str(hex(uuid.getnode()))}
        try:
            profile["packages"] = _installed_modules()
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
        """This function is used to handle exceptions in ipython."""

        self.track_exception_ipython(etype, evalue, tb, AmpliMetric.dq_galileo_warning)
        # We need to call the default ipython exception handler to raise the error
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    def track_exception_ipython(
        self,
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
        scope: AmpliMetric = AmpliMetric.dq_log_batch_error,
    ) -> None:
        """We parse the current environment and send the error to the api."""
        data = parse_exception_ipython(etype, evalue, tb)
        self.last_error = data
        self.track_dq_error(data, scope)

    def handle_exception(
        self,
        etype: Optional[Type[BaseException]],
        evalue: Optional[BaseException],
        tb: Optional[TracebackType],
        scope: AmpliMetric = AmpliMetric.dq_log_batch_error,
    ) -> None:
        """This function is used to handle exceptions in python."""
        if etype is not None and evalue is not None and tb is not None:
            try:
                data = parse_exception(etype, evalue, tb)
                self.last_error = data
                self.api_client.send_analytics(
                    self.config.current_project_id,
                    self.config.current_run_id,
                    self.config.task_type,
                    data,
                    scope,
                )
            except Exception:
                pass

    def log(self, error_message: str, run_task_type: str = "Unknown") -> None:
        """This function is used to log a message to the api."""
        try:
            self.api_client.send_analytics(
                self.config.current_project_id,
                self.config.current_run_id,
                run_task_type,
                error_message,
            )
        except Exception:
            pass

    def capture_exception(
        self,
        error: Union[Exception, None],
        scope: AmpliMetric = AmpliMetric.dq_general_exception,
    ) -> None:
        """This function is used to take an exception that is passed as an argument."""
        if error is not None:
            exc_info = exception_from_error(error)
        else:
            exc_info = sys.exc_info()
        self.handle_exception(*exc_info, scope)

    def log_import(self, module: str) -> None:
        """This function is used to log an import of a module."""
        data = get_device_info()
        data["method"] = "import"
        data["value"] = module
        data["arguments"] = ""
        self.last_log = data
        self.track_dq_error(data, AmpliMetric.dq_import)

    def log_function(self, function: str) -> None:
        """This function is used to log an functional call"""
        data = get_device_info()
        data["method"] = "function"
        data["value"] = function
        data["arguments"] = ""
        self.last_log = data
        self.track_dq_error(data, AmpliMetric.dq_function_call)

    def track_dq_error(self, data: Dict, scope: AmpliMetric) -> None:
        """This function is used to send the error to the api in a thread."""
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
        """This function is used to set the config post init."""
        self.config = config
