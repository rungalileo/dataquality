import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import BaseModel

from dataquality.clients.api import ApiClient
from dataquality.core._config import Config
from dataquality.utils.ampli import AmpliMetric
from dataquality.utils.profiler import (
    _installed_modules,
    change_function,
    exception_from_error,
    get_device_info,
    parse_exception,
    parse_exception_ipython,
)


class ProfileModel(BaseModel):
    """User profile"""

    packages: Optional[Dict[str, str]]
    uuid: Optional[str]


class Borg:
    # We use a borg pattern to share state across all instances of this class.
    # Due to submitting some errors in a thread,
    # we want to share the thread pool executor
    _shared_state: Dict[str, Any] = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_state


class Analytics(Borg):
    """Analytics is used to track errors and logs in the background"""

    _telemetrics_disabled: bool = True

    def __init__(self, ApiClient: Type[ApiClient], config: Config) -> None:
        """To initialize the Analytics class you need to pass in an ApiClient and the dq config.
        :param ApiClient: The ApiClient class
        :param config: The dq config
        """
        super().__init__()

        try:
            self._telemetrics_disabled = self._is_telemetrics_disabled(config)
            if self._telemetrics_disabled:
                return
            self.api_caller = ThreadPoolExecutor(max_workers=5)
            self.api_client = ApiClient()
            self.config = config

            if not getattr(self, "_initialized", False) and not getattr(
                self, "_locked", False
            ):
                self._locked = True
                self.last_error: Dict = {}
                self.last_log: Dict = {}
                self.user: ProfileModel = self._setup_user()
                self._init()
                self._initialized = True
                self._locked = False

        except Exception as e:
            self.debug_logging(e)

    def _is_telemetrics_disabled(self, config: Config) -> bool:
        """This function is used to check if the telemetrics are enabled."""
        api_url = os.environ.get("GALILEO_CONSOLE_URL", "")
        if hasattr(config, "api_url"):
            if isinstance(config.api_url, str):
                api_url = config.api_url
        elif hasattr(config, "get"):
            api_url = config.get("api_url", "")  # type: ignore
        # If dq telemetrics is enabled via env return false
        if os.environ.get("DQ_TELEMETRICS", False):
            return os.environ["DQ_TELEMETRICS"] != "1"
        # else check if galileo is in the api url
        return "galileo" not in api_url.lower()

    def debug_logging(self, log_message: Any, *args: Tuple) -> None:
        """This function is used to log debug messages.
        It will only log if the DQ_DEBUG environment variable is set to True."""
        # Logging in debugging mode
        if os.environ.get("DQ_DEBUG", False):
            print(log_message, *args)

    def _init(self) -> None:
        """This function is used to initialize the Exception hook.
        We use a different hook for ipython then normal python.
        """
        if self._telemetrics_disabled:
            return
        try:
            from IPython import get_ipython

            ip = get_ipython()
            ip.set_custom_exc((Exception,), self.ipython_exception_handler)
        except Exception:
            new_hook = change_function(
                sys.excepthook, self.handle_exception, AmpliMetric.dq_general_exception
            )

            sys.excepthook = new_hook

    def _setup_user(self) -> ProfileModel:
        """This function is used to setup the user information.
        This includes all installed packages.
        """
        profile = ProfileModel(**{"uuid": str(hex(uuid.getnode()))})
        try:
            profile.packages = _installed_modules()
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

        # we hook into the traceback,
        # inbetween we log the exception
        # and then show the original traceback.
        # because recently the track_exception_ipython was failing
        # we added a try except to make sure the original traceback is shown
        try:
            if not self._telemetrics_disabled:
                self.track_exception_ipython(
                    etype, evalue, tb, AmpliMetric.dq_general_exception
                )
        except Exception:
            # TODO: create internal logging endpoint
            pass
        # We need to call the default ipython exception handler to raise the error
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

    def track_exception_ipython(
        self,
        etype: Type[BaseException],
        evalue: BaseException,
        tb: TracebackType,
        scope: AmpliMetric = AmpliMetric.dq_general_exception,
    ) -> None:
        """We parse the current environment and send the error to the api."""
        if self._telemetrics_disabled:
            return
        data = parse_exception_ipython(etype, evalue, tb)
        self.last_error = data
        self.log(data, scope)

    def handle_exception(
        self,
        etype: Optional[Type[BaseException]],
        evalue: Optional[BaseException],
        tb: Optional[TracebackType],
        scope: AmpliMetric = AmpliMetric.dq_general_exception,
    ) -> None:
        """This function is used to handle exceptions in python."""
        if self._telemetrics_disabled:
            return
        if etype is not None and evalue is not None and tb is not None:
            try:
                data = parse_exception(etype, evalue, tb)
                self.last_error = data
                self.api_client.send_analytics(
                    str(self.config.current_project_id),
                    str(self.config.current_run_id),
                    str(self.config.task_type),
                    data,
                    scope,
                )
            except Exception:
                pass

    def capture_exception(
        self,
        error: Optional[Exception],
        scope: AmpliMetric = AmpliMetric.dq_galileo_warning,
    ) -> None:
        """This function is used to take an exception that is passed as an argument."""
        if self._telemetrics_disabled:
            return
        if error:
            exc_info = exception_from_error(error)
        else:
            exc_info = sys.exc_info()
        self.handle_exception(*exc_info, scope)

    def log_import(self, module: str) -> None:
        """This function is used to log an import of a module."""
        if self._telemetrics_disabled:
            return
        data = get_device_info()
        data["method"] = "import"
        data["value"] = module
        data["arguments"] = ""
        self.last_log = data
        self.log(data, AmpliMetric.dq_import)

    def log_function(self, function: str) -> None:
        """This function is used to log an functional call"""
        if self._telemetrics_disabled:
            return
        data = get_device_info()
        data["method"] = "function"
        data["value"] = function
        data["arguments"] = ""
        self.last_log = data
        self.log(data, AmpliMetric.dq_function_call)

    def log(self, data: Dict, scope: AmpliMetric) -> None:
        """This function is used to send the error to the api in a thread."""
        if self._telemetrics_disabled:
            return
        try:
            self.api_caller.submit(
                self.api_client.send_analytics,
                str(self.config.current_project_id),
                str(self.config.current_run_id),
                str(self.config.task_type),
                data,
                scope,
            )
        except Exception as e:
            self.debug_logging(e)

    def set_config(self, config: Any) -> None:
        """This function is used to set the config post init."""
        self.config = config
