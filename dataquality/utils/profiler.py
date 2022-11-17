import platform
import sys
import traceback
from functools import wraps
from types import ModuleType, TracebackType
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple, Type, Union

from dataquality.exceptions import GalileoException
from dataquality.utils.ampli import AmpliMetric

OptExcInfo = Tuple[
    Union[Type[BaseException], None],
    Union[BaseException, None],
    Union[TracebackType, None],
]


def parse_exception_ipython(
    etype: Type[BaseException], evalue: BaseException, tb: TracebackType
) -> Dict:
    """Parse exception for IPython"""

    from IPython import get_ipython

    # We need to get the lines of code that caused the error
    lines = "\n".join([h[2] for h in get_ipython().history_manager.get_range()])
    # To convert the exception to a string we need to call the TracebackException
    for line in traceback.TracebackException(
        type(evalue), evalue, tb, limit=1024
    ).format(chain=True):
        lines += line
    # We track the parsed error
    error_type = etype.__name__
    error_messages = []
    # The BaseException has args we want to log.
    # These are our error messages
    for arg in evalue.args:
        try:
            # We try to convert the arg to a string
            # this was causing issues with some errors
            # that had non-string args
            error_messages.append(str(arg))
        except Exception:
            pass
    error_message = ", ".join(error_messages)
    error_stacktrace = lines
    return {
        **get_device_info(),
        "error_type": str(error_type),
        "error_message": str(error_message),
        "error_stacktrace": str(error_stacktrace),
    }


def parse_exception(
    etype: Type[BaseException], evalue: BaseException, tb: TracebackType
) -> Dict:
    """Parse exception for Python"""
    error_type = etype.__name__
    error_message = ", ".join(evalue.args)
    lines = ""  # .join([h[2] for h in get_ipython().history_manager.get_range()])
    for line in traceback.TracebackException(
        type(evalue), evalue, tb, limit=1024
    ).format(chain=True):
        lines += line
    error_stacktrace = lines
    return {
        **get_device_info(),
        "error_type": str(error_type),
        "error_message": str(error_message),
        "error_stacktrace": str(error_stacktrace),
    }


def get_device_info() -> Dict:
    """Get device info. For example,
    Operating system, Python version, etc."""
    device_architecture = platform.machine()
    os_name = platform.system()
    os_version = platform.release()
    runtime_name = platform.python_implementation()
    runtime_version = platform.python_version()
    jupyter_env = ""
    jupyter = False
    try:
        from IPython import get_ipython

        jupyter_env = str(type(get_ipython()))
        jupyter = True
    except Exception:
        pass

    context_name = platform.python_implementation()
    context_version = "%s.%s.%s" % (sys.version_info[:3])
    context_build = sys.version

    filename, line, procname, text = traceback.extract_stack()[0]

    meta_filename = filename
    meta_function = procname

    return {
        "device_architecture": str(device_architecture),
        "os_name": str(os_name),
        "os_version": str(os_version),
        "runtime_name": str(runtime_name),
        "runtime_version": str(runtime_version),
        "jupyter_env": str(jupyter_env),
        "jupyter": str(jupyter),
        "context_name": str(context_name),
        "context_version": str(context_version),
        "context_build": str(context_build),
        "meta_filename": str(meta_filename),
        "meta_function": str(meta_function),
    }


def change_function(
    func: Callable, handle_request: Callable, scope: Union[AmpliMetric, None] = None
) -> Callable:
    """Change function to hook into the exception in python"""

    @wraps(func)
    def showtraceback(*args: List[Any], **kwargs: Dict) -> Callable:
        # extract exception type, value and traceback
        #        etype, evalue, tb = sys.exc_info()
        try:
            handle_request(*args, scope)
        except Exception:
            pass
        return func(*args, **kwargs)

    return showtraceback


def _installed_modules() -> Dict[str, str]:
    """Get installed modules and their version"""
    try:
        import pkg_resources
    except ImportError:
        return {}

    return {info.key: info.version for info in pkg_resources.working_set}


def _list_sys_modules() -> Set[str]:
    """Get all modules in sys.modules"""
    modules = set()
    for k in set(sys.modules):
        if "." in k:
            modules.add(k[: k.index(".")])
        else:
            modules.add(k)
    return modules


def _list_global_imports() -> Iterable[str]:
    """Get all global imports"""
    for val in globals().values():
        if isinstance(val, ModuleType):
            yield val.__name__


def exception_from_error(error: Union[BaseException, OptExcInfo]) -> OptExcInfo:
    """Get stacktrace from an exception"""
    # Inspired by: https://github.com/getsentry/sentry-python/tree/master/sentry_sdk
    # Therefore BSD2 licensed
    if isinstance(error, tuple) and len(error) == 3:
        return error
    elif isinstance(error, BaseException):
        tb = getattr(error, "__traceback__", None)
        if tb is not None:
            exception_type = type(error)
            exception_value = error
        else:
            _exception_type, _exception_value, tb = sys.exc_info()
            if _exception_value is not error:
                tb = None
                exception_value = error
                exception_type = type(error)
            elif _exception_type is not None:
                exception_type = _exception_type
                exception_value = _exception_value

        return exception_type, exception_value, tb

    else:
        raise GalileoException(f"Invalid error type: {error}")
