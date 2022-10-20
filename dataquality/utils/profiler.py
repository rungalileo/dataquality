# License: BSD-2-Clause license
# part of the code is taken from sentry logging implemtnation
import linecache
import os
import platform
import re
import sys
import traceback
from functools import wraps
from types import FrameType, ModuleType, TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from dataquality.utils.ampli import AmpliMetric

ExcInfo = Tuple[
    Optional[Type[BaseException]], Optional[BaseException], Optional[TracebackType]
]

MAX_STRING_LENGTH = 1024
BASE64_ALPHABET = re.compile(r"^[a-zA-Z0-9/+=]*$")


def parse_exception_ipython(
    etype: Type[BaseException],
    evalue: BaseException,
    tb: TracebackType,
    stacktrace: str,
) -> Dict:
    error_type = etype.__name__
    error_message = ", ".join(evalue.args)
    error_stacktrace = stacktrace
    return {
        **get_device_info(),
        "error_type": str(error_type),
        "error_message": str(error_message),
        "error_stacktrace": str(error_stacktrace),
    }


def parse_exception(
    etype: Type[BaseException], evalue: BaseException, tb: TracebackType
) -> Dict:
    error_type = etype.__name__
    error_message = ", ".join(evalue.args)
    error_stacktrace = serialize_frame(tb.tb_frame)
    return {
        **get_device_info(),
        "error_type": str(error_type),
        "error_message": str(error_message),
        "error_stacktrace": str(error_stacktrace),
    }


def get_device_info() -> Dict:
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


def strip_string(value: str, max_length: Optional[int] = None) -> str:
    if not value:
        return value
    if max_length is None:
        # This is intentionally not just the default such that
        # one can patch `MAX_STRING_LENGTH` and affect `strip_string`.
        max_length = MAX_STRING_LENGTH

    length = len(value)
    if length > max_length:
        return value[: max_length - 3] + "..."
    return value


def get_lines_from_file(
    filename: str,
    lineno: int,
    loader: Optional[Any],
    module: Optional[str] = None,
) -> Tuple[List[str], Optional[str], List[str]]:
    context_lines = 5
    source = None
    if loader is not None and hasattr(loader, "get_source"):
        try:
            source_str: Optional[str] = loader.get_source(module)
        except (ImportError, IOError):
            source_str = None
        if source_str is not None:
            source = source_str.splitlines()

    if source is None:
        try:
            source = linecache.getlines(filename)
        except (OSError, IOError):
            return [], None, []

    if not source:
        return [], None, []

    lower_bound = max(0, lineno - context_lines)
    upper_bound = min(lineno + 1 + context_lines, len(source))

    try:
        pre_context = [
            strip_string(line.strip("\r\n")) for line in source[lower_bound:lineno]
        ]
        context_line = strip_string(source[lineno].strip("\r\n"))
        post_context = [
            strip_string(line.strip("\r\n"))
            for line in source[(lineno + 1) : upper_bound]
        ]
        return pre_context, context_line, post_context
    except IndexError:
        # the file may have changed since it was loaded into memory
        return [], None, []


def filename_for_module(
    module: Optional[str], abs_path: Optional[str]
) -> Optional[str]:

    if not abs_path or not module:
        return abs_path

    try:
        if abs_path.endswith(".pyc"):
            abs_path = abs_path[:-1]

        base_module = module.split(".", 1)[0]
        if base_module == module:
            return os.path.basename(abs_path)

        base_module_path = sys.modules[base_module].__file__
        if not base_module_path:
            return abs_path

        return abs_path.split(base_module_path.rsplit(os.sep, 2)[0], 1)[-1].lstrip(
            os.sep
        )
    except Exception:
        return abs_path


def get_source_context(
    frame: FrameType, tb_lineno: int
) -> Tuple[List[str], Optional[str], List[str]]:
    try:
        abs_path: Optional[str] = frame.f_code.co_filename
    except Exception:
        abs_path = None
    try:
        module = frame.f_globals["__name__"]
    except Exception:
        return [], None, []
    try:
        loader = frame.f_globals["__loader__"]
    except Exception:
        loader = None
    lineno = tb_lineno - 1
    if lineno is not None and abs_path:
        return get_lines_from_file(abs_path, lineno, loader, module)
    return [], None, []


def serialize_frame(
    frame: FrameType, tb_lineno: Optional[int] = None, with_locals: bool = True
) -> Dict[str, Any]:
    f_code = getattr(frame, "f_code", None)
    if not f_code:
        abs_path = None
        function = None
    else:
        abs_path = frame.f_code.co_filename
        function = frame.f_code.co_name
    try:
        module = frame.f_globals["__name__"]
    except Exception:
        module = None

    if tb_lineno is None:
        tb_lineno = frame.f_lineno

    pre_context, context_line, post_context = get_source_context(frame, tb_lineno)

    rv: Dict[str, Any] = {
        "filename": filename_for_module(module, abs_path) or None,
        "abs_path": os.path.abspath(abs_path) if abs_path else None,
        "function": function or "<unknown>",
        "module": module,
        "lineno": tb_lineno,
        "pre_context": pre_context,
        "context_line": context_line,
        "post_context": post_context,
    }
    if with_locals:
        rv["vars"] = frame.f_locals

    return rv


def _generate_installed_modules() -> Iterator[Tuple[str, str]]:
    try:
        import pkg_resources
    except ImportError:
        return

    for info in pkg_resources.working_set:
        yield info.key, info.version


def _list_sys_modules() -> set[str]:
    modules = set()
    for k in set(sys.modules):
        if "." in k:
            modules.add(k[: k.index(".")])
        else:
            modules.add(k)
    return modules


def _list_global_imports() -> Iterable[str]:
    for val in globals().values():
        if isinstance(val, ModuleType):
            yield val.__name__


def exc_info_from_error(error: Union[BaseException, ExcInfo]) -> ExcInfo:
    #
    if isinstance(error, tuple) and len(error) == 3:
        exc_type, exc_value, tb = error
    elif isinstance(error, BaseException):
        tb = getattr(error, "__traceback__", None)
        if tb is not None:
            exc_type = type(error)
            exc_value = error
        else:
            exc_type, exc_value, tb = sys.exc_info()
            if exc_value is not error:
                tb = None
                exc_value = error
                exc_type = type(error)

    else:
        raise ValueError("Expected Exception object to report, got %s!" % type(error))

    return exc_type, exc_value, tb
