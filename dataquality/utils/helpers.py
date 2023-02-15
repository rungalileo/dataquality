import os
import webbrowser
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
GALILEO_DISABLED = "GALILEO_DISABLED"
GALILEO_VERBOSE = "GALILEO_VERBOSE"


def check_noop(func: Callable[P, T]) -> Callable[P, Optional[T]]:
    """Checks if GALILEO_DISABLED is set. If so, skip the function call

    https://peps.python.org/pep-0612/
    """

    # Wrap is used to preserve the docstring of the original function
    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        if galileo_disabled():
            return None
        return func(*args, **kwargs)

    return decorator


def galileo_disabled() -> bool:
    return os.getenv(GALILEO_DISABLED) in (True, "TRUE", "True", "true", 1)


def disable_galileo() -> None:
    os.environ[GALILEO_DISABLED] = "True"


def enable_galileo() -> None:
    os.environ[GALILEO_DISABLED] = "False"


def galileo_verbose_logging() -> bool:
    return os.getenv(GALILEO_VERBOSE) in (True, "TRUE", "True", "true", 1)


def enable_galileo_verbose() -> None:
    os.environ[GALILEO_VERBOSE] = "True"


def disable_galileo_verbose() -> None:
    os.environ[GALILEO_VERBOSE] = "False"


# generic hook for adding a debugger to a function
def wrap_fn(exist_func: Callable, hook_fn: Callable) -> Callable:
    """Hook a function to a function call
    :param exist_func: The function to hook
    :param hook_fn: The hook function
    :return: The hooked function

    Example:
    # example debugger
    def myobserver(orig_func, *args, **kwargs):
        # -----------------------
        # Your logic goes here
        # -----------------------
        print("debugging xyz")
        return orig_func(*args, **kwargs)

    # hook the function

    example_class.func = hook(example_class.func, myobserver)
    """

    @wraps(exist_func)
    def run(*args: Tuple, **kwargs: Dict[str, Any]) -> Callable:
        return hook_fn(exist_func, *args, **kwargs)

    return run


def map_indices_to_ids(id_map: List, indices: List) -> List:
    """Maps the indices to the ids
    :param id_map: The list used for mapping indices to ids
    :param indices: The indices to map
    :return: The ids
    """
    return [id_map[i] for i in indices]


def open_console_url(link: Optional[str] = "") -> None:
    """Tries to open the console url in the browser, if possible.

    This will work in local environments like jupyter or a python script, but won't
    work in colab (because colab is running on a server, so there's no "browser" to
    interact with). This also prints out the link for users to click so even in those
    environments they still have something to interact with.
    """
    if not link:
        return
    try:
        webbrowser.open(link)
    # In some environments, webbrowser will raise. Other times it fails silently (colab)
    except Exception:
        pass
    finally:
        print(f"Click here to see your run! {link}")
