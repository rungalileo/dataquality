import os
from functools import wraps
from typing import Callable, Optional, TypeVar

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
