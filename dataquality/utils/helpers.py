import os
from functools import wraps
from typing import Callable, Optional, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
GALILEO_DISABLED = "GALILEO_DISABLED"


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
