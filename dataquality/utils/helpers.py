import os
from typing import Callable, Optional, TypeVar

from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")
GALILEO_DISABLED = "GALILEO_DISABLED"


def check_noop(func: Callable[P, T]) -> Callable[P, Optional[T]]:
    """Checks if GALILEO_DISABLED is set. If so, skip the function call

    https://peps.python.org/pep-0612/
    """

    def decorator(*args: P.args, **kwargs: P.kwargs) -> Optional[T]:
        if os.getenv(GALILEO_DISABLED):
            return None
        return func(*args, **kwargs)

    return decorator
