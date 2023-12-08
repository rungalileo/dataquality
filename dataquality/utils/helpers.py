import os
import webbrowser
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from typing_extensions import ParamSpec

from dataquality.exceptions import GalileoException

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
    try:
        return [id_map[i] for i in indices]
    except IndexError:
        raise GalileoException(
            "The indicies of the model output are not matching the logged data "
            "samples. If you are using RandomSampler or WeightedRandomSampler, "
            "pass dataloader_random_sampling=True to the watch function"
        )


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


def gpu_available() -> bool:
    import torch

    return torch.cuda.is_available()


def mps_available() -> bool:
    """Checks for an MPS compatible GPU on Apple machines.

    This will enabled Metal acceleration for model training when supported.
    """
    import torch

    try:
        return torch.backends.mps.is_available()
    except Exception:
        return False


def has_len(arr: Any) -> bool:
    """Checks if an array has length

    Array can be list, numpy array, or tensorflow tensor. Tensorflow tensors don't
    let you call len(), they throw a TypeError so we catch that here and check
    shape https://github.com/tensorflow/tensorflow/blob/master/tensorflow/...
    python/framework/ops.py#L929
    """
    try:
        has_len = len(arr) != 0
    except TypeError:
        has_len = bool(arr.shape[0])
    return has_len


def _validate_str(
    label_or_task: str,
    labels_or_tasks_name: str,
) -> None:
    """Checks if the labels and tasks are valid for the UI

    The UI only supports alphanumeric characters, dashes, and underscores
    """
    if not label_or_task.replace("-", "").replace("_", "").isalnum():
        raise GalileoException(
            f"{labels_or_tasks_name} `{label_or_task}` is not valid. Only alphanumeric "
            "characters, dashes, and underscores are supported."
        )


def validate_labels_and_tasks(
    labels_or_tasks: Union[List[List[str]], List[str]],
    labels_or_tasks_name: str,
) -> None:
    """Checks if the labels and tasks are valid for the UI

    The UI only supports alphanumeric characters, dashes, and underscores
    """
    if len(labels_or_tasks) == 0:
        return

    if isinstance(labels_or_tasks[0], list):
        for lst in labels_or_tasks:
            assert isinstance(lst, list)
            validate_labels_and_tasks(lst, labels_or_tasks_name)

    elif isinstance(labels_or_tasks[0], str):
        for val in labels_or_tasks:
            assert isinstance(val, str)
            _validate_str(val, labels_or_tasks_name)
