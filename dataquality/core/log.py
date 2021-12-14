import pydoc
import warnings
from typing import Any, Dict, List, Type, Union

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger import BaseGalileoModelLogger


def log_input_data(**kwargs: Dict[str, Any]) -> None:
    """Logs input data for model training/test/validation.

    The expected arguments come from the task_type's data
    logger: See print(dataquality.get_model_logger().__doc__) for details
    """
    assert config.task_type, "You must call dataquality.init before logging data"
    data_logger = get_data_logger()(**kwargs)
    data_logger.log()


# Backwards compatibility
def log_batch_input_data(**kwargs: Any) -> None:
    warnings.warn(
        "log_batch_input_data is deprecated. Use log_input_data", DeprecationWarning
    )
    log_input_data(**kwargs)


def log_model_outputs(**kwargs: Any) -> None:
    """Logs model outputs for model during training/test/validation.

    The expected arguments come from the task_type's model
    logger: See print(dataquality.get_model_logger().__doc__) for details
    """
    assert config.task_type, "You must call dataquality.init before logging data"
    model_logger = get_model_logger()(**kwargs)
    model_logger.log()


def set_labels_for_run(labels: Union[List[List[str]], List[str]]) -> None:
    """
    Creates the mapping of the labels for the model to their respective indexes.

    :param labels: An ordered list of labels (ie ['dog','cat','fish']
    If this is a multi-label type, then labels are a list of lists where each inner
    list indicates the label for the given task

    This order MUST match the order of probabilities that the model outputs.

    In the multi-label case, the outer order must match the task-order of the
    task-probabilities logged as well.
    """
    if isinstance(labels[0], List):  # multi-label
        cleaned_labels = []
        for task_labels in labels:
            if len(task_labels) == 1:
                task_labels = [task_labels[0], f"NOT_{task_labels[0]}"]
            cleaned_labels.append([str(i) for i in task_labels])
        config.labels = cleaned_labels
    else:
        if len(labels) == 1:
            labels = [labels[0], f"NOT_{labels[0]}"]
        config.labels = [str(i) for i in labels]

    config.update_file_config()


def get_model_logger(task_type: str = None) -> Type[BaseGalileoLogger]:
    task_type = _get_task_type(task_type)
    return BaseGalileoModelLogger.get_logger(task_type)


def get_data_logger(task_type: str = None) -> Type[BaseGalileoLogger]:
    task_type = _get_task_type(task_type)
    return BaseGalileoDataLogger.get_logger(task_type)


def _get_task_type(task_type: str = None) -> str:
    task = task_type or config.task_type
    if not task:
        raise GalileoException(
            "You must provide either a task_type or first call"
            "dataqualtiy.init and provide one"
        )
    return task
