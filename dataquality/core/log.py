from typing import Any, Dict, List, Type, Union

from deprecate import deprecated

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger import BaseGalileoModelLogger


def log_input_data(**kwargs: Dict[str, Any]) -> None:
    assert config.task_type, "You must call dataquality.init before logging data"
    data_logger = get_data_logger()(**kwargs)
    data_logger.log()


# Backwards compatibility
@deprecated(
    target=None, template_mgs="`%(source_name)s` was deprecated, use `log_input_data`"
)
def log_batch_input_data(**kwargs: Any) -> None:
    log_input_data(**kwargs)


def log_model_outputs(**kwargs: Any) -> None:
    assert config.task_type, "You must call dataquality.init before logging data"
    model_logger = get_model_logger()(**kwargs)
    model_logger.log()


def set_labels_for_run(labels: Union[Dict[str, List[str]], List[str]]) -> None:
    """
    Creates the mapping of the labels for the model to their respective indexes.

    :param labels: An ordered list of labels (ie ['dog','cat','fish']
    If this is a multi-label type, then labels are a dictionary where each value is
    as listed above and the key is the label ("task")
    This order MUST match the order of probabilities that the model outputs
    :return: None
    """
    if isinstance(labels, List):
        if len(labels) == 1:
            labels = [labels[0], f"NOT_{labels[0]}"]
        config.labels = [str(i) for i in labels]
    elif isinstance(labels, Dict):  # multi-label
        for k in list(labels.keys()):
            if len(k) == 1:
                labels[k] = [labels[k][0], f"NOT_{labels[k][0]}"]
            labels[k] = [str(i) for i in labels[k]]
        config.labels = labels
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
