from typing import Any, Callable, List, Optional, Type, Union

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.model_logger import BaseGalileoModelLogger
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType


def add_doc(doc: str) -> Callable:
    def _doc(func: Callable) -> Callable:
        func.__doc__ = doc
        return func

    return _doc


def log_data_samples(*, texts: List[str], ids: List[int], **kwargs: Any) -> None:
    """Logs a batch of input samples for model training/test/validation/inference.

    Fields are expected as lists of their content. Field names are in the plural of
    `log_input_sample` (text -> texts)
    The expected arguments come from the task_type's data logging schema.
    logger: See dq.docs() for details
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    data_logger.log_data_samples(texts=texts, ids=ids, **kwargs)


def log_data_sample(*, text: str, id: int, **kwargs: Any) -> None:
    """Log a single input example to disk"""
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    data_logger.log_data_sample(text=text, id=id, **kwargs)


def log_dataset(
    dataset: DataSet,
    *,
    text: Union[str, int] = "text",
    id: Union[str, int] = "id",
    **kwargs: Any,
) -> None:
    """Log an iterable or other dataset to disk.

    Dataset provided must be an either a pandas or vaex dataframe, or an iterable that
    can be traversed row by row, and for each row, the fields can be indexed into
    either via string keys or int indexes.

    valid examples:
        d = [
            {"my_text": "sample1", "my_labels": "A", "my_id": 1},
            {"my_text": "sample2", "my_labels": "A", "my_id": 2},
            {"my_text": "sample3", "my_labels": "B", "my_id": 3},
        ]
        dq.log_dataset(d, text="my_text", id="my_id", label="my_labels")
        Another:
        d = [
            ("sample1", "A", "ID1"),
            ("sample2", "A", "ID2"),
            ("sample3", "B", "ID3"),
        ]
        dq.log_dataset(d, text=0, id=2, label=1)
    invalid example:
        d = {
            "my_text": ["sample1", "sample2", "sample3"],
            "my_labels": ["A", "A", "B"],
            "my_id": [1, 2, 3],
        }

    In the invalid case, use log_data_samples:
        dq.log_data_samples(texts=d["my_text"], labels=d["my_labels"], ids=d["my_ids"])


    Keyword arguments are specific to the task type. See dq.docs() for details
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    data_logger.log_dataset(dataset, text=text, id=id, **kwargs)


def log_model_outputs(**kwargs: Any) -> None:
    """Logs model outputs for model during training/test/validation.

    The expected arguments come from the task_type's model
    logger: See dataquality.get_model_logger().doc() for details
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    model_logger = get_model_logger()(**kwargs)
    model_logger.log()


def set_labels_for_run(labels: Union[List[List[str]], List[str]]) -> None:
    """
    Creates the mapping of the labels for the model to their respective indexes.

    :param labels: An ordered list of labels (ie ['dog','cat','fish']
    If this is a multi-label type, then labels are a list of lists where each inner
    list indicates the label for the given task

    This order MUST match the order of probabilities that the model outputs.

    In the multi-label case, the outer order (order of the tasks) must match the
    task-order of the task-probabilities logged as well.
    """
    get_data_logger().logger_config.labels = labels


def set_tasks_for_run(tasks: List[str]) -> None:
    """Sets the task names for the run (multi-label case only).

    This order MUST match the order of the labels list provided in log_input_data
    and the order of the probability vectors provided in log_model_outputs.

    This also must match the order of the labels logged in set_labels_for_run (meaning
    that the first list of labels must be the labels of the first task passed in here)
    """
    if config.task_type != TaskType.text_multi_label:
        raise GalileoException("You can only set task names for multi-label use cases.")
    get_data_logger().logger_config.tasks = tasks


def set_tagging_schema(tagging_schema: TaggingSchema) -> None:
    """Sets the tagging schema for NER models

    Only valid for text_ner task_types. Others will throw an exception
    """
    get_data_logger().set_tagging_schema(tagging_schema)


def get_model_logger(task_type: TaskType = None) -> Type[BaseGalileoModelLogger]:
    task_type = _get_task_type(task_type)
    return BaseGalileoModelLogger.get_logger(task_type)


def get_data_logger(task_type: TaskType = None) -> BaseGalileoDataLogger:
    task_type = _get_task_type(task_type)
    return BaseGalileoDataLogger.get_logger(task_type)()


def _get_task_type(task_type: TaskType = None) -> TaskType:
    task = task_type or config.task_type
    if not task:
        raise GalileoException(
            "You must provide either a task_type or first call"
            "dataqualtiy.init and provide one"
        )
    return task


def docs() -> None:
    """Print the documentation for your specific input and output logging format

    Based on your task_type, this will print the appropriate documentation
    """
    get_data_logger().doc()
    get_model_logger().doc()


def set_epoch(epoch: int) -> None:
    """Set the current epoch.

    When set, logging model outputs will use this if not logged explicitly
    """
    get_data_logger().logger_config.cur_epoch = epoch


def set_split(split: Split, inference_name: Optional[str] = None) -> None:
    """Set the current split.

    When set, logging data inputs/model outputs will use this if not logged explicitly
    When setting split to inference, inference_name must be included
    """
    get_data_logger().logger_config.cur_inference_name = inference_name
    setattr(get_data_logger().logger_config, f"{split}_logged", True)
    # Set cur_inference_name before split for pydantic validation
    get_data_logger().logger_config.cur_split = split
