from typing import Any, Dict, List, Optional, Type, Union

import numpy as np

from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.data_logger.base_data_logger import ITER_CHUNK_SIZE, DataSet
from dataquality.loggers.logger_config.text_multi_label import (
    text_multi_label_logger_config,
)
from dataquality.loggers.model_logger import BaseGalileoModelLogger
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import check_noop

DEFAULT_RANDOM_EMB_DIM = 2
a = Analytics(ApiClient, config)  # type: ignore


@check_noop
def log_data_samples(
    *,
    texts: List[str],
    ids: List[int],
    meta: Dict[str, List[Union[str, float, int]]] = None,
    **kwargs: Any,
) -> None:
    """Logs a batch of input samples for model training/test/validation/inference.

    Fields are expected as lists of their content. Field names are in the plural of
    `log_input_sample` (text -> texts)
    The expected arguments come from the task_type being used: See dq.docs() for details

    ex (text classification):
    .. code-block:: python

        all_labels = ["A", "B", "C"]
        dq.set_labels_for_run(labels = all_labels)

        texts: List[str] = [
            "Text sample 1",
            "Text sample 2",
            "Text sample 3",
            "Text sample 4"
        ]

        labels: List[str] = ["B", "C", "A", "A"]

        meta = {
            "sample_importance": ["high", "low", "low", "medium"]
            "quality_ranking": [9.7, 2.4, 5.5, 1.2]
        }

        ids: List[int] = [0, 1, 2, 3]
        split = "training"

        dq.log_data_samples(texts=texts, labels=labels, ids=ids, meta=meta split=split)

    :param texts: List[str] the input samples to your model
    :param ids: List[int | str] the ids per sample
    :param split: Optional[str] the split for this data. Can also be set via
        dq.set_split
    :param meta: Dict[str, List[str | int | float]]. Log additional metadata fields to
    each sample. The name of the field is the key of the dictionary, and the values are
    a list that correspond in length and order to the text samples.
    :param kwargs: See dq.docs() for details on other task specific parameters
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    data_logger.log_data_samples(texts=texts, ids=ids, meta=meta, **kwargs)


@check_noop
def log_data_sample(*, text: str, id: int, **kwargs: Any) -> None:
    """Log a single input example to disk

    Fields are expected singular elements. Field names are in the singular of
    `log_input_samples` (texts -> text)
    The expected arguments come from the task_type being used: See dq.docs() for details

    :param text: List[str] the input samples to your model
    :param id: List[int | str] the ids per sample
    :param split: Optional[str] the split for this data. Can also be set via
        dq.set_split
    :param kwargs: See dq.docs() for details on other task specific parameters
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    # For logging a single sample, we don't want a progress bar because it will be
    # nearly instant, and it's likely that the user will call this many times which
    # would flood the output.
    # We don't need to reset log_export_progress because this class instance is
    # ephemeral
    data_logger.log_export_progress = False
    data_logger.log_data_sample(text=text, id=id, **kwargs)


@check_noop
def log_dataset(
    dataset: DataSet,
    *,
    batch_size: int = ITER_CHUNK_SIZE,
    text: Union[str, int] = "text",
    id: Union[str, int] = "id",
    split: Optional[Split] = None,
    meta: Optional[List[Union[str, int]]] = None,
    **kwargs: Any,
) -> None:
    """Log an iterable or other dataset to disk. Useful for logging memory mapped files

    Dataset provided must be an iterable that can be traversed row by row, and for each
    row, the fields can be indexed into either via string keys or int indexes. Pandas
    and Vaex dataframes are also allowed, as well as HuggingFace Datasets

    valid examples:
        d = [
            {"my_text": "sample1", "my_labels": "A", "my_id": 1, "sample_quality": 5.3},
            {"my_text": "sample2", "my_labels": "A", "my_id": 2, "sample_quality": 9.1},
            {"my_text": "sample3", "my_labels": "B", "my_id": 3, "sample_quality": 2.7},
        ]
        dq.log_dataset(
            d, text="my_text", id="my_id", label="my_labels", meta=["sample_quality"]
        )

        Logging a pandas dataframe, df:
              text label  id  sample_quality
        0  sample1     A   1             5.3
        1  sample2     A   2             9.1
        2  sample3     B   3             2.7
        # We don't need to set text id or label because it matches the default
        dq.log_dataset(d, meta=["sample_quality"])

        Logging and iterable of tuples:
        d = [
            ("sample1", "A", "ID1"),
            ("sample2", "A", "ID2"),
            ("sample3", "B", "ID3"),
        ]
        dq.log_dataset(d, text=0, id=2, label=1)

    Invalid example:
        d = {
            "my_text": ["sample1", "sample2", "sample3"],
            "my_labels": ["A", "A", "B"],
            "my_id": [1, 2, 3],
            "sample_quality": [5.3, 9.1, 2.7]
        }

    In the invalid case, use `dq.log_data_samples`:
        meta = {"sample_quality": d["sample_quality"]}
        dq.log_data_samples(
            texts=d["my_text"], labels=d["my_labels"], ids=d["my_ids"], meta=meta
        )

    Keyword arguments are specific to the task type. See dq.docs() for details

    :param dataset: The iterable or dataframe to log
    :batch_size: The number of data samples to log at a time. Useful when logging a
    memory mapped dataset. A larger batch_size will result in faster logging at the
    expense of more memory usage. Default 100,000
    :param text: str | int The column, key, or int index for text data. Default "text"
    :param id: str | int The column, key, or int index for id data. Default "id"
    :param split: Optional[str] the split for this data. Can also be set via
        dq.set_split
    :param meta: List[str | int] Additional keys/columns to your input data to be
        logged as metadata. Consider a pandas dataframe, this would be the list of
        columns corresponding to each metadata field to log
    :param kwargs: See help(dq.get_data_logger().log_dataset) for more details here
    or dq.docs() for more general task details
    """
    a.log_function("dq/log_dataset")
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    data_logger = get_data_logger()
    data_logger.log_dataset(
        dataset,
        batch_size=batch_size,
        text=text,
        id=id,
        split=split,
        meta=meta,
        **kwargs,
    )


@check_noop
def log_model_outputs(
    *,
    embs: Optional[Union[List, np.ndarray]],
    ids: Union[List, np.ndarray],
    split: Optional[Split] = None,
    epoch: Optional[int] = None,
    logits: Union[List, np.ndarray] = None,
    probs: Union[List, np.ndarray] = None,
    inference_name: str = None,
    exclude_embs: bool = False,
) -> None:
    """Logs model outputs for model during training/test/validation.

    :param embs: The embeddings per output sample
    :param ids: The ids for each sample. Must match input ids of logged samples
    :param split: The current split. Must be set either here or via dq.set_split
    :param epoch: The current epoch. Must be set either here or via dq.set_epoch
    :param logits: The logits for each sample
    :param probs: Deprecated, use logits. If passed in, a softmax will NOT be applied
    :param inference_name: Inference name indicator for this inference split.
        If logging for an inference split, this is required.
    :param exclude_embs: Optional flag to exclude embeddings from logging. If True and
        embs is set to None, this will generate random embs for each sample.

    The expected argument shapes come from the task_type being used
    See dq.docs() for more task specific details on parameter shape
    """
    assert all(
        [config.task_type, config.current_project_id, config.current_run_id]
    ), "You must call dataquality.init before logging data"
    assert (probs is not None) or (
        logits is not None
    ), "You must provide either logits or probs"
    assert (embs is None and exclude_embs) or (
        embs is not None and not exclude_embs
    ), "embs can be omitted if and only if exclude_embs is True"
    if embs is None and exclude_embs:
        embs = np.random.rand(len(ids), DEFAULT_RANDOM_EMB_DIM)

    model_logger = get_model_logger()(
        embs=embs,
        ids=ids,
        split=Split[split].value if split else "",
        epoch=epoch,
        logits=logits,
        probs=probs,
        inference_name=inference_name,
    )
    model_logger.log()


@check_noop
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
    a.log_function("dq/set_labels_for_run")

    get_data_logger().logger_config.labels = labels


@check_noop
def set_tasks_for_run(tasks: List[str], binary: bool = True) -> None:
    """Sets the task names for the run (multi-label case only).

    This order MUST match the order of the labels list provided in log_input_data
    and the order of the probability vectors provided in log_model_outputs.

    This also must match the order of the labels logged in set_labels_for_run (meaning
    that the first list of labels must be the labels of the first task passed in here)

    :param tasks: The list of tasks for your run
    :param binary: Whether this is a binary multi label run. If true, tasks will also
    be set as your labels, and you should NOT call `dq.set_labels_for_run` it will be
    handled for you. Default True
    """
    if config.task_type != TaskType.text_multi_label:
        raise GalileoException("You can only set task names for multi-label use cases.")
    get_data_logger().logger_config.tasks = tasks
    text_multi_label_logger_config.binary = binary
    if binary:
        # The labels validator will handle adding the "NOT_" to each label
        text_multi_label_logger_config.labels = [[task] for task in tasks]


@check_noop
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
            "You must provide either a task_type or first call "
            "dataqualtiy.init and provide one"
        )
    return task


def docs() -> None:
    """Print the documentation for your specific input and output logging format

    Based on your task_type, this will print the appropriate documentation
    """
    get_data_logger().doc()
    get_model_logger().doc()


@check_noop
def set_epoch(epoch: int) -> None:
    """Set the current epoch.

    When set, logging model outputs will use this if not logged explicitly
    """
    get_data_logger().logger_config.cur_epoch = epoch


@check_noop
def set_split(split: Split, inference_name: Optional[str] = None) -> None:
    """Set the current split.

    When set, logging data inputs/model outputs will use this if not logged explicitly
    When setting split to inference, inference_name must be included
    """
    get_data_logger().logger_config.cur_inference_name = inference_name
    split = Split[split]
    setattr(get_data_logger().logger_config, f"{split}_logged", True)
    # Set cur_inference_name before split for pydantic validation
    get_data_logger().logger_config.cur_split = split


@check_noop
def set_epoch_and_split(
    epoch: int, split: Split, inference_name: Optional[str] = None
) -> None:
    """Set the current epoch and set the current split.
    When set, logging data inputs/model outputs will use this if not logged explicitly
    When setting split to inference, inference_name must be included
    """
    set_epoch(epoch)
    set_split(split, inference_name)
