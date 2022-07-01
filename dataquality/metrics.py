import os
import warnings
from typing import Dict, List, Optional
from uuid import uuid4

import vaex
from vaex.dataframe import DataFrame

from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.schemas.dataframe import FileType
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType

api_client = ApiClient()
object_store = ObjectStore()


def get_run_summary(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
) -> Dict:
    """Gets the summary for a run/split

    Calculates metrics (f1, recall, precision) overall (weighted) and per label.
    Also returns the top 50 rows of the dataframe (sorted by data_error_potential)

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    """
    return api_client.get_run_summary(
        project_name, run_name, split, task, inference_name
    )


def get_metrics(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    category: str = "gold",
) -> Dict[str, List]:
    """Calculates available metrics for a run/split, grouped by a particular category

    The category/column provided (can be gold, pred, or any categorical metadata column)
    will result in metrics per "group" or unique value of that category/column

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    :param category: The category/column to calculate metrics for. Default "gold"
        Can be "gold" for ground truth, "pred" for predicted values, or any metadata
        column logged (or smart feature).
    """
    metrics = api_client.get_run_metrics(
        project_name,
        run_name,
        split,
        task=task,
        inference_name=inference_name,
        category=category,
    )
    # Filter out metrics not available for this request
    metrics = {k: v for k, v in metrics.items() if v}
    return metrics


def display_distribution(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    column: str = "data_error_potential",
) -> None:
    """Displays the column distribution for a run. Plotly must be installed

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    :param column: The column to get the distribution for. Default data error potential
    """
    try:
        import plotly.express as px
    except ImportError:
        raise GalileoException(
            "You must install plotly to use this function. Run `pip install plotly`"
        )
    distribution = api_client.get_column_distribution(
        project_name,
        run_name,
        split,
        column=column,
        task=task,
        inference_name=inference_name,
    )
    bins, counts = distribution["bins"], distribution["counts"]
    labels = {"x": column, "y": "Count"}

    color_scale, color = None, None
    if column == "data_error_potential":
        summary = api_client.get_run_summary(
            project_name, run_name, split, task, inference_name
        )["split_run_results"]
        easy = summary["easy_samples_threshold"]
        hard = summary["hard_samples_threshold"]
        color_scale = [
            (0, "green"),
            (easy, "yellow"),
            (hard, "red"),
            (1, "darkred"),
        ]
        color = bins[1:]

    fig = px.bar(
        x=bins[1:],
        y=counts,
        labels=labels,
        color=color,
        color_continuous_scale=color_scale,
    )
    fig.show()


def get_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
) -> DataFrame:
    """Gets the dataframe for a run/split

    Downloads an arrow (or specified type) file to your machine and returns a loaded
    Vaex dataframe.

    Special note for NER. By default, the data will be downloaded at a sample level
    (1 row per sample text), with spans for each sample in a `spans` column in a
    spacy-compatible JSON format. If include_emb is True, the data will be expanded
    into span level (1 row per span, with sample text repeated for each span row), in
    order to join the span-level embeddings

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param file_type: The file type to download the data as. Default arrow
    :param include_embs: Whether to include the embeddings in the data. Default False
    :param include_probs: Whether to include the probs in the data. Default False
    :param include_token_indices: (NER only) Whether to include logged
        text_token_indices in the data. Useful for reconstructing tokens for retraining
    """
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)

    file_name = f"/tmp/{uuid4()}-data.{file_type}"
    api_client.export_run(project_name, run_name, split, file_name=file_name)
    data_df = vaex.open(file_name)
    # See docstring. In this case, we need span-level data
    if include_embs and task_type == TaskType.text_ner:
        # In NER, the `probabilities` contains the span level data
        span_df = get_probabilities(project_name, run_name, split)
        data_df = span_df.join(data_df[["text", "sample_id"]], on="sample_id")

    tasks = []
    if task_type == TaskType.text_multi_label:
        tasks = api_client.get_tasks_for_run(project_name, run_name)
        labels = [
            api_client.get_labels_for_run(project_name, run_name, task)
            for task in tasks
        ]
        data_df = _index_df(data_df, labels, tasks)
    if task_type == TaskType.text_classification:
        labels_per_task = api_client.get_labels_for_run(project_name, run_name)
        data_df = _index_df(data_df, labels_per_task)

    if include_embs:
        emb_df = get_embeddings(project_name, run_name, split)
        data_df = data_df.join(emb_df, on="id")
    if include_probs:
        if task_type == task_type.text_ner:
            warnings.warn(
                "Probabilities are not available for NER runs, ignoring", GalileoWarning
            )
        else:
            prob_df = get_probabilities(project_name, run_name, split)
            prob_cols = prob_df.get_column_names(regex="prob*") + ["id"]
            data_df = data_df.join(prob_df[prob_cols], on="id")
            data_df = _rename_prob_cols(data_df, tasks)
    if include_token_indices:
        if task_type != task_type.text_ner:
            warnings.warn(
                "Token indices are only available for NER, ignoring", GalileoWarning
            )
        else:
            raw_tokens = get_raw_data(project_name, run_name, split)
            raw_tokens = raw_tokens[["id", "text_token_indices"]]
            raw_tokens.rename("id", "sample_id")
            data_df = data_df.join(raw_tokens, on="sample_id")
    return data_df


def get_epochs(project_name: str, run_name: str, split: Split) -> List[int]:
    """Returns the epochs logged for a run/split

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    """
    return api_client.get_epochs_for_run(project_name, run_name, split)


def get_embeddings(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
    """Downloads the embeddings for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the embeddings from the final epoch. Note that only the
    n and n-1 epoch embeddings are available for download

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
    return _get_hdf5_file_for_epoch(
        project_name, run_name, split, "emb/emb.hdf5", epoch
    )


def get_probabilities(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
    """Downloads the probabilities for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the probabilities from the final epoch.

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
    return _get_hdf5_file_for_epoch(
        project_name, run_name, split, "prob/prob.hdf5", epoch
    )


def get_raw_data(
    project_name: str, run_name: str, split: Split, epoch: int = None
) -> DataFrame:
    """Downloads the raw logged data for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the probabilities from the final epoch.

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param epoch: The epoch to get embeddings for. Default final epoch
    """

    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)
    if task_type == TaskType.text_ner:
        object_name = "data/data.arrow"
    else:
        object_name = "data/data.hdf5"
    return _get_hdf5_file_for_epoch(project_name, run_name, split, object_name, epoch)


def get_xray_cards(
    project_name: str, run_name: str, split: Split, inference_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """Get xray cards for a project/run/split

    Xray cards are automatic insights calculated and provided by Galileo on your data
    """
    return api_client.get_xray_cards(project_name, run_name, split, inference_name)


def get_label_for_run(
    project_name: str, run_name: str, task: Optional[str] = None
) -> List[str]:
    """Gets labels for a given run. If multi-label, a task must be provided"""
    return api_client.get_labels_for_run(project_name, run_name, task)


def get_tasks_for_run(project_name: str, run_name: str) -> List[str]:
    """Gets task names for a multi-label run"""
    return api_client.get_tasks_for_run(project_name, run_name)


def _get_hdf5_file_for_epoch(
    project_name: str, run_name: str, split: Split, object_name: str, epoch: int = None
) -> DataFrame:
    emb = "emb" in object_name
    epoch = _validate_epoch(project_name, run_name, split, epoch, emb=emb)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    object_name = f"{project_id}/{run_id}/{split}/{epoch}/{object_name}"
    file_name = f"/tmp/{uuid4()}-{os.path.split(object_name)[-1]}"
    object_store.download_file(object_name, file_name)
    return vaex.open(file_name)


def _validate_epoch(
    project_name: str, run_name: str, split: Split, epoch: int = None, emb: bool = False
) -> int:
    epochs = get_epochs(project_name, run_name, split)
    last_epoch = epochs[-1]
    if not epochs:
        raise GalileoException(f"No epochs found for {project_name}/{run_name}")
    if epoch is None:
        epoch = last_epoch
    if epoch not in epochs:
        raise GalileoException(
            f"Run {project_name}/{run_name} has epochs {epochs}. You requested {epoch}"
        )
    # We only log the last 2 epochs of embeddings, so it must be one of those
    if emb and epoch < last_epoch - 1:
        raise GalileoException(
            f"Only the last 2 epochs of embeddings are available. "
            f"Must request {last_epoch} or {last_epoch-1}"
        )
    return epoch


def _index_df(df: DataFrame, labels: List, tasks: Optional[List] = None) -> DataFrame:
    """Indexes gold and pred columns"""
    # We do this so if this is TC (no tasks), we can still iterate with the same logic
    tasks = tasks or [None]
    for ind, task in enumerate(tasks):
        # If multi label, must do it per task. If TC, then it's just 1 list of labels
        task_labels = labels[ind] if task else labels
        for col in ["gold", "pred"]:
            df_col = f"{col}_{task}" if task else col
            df[f"{df_col}_idx"] = df[df_col]
            df = df.ordinal_encode(f"{df_col}_idx", values=task_labels, lazy=True)
    return df


def _rename_prob_cols(df: DataFrame, tasks: List[str]) -> DataFrame:
    for ind, task in enumerate(tasks):
        df.rename(f"prob_{ind}", f"prob_{task}")
    return df
