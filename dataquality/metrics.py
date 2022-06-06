import os
import warnings
from typing import Dict, List, Optional

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
    return api_client.get_run_metrics(
        project_name,
        run_name,
        split,
        task=task,
        inference_name=inference_name,
        category=category,
    )


def display_dep_distribution(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
) -> None:
    """Displays the DEP distribution for a run. Plotly must be installed

    Calculates metrics (f1, recall, precision) overall (weighted) and per label.
    Also returns the top 50 rows of the dataframe (sorted by data_error_potential)

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    """
    try:
        import plotly.express as px
    except ImportError:
        raise GalileoException(
            "You must install plotly to use this function. Run `pip install plotly`"
        )
    summary = api_client.get_run_summary(
        project_name, run_name, split, task, inference_name
    )["split_run_results"]
    easy, hard = summary["easy_samples_threshold"], summary["hard_samples_threshold"]
    dep = summary["model_metrics"]["dep_distribution"]
    dep_bins, dep_counts = dep["dep_bins"], dep["count"]

    fig = px.bar(
        x=dep_bins[1:],
        y=dep_counts,
        labels={"x": "DEP", "y": "Count"},
        color=dep_bins[1:],
        color_continuous_scale=[
            (0, "green"),
            (easy, "yellow"),
            (hard, "red"),
            (1, "darkred"),
        ],
    )
    fig.show()


def get_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
) -> DataFrame:
    """Gets the dataframe for a run/split

    Downloads an arrow (or specified type) file to your machine and returns a loaded
    Vaex dataframe

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param file_type: The file type to download the data as. Default arrow
    :param include_embs: Whether to include the embeddings in the data. Default False
    :param include_probs: Whether to include the probs in the data. Default False
    """
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)

    file_name = f"data.{file_type}"
    api_client.export_run(project_name, run_name, split, file_name=file_name)
    data_df = vaex.open(file_name)

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


def _get_hdf5_file_for_epoch(
    project_name: str, run_name: str, split: Split, object_name: str, epoch: int = None
) -> DataFrame:
    emb = "emb" in object_name
    epoch = _validate_epoch(project_name, run_name, split, epoch, emb=emb)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    object_name = f"{project_id}/{run_id}/{split}/{epoch}/{object_name}"
    file_name = os.path.split(object_name)[-1]
    object_store.download_file(object_name, file_name)
    print(f"Your file has been written to {file_name}")
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
