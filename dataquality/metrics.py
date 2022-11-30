import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import vaex
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
from vaex.dataframe import DataFrame

from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.schemas.dataframe import FileType
from dataquality.schemas.edit import Edit
from dataquality.schemas.metrics import FilterParams
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split, conform_split
from dataquality.schemas.task_type import TaskType

api_client = ApiClient()
object_store = ObjectStore()
a = Analytics(ApiClient, config)  # type: ignore
a.log_import("dq/metrics")


def _cache_key(*args: Tuple, **kwargs: Dict[str, Any]) -> Tuple:
    """Custom cache key that includes the updated_at timestamp for a run

    https://cachetools.readthedocs.io/en/latest/#cachetools.keys.typedkey

    We use this cache only for the heavy functions in metrics (downloading things
    from the server). We have a modified cache key that includes the last time
    this run was updated, in case that has changed since the last call.

    NOTE: It is assumed this cache takes the project_name and run_name as either args
    0 and 1, or 1 and 2 (see comments), never kwargs. As such, this cache is meant to be
    used for internal (not user facing) functions.
    """
    # First 2 arguments are project and run name
    if isinstance(args[0], str):
        project_name, run_name = args[0], args[1]
    # First argument is the dataframe, then project and run name
    else:
        project_name, run_name = args[1], args[2]
    updated_ts = api_client.get_project_run_by_name(
        str(project_name), str(run_name)
    ).get("updated_at")
    key = hashkey(*args, *kwargs.items())
    key += (updated_ts,)
    return key


def _get_cache() -> LRUCache:
    return LRUCache(maxsize=128)


def create_edit(
    project_name: str,
    run_name: str,
    split: Split,
    edit: Union[Edit, Dict],
    filter: Union[FilterParams, Dict],
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
) -> Dict:
    """Creates an edit for a run given a filter

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split
    :param edit: The edit to make. see `help(Edit)` for more information
    :param task: Required task name if run is MLTC
    :param inference_name: Required inference name if split is inference
    """
    split = conform_split(split)
    filter_params = _validate_filter(filter)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    edit = _conform_edit(edit)
    edit.project_id = project_id
    edit.run_id = run_id
    edit.split = split
    edit.task = task
    edit.inference_name = inference_name
    edit.filter = FilterParams(**filter_params)
    return api_client.create_edit(edit)


def get_run_summary(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    filter: Union[FilterParams, Dict] = None,
) -> Dict:
    """Gets the summary for a run/split

    Calculates metrics (f1, recall, precision) overall (weighted) and per label.
    Also returns the top 50 rows of the dataframe (sorted by data_error_potential)

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    :param filter: Optional filter to provide to restrict the summary to only to
        matching rows. See `dq.schemas.metrics.FilterParams`
    """
    split = conform_split(split)
    filter_params = _validate_filter(filter)
    return api_client.get_run_summary(
        project_name, run_name, split, task, inference_name, filter_params=filter_params
    )


def get_metrics(
    project_name: str,
    run_name: str,
    split: Split,
    task: Optional[str] = None,
    inference_name: Optional[str] = None,
    category: str = "gold",
    filter: Union[FilterParams, Dict] = None,
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
    :param filter: Optional filter to provide to restrict the metrics to only to
        matching rows. See `dq.schemas.metrics.FilterParams`
    """
    a.log_function("dq/metrics/get_metrics")

    split = conform_split(split)
    metrics = api_client.get_run_metrics(
        project_name,
        run_name,
        split,
        task=task,
        inference_name=inference_name,
        category=category,
        filter_params=_validate_filter(filter),
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
    filter: Union[FilterParams, Dict] = None,
) -> None:
    """Displays the column distribution for a run. Plotly must be installed

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    :param column: The column to get the distribution for. Default data error potential
    :param filter: Optional filter to provide to restrict the distribution to only to
        matching rows. See `dq.schemas.metrics.FilterParams`
    """
    try:
        import plotly.express as px
    except ImportError:
        raise GalileoException(
            "You must install plotly to use this function. Run `pip install plotly`"
        )
    split = conform_split(split)
    distribution = api_client.get_column_distribution(
        project_name,
        run_name,
        split,
        column=column,
        task=task,
        inference_name=inference_name,
        filter_params=_validate_filter(filter),
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


@cached(_get_cache(), key=_cache_key)
def _download_df(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str,
    file_type: FileType,
    hf_format: bool,
    tagging_schema: Optional[TaggingSchema],
    filter_params: FilterParams,
) -> DataFrame:
    """Helper function to download the dataframe to take advantage of caching

    The filter_params class FilterParams is hashable (wherein a dict is not) so we
    force passing that in `get_dataframe`
    """
    split = conform_split(split)

    file_name = f"/tmp/{uuid4()}-data.{file_type}"
    api_client.export_run(
        project_name,
        run_name,
        split,
        inference_name=inference_name,
        file_name=file_name,
        filter_params=filter_params.dict(),
        hf_format=hf_format,
        tagging_schema=tagging_schema,
    )
    data_df = vaex.open(file_name)
    return data_df


def get_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
    hf_format: bool = False,
    tagging_schema: Optional[TaggingSchema] = None,
    filter: Union[FilterParams, Dict] = None,
    as_pandas: bool = True,
    include_data_embs: bool = False,
) -> Union[pd.DataFrame, DataFrame]:
    """Gets the dataframe for a run/split

    Downloads an arrow (or specified type) file to your machine and returns a loaded
    Vaex dataframe.

    Special note for NER. By default, the data will be downloaded at a sample level
    (1 row per sample text), with spans for each sample in a `spans` column in a
    spacy-compatible JSON format. If include_embs or include_probs is True,
    the data will be expanded into span level (1 row per span, with sample text repeated
    for each span row), in order to join the span-level embeddings/probs

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference. The name of the inference
        split to get data for.
    :param file_type: The file type to download the data as. Default arrow
    :param include_embs: Whether to include the embeddings in the data. Default False
    :param include_probs: Whether to include the probs in the data. Default False
    :param include_token_indices: (NER only) Whether to include logged
        text_token_indices in the data. Useful for reconstructing tokens for retraining
    :param hf_format: (NER only)
        Whether to export the dataframe in a HuggingFace compatible format
    :param tagging_schema: (NER only)
        If hf_format is True, you must pass a tagging schema
    :param filter: Optional filter to provide to restrict the distribution to only to
        matching rows. See `dq.schemas.metrics.FilterParams`
    :param as_pandas: Whether to return the dataframe as a pandas df (or vaex if False)
        If you are having memory issues (the data is too large), set this to False,
        and vaex will memory map the data. If any columns returned are multi-dimensional
        (embeddings, probabilities etc), vaex will always be returned, because pandas
        cannot support multi-dimensional columns. Default True
    :param include_data_embs: Whether to include the off the shelf data embeddings
    """
    split = conform_split(split)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)
    filter_params = FilterParams(**_validate_filter(filter))

    data_df = _download_df(
        project_name,
        run_name,
        split,
        inference_name,
        file_type,
        hf_format,
        tagging_schema,
        filter_params,
    )
    return _process_exported_dataframe(
        data_df,
        project_name,
        run_name,
        split,
        task_type,
        inference_name,
        include_embs,
        include_probs,
        include_token_indices,
        hf_format,
        as_pandas,
        include_data_embs,
    )


def get_edited_dataframe(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    file_type: FileType = FileType.arrow,
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
    hf_format: bool = False,
    tagging_schema: Optional[TaggingSchema] = None,
    as_pandas: bool = True,
    include_data_embs: bool = False,
) -> Union[pd.DataFrame, DataFrame]:
    """Gets the edited dataframe for a run/split

    Exports a run/split's data with all active edits in the edits cart and returns
    a vaex or pandas dataframe

    Special note for NER. By default, the data will be downloaded at a sample level
    (1 row per sample text), with spans for each sample in a `spans` column in a
    spacy-compatible JSON format. If include_embs or include_probs is True,
    the data will be expanded into span level (1 row per span, with sample text repeated
    for each span row), in order to join the span-level embeddings/probs

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference. The name of the inference
        split to get data for.
    :param file_type: The file type to download the data as. Default arrow
    :param include_embs: Whether to include the embeddings in the data. Default False
    :param include_probs: Whether to include the probs in the data. Default False
    :param include_token_indices: (NER only) Whether to include logged
        text_token_indices in the data. Useful for reconstructing tokens for retraining
    :param hf_format: (NER only)
        Whether to export the dataframe in a HuggingFace compatible format
    :param tagging_schema: (NER only)
        If hf_format is True, you must pass a tagging schema
    :param as_pandas: Whether to return the dataframe as a pandas df (or vaex if False)
        If you are having memory issues (the data is too large), set this to False,
        and vaex will memory map the data. If any columns returned are multi-dimensional
        (embeddings, probabilities etc), vaex will always be returned, because pandas
        cannot support multi-dimensional columns. Default True
    :param include_data_embs: Whether to include the off the shelf data embeddings
    """
    split = conform_split(split)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)

    file_name = f"/tmp/{uuid4()}-data.{file_type}"
    api_client.export_edits(
        project_name,
        run_name,
        split,
        inference_name=inference_name,
        file_name=file_name,
        hf_format=hf_format,
        tagging_schema=tagging_schema,
    )
    data_df = vaex.open(file_name)
    return _process_exported_dataframe(
        data_df,
        project_name,
        run_name,
        split,
        task_type,
        inference_name,
        include_embs,
        include_probs,
        include_token_indices,
        hf_format,
        as_pandas,
        include_data_embs,
    )


@cached(_get_cache(), key=_cache_key)
def _process_exported_dataframe(
    data_df: DataFrame,
    project_name: str,
    run_name: str,
    split: Split,
    task_type: TaskType,
    inference_name: str = "",
    include_embs: bool = False,
    include_probs: bool = False,
    include_token_indices: bool = False,
    hf_format: bool = False,
    as_pandas: bool = True,
    include_data_embs: bool = False,
) -> Union[pd.DataFrame, DataFrame]:
    """Process dataframe after export of run or edits.

    See `get_dataframe` and `get_edited_dataframe` for details"""
    split = conform_split(split)
    # See docstring. In this case, we need span-level data
    # You can't attach embeddings/probs to the huggingface data, since the HF format is
    # sample level, and the embeddings are span level
    embs = include_embs or include_data_embs
    if (embs or include_probs) and task_type == TaskType.text_ner and not hf_format:
        # In NER, the `probabilities` contains the span level data
        span_df = get_probabilities(project_name, run_name, split, inference_name)
        keep_cols = [i for i in span_df.get_column_names() if "prob" not in i]
        span_df = span_df[keep_cols]
        # These are the token (not char) indices, lets make that clear
        span_df.rename("span_start", "span_token_start")
        span_df.rename("span_end", "span_token_end")

        for i in data_df.get_column_names():
            if i != "sample_id":
                data_df.rename(i, f"sample_{i}")
        data_df = data_df.join(span_df, on="sample_id", allow_duplication=True)

    tasks = []
    if task_type == TaskType.text_multi_label:
        tasks = api_client.get_tasks_for_run(project_name, run_name)
        labels = [
            api_client.get_labels_for_run(project_name, run_name, task)
            for task in tasks
        ]
        data_df = _index_df(data_df, labels, tasks)
        data_df = _clean_mltc_df(data_df)
    if task_type == TaskType.text_classification:
        labels_per_task = api_client.get_labels_for_run(project_name, run_name)
        data_df = _index_df(data_df, labels_per_task)

    if include_embs:
        # Embeddings are span level, but huggingface is sample level, so can't combine
        if hf_format:
            warnings.warn(
                "Embeddings are not available in HF format, ignoring", GalileoWarning
            )
        else:
            emb_df = get_embeddings(project_name, run_name, split, inference_name)
            data_df = data_df.join(emb_df, on="id")
    if include_data_embs:
        # Embeddings are span level, but huggingface is sample level, so can't combine
        if hf_format:
            warnings.warn(
                "Embeddings are not available in HF format, ignoring", GalileoWarning
            )
        else:
            emb_df = get_data_embeddings(
                project_name, run_name, split, inference_name
            ).copy()
            emb_df.rename("emb", "data_emb")
            data_df = data_df.join(emb_df, on="id")
    if include_probs:
        if hf_format:
            warnings.warn(
                "Probabilities are not available in HF format, ignoring", GalileoWarning
            )
        else:
            prob_df = get_probabilities(project_name, run_name, split, inference_name)
            # Includes `prob` for TC, `prob_#` for MLTC, and `conf/loss_prob` for NER
            prob_cols = prob_df.get_column_names(regex=r".*prob*") + ["id"]
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
            if "sample_id" in data_df.get_column_names():
                data_df.rename("sample_id", "id")
            data_df = data_df.join(raw_tokens, on="id")
    if as_pandas:
        # If any columns are multi-dimensional (embeddings, probs etc), must return vaex
        for col in data_df.get_column_names():
            if data_df[col].ndim > 1:
                return data_df
        pdf = data_df.to_pandas_df()
        # The spans come back as json.dumps string data, we can load it for our users
        # Back into JSON data so they get the actual span objects
        if task_type == TaskType.text_ner and "spans" in pdf.columns:
            pdf["spans"] = pdf["spans"].apply(json.loads)
        return pdf
    return data_df


def get_epochs(project_name: str, run_name: str, split: Split) -> List[int]:
    """Returns the epochs logged for a run/split

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    """
    split = conform_split(split)
    return api_client.get_epochs_for_run(project_name, run_name, split)


def get_embeddings(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    epoch: int = None,
) -> DataFrame:
    """Downloads the embeddings for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the embeddings from the final epoch. Note that only the
    n and n-1 epoch embeddings are available for download

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
    split = conform_split(split)
    return _get_hdf5_file_for_epoch(
        project_name,
        run_name,
        split,
        "emb/emb.hdf5",
        inference_name,
        epoch,
    )


def get_data_embeddings(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
) -> DataFrame:
    """Downloads the data (off the shelf) embeddings for a run/split

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference
    """
    split = conform_split(split)
    return _get_hdf5_file_for_epoch(
        project_name,
        run_name,
        split,
        "data_emb/data_emb.hdf5",
        inference_name,
    )


def get_probabilities(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    epoch: int = None,
) -> DataFrame:
    """Downloads the probabilities for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the probabilities from the final epoch.

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
    split = conform_split(split)
    return _get_hdf5_file_for_epoch(
        project_name, run_name, split, "prob/prob.hdf5", inference_name, epoch
    )


def get_raw_data(
    project_name: str,
    run_name: str,
    split: Split,
    inference_name: str = "",
    epoch: int = None,
) -> DataFrame:
    """Downloads the raw logged data for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the probabilities from the final epoch.

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param inference_name: Required if split is inference
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
    split = conform_split(split)
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    task_type = api_client.get_task_type(project_id, run_id)
    if task_type == TaskType.text_ner:
        object_name = "data/data.arrow"
    else:
        object_name = "data/data.hdf5"
    return _get_hdf5_file_for_epoch(
        project_name, run_name, split, object_name, inference_name, epoch
    )


def get_xray_cards(
    project_name: str, run_name: str, split: Split, inference_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """Get xray cards for a project/run/split

    Xray cards are automatic insights calculated and provided by Galileo on your data
    """
    split = conform_split(split)
    return api_client.get_xray_cards(project_name, run_name, split, inference_name)


def get_labels_for_run(
    project_name: str, run_name: str, task: Optional[str] = None
) -> List[str]:
    """Gets labels for a given run. If multi-label, a task must be provided"""
    return api_client.get_labels_for_run(project_name, run_name, task)


def get_tasks_for_run(project_name: str, run_name: str) -> List[str]:
    """Gets task names for a multi-label run"""
    return api_client.get_tasks_for_run(project_name, run_name)


@cached(_get_cache(), key=_cache_key)
def _get_hdf5_file_for_epoch(
    project_name: str,
    run_name: str,
    split: Split,
    object_name: str,
    inference_name: str = "",
    epoch: int = None,
) -> DataFrame:
    split = conform_split(split)
    emb = "emb" in object_name
    if split == Split.inference and inference_name:
        split_path = inference_name
    else:
        split_path = str(_validate_epoch(project_name, run_name, split, epoch, emb=emb))
    project_id, run_id = api_client._get_project_run_id(project_name, run_name)
    object_name = f"{project_id}/{run_id}/{split}/{split_path}/{object_name}"
    file_name = f"/tmp/{uuid4()}-{os.path.split(object_name)[-1]}"
    object_store.download_file(object_name, file_name)
    return vaex.open(file_name)


def _validate_epoch(
    project_name: str, run_name: str, split: Split, epoch: int = None, emb: bool = False
) -> int:
    split = conform_split(split)
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
            if col not in df.get_column_names():
                continue
            df_col = f"{col}_{task}" if task else col
            df[f"{df_col}_idx"] = df[df_col]
            df = df.ordinal_encode(f"{df_col}_idx", values=task_labels, lazy=True)
    return df


def _rename_prob_cols(df: DataFrame, tasks: List[str]) -> DataFrame:
    for ind, task in enumerate(tasks):
        df.rename(f"prob_{ind}", f"prob_{task}")
    return df


def _validate_filter(filter: Union[FilterParams, Dict] = None) -> Dict:
    # Validate the fields provided with pydantic before making request
    return FilterParams(**dict(filter or {})).dict()


def _conform_edit(edit: Union[Edit, Dict]) -> Edit:
    if isinstance(edit, Edit):
        return edit
    return Edit(**edit)


def _clean_mltc_df(df: DataFrame) -> DataFrame:
    """In MLTC, don't return the non-task-indexes gold/pred/dep columns"""
    drop_cols = ["gold", "pred", "data_error_potential", "prob"]
    keep_cols = [col for col in df.get_column_names() if col not in drop_cols]
    return df[keep_cols]
