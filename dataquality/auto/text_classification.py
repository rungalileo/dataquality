import os
import webbrowser
from random import choice
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, load_dataset

import dataquality as dq
from dataquality.auto.tc_trainer import get_trainer
from dataquality.exceptions import GalileoException
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType

DEMO_DATASETS = [
    "rungalileo/newsgroups",
    "rungalileo/trec6",
    "rungalileo/conv_intent",
    "rungalileo/emotion",
    "rungalileo/amazon_polarity_30k",
    "rungalileo/sst2",
]


def _convert_pandas_object_dtype(df: pd.DataFrame) -> pd.DataFrame:
    """Converts columns of object type to string type for huggingface

    Huggingface DataSets cannot handle mixed-type columns as columns due to Arrow. This
    casts those columns to strings
    """
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("str")
    return df


def _convert_df_to_dataset(df: pd.DataFrame, labels: List[str] = None) -> Dataset:
    """Converts a pandas dataframe to a well-formed huggingface dataset

    The main thing happening here is that we are taking the pandas label column and
    mapping it to a Dataset ClassLabel if possible. If not, it will get parsed later
    or a validation error will be thrown if not possible in `_validate_dataset_dict`
    """
    df_copy = _convert_pandas_object_dtype(df.copy())
    # If there's no label column, we can't do any ClassLabel conversions. Validation
    # of the huggingface DatasetDict will handle this missing label column if it's an
    # issue. See `_validate_dataset_dict`
    ds = Dataset.from_pandas(df_copy)
    return _add_class_label_to_dataset(ds, labels)


def _add_class_label_to_dataset(ds: Dataset, labels: List[str] = None) -> Dataset:
    """Map a not ClassLabel 'label' column to a ClassLabel, if possible"""
    if "label" not in ds.features or isinstance(ds.features["label"], ClassLabel):
        return ds
    labels = labels if labels is not None else sorted(set(ds["label"]))
    # For string columns, map the label2idx so we can cast to ClassLabel
    if ds.features["label"].dtype == "string":
        label_to_idx = dict(zip(labels, range(len(labels))))
        ds = ds.map(lambda row: {"label": label_to_idx[row["label"]]})

    # https://github.com/python/mypy/issues/6239
    class_label = ClassLabel(num_classes=len(labels), names=labels)  # type: ignore
    ds = ds.cast_column("label", class_label)
    return ds


def _convert_to_hf_dataset(
    data: Union[pd.DataFrame, Dataset, str], labels: List[str] = None
) -> Dataset:
    """Loads the data into (hf) Dataset format.

    Data can be one of Dataset, pandas df, str. If str, it's either a path to a local
    file or a path to a remote huggingface Dataset that we load with `load_dataset`
    """
    if isinstance(data, Dataset):
        return data
    if isinstance(data, pd.DataFrame):
        return _convert_df_to_dataset(data, labels)
    if isinstance(data, str):
        # If there is no file extension, it's a huggingface Dataset, so we load it from
        # huggingface hub
        ext = os.path.splitext(data)[-1]
        if not ext:
            ds = load_dataset(data)
        else:
            # .csv -> read_csv, .parquet -> read_parquet
            func = f"read_{ext.lstrip('.')}"
            if not hasattr(pd, func):
                raise GalileoException(
                    "Local file path extension must be readable by panda. "
                    f"Found {ext} which is not"
                )
            df = getattr(pd, func)(data)
            ds = _convert_df_to_dataset(df, labels)
        assert isinstance(ds, Dataset), (
            f"Loaded data should be of type Dataset, but found {type(ds)}. If ds is a "
            f"DatasetDict, consider passing it to `hf_data` (dq.auto(hf_data=data))"
        )
        return ds
    raise GalileoException(
        "Dataset must be one of pandas df, huggingface Dataset, or string path"
    )


def _validate_dataset_dict(dd: DatasetDict, labels: List[str] = None) -> DatasetDict:
    """Validates the core components of the provided (or created) DatasetDict)

    The DatasetDict that the user provides or that we create from the provided
    train/test/val data must have the following:
        * all keys must be one of our valid key names
        * it must have a `text` column
        * it must have a `label` column
            * if the `label` column isn't a ClassLabel, we convert it to one

    We then also convert the keys of the DatasetDict to our `Split` key enum so
    we can access it easier in the future
    """
    valid_keys = Split.get_valid_keys()
    for key in list(dd.keys()):
        assert (
            key in valid_keys
        ), f"All keys of dataset must be one of {valid_keys}. Found {list(dd.keys())}"
        ds = dd.pop(key)
        assert "text" in ds.features, "Dataset must have column `text`"
        assert "label" in ds.features, "Dataset must have column `label`"
        if "id" not in ds.features:
            ds = ds.add_column("id", list(range(ds.num_rows)))
        if not isinstance(ds.features["label"], ClassLabel):
            ds = _add_class_label_to_dataset(ds, labels)
        # Use the split Enums
        dd[Split[key]] = ds
    return dd


def _get_dataset_dict(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    labels: List[str] = None,
) -> DatasetDict:
    """Creates and/or validates the DatasetDict provided by the user.

    If the user provides a DatasetDict, we simply validate it. Otherwise, we
    parse a combination of the parameters provided, generate a DatasetDict of their
    training data, and validate that.
    """
    dd = DatasetDict()
    if all([hf_data is None, train_data is None]):
        hf_data = choice(DEMO_DATASETS)
        print(f"No dataset provided, using {hf_data} for run")
    if hf_data:
        ds = load_dataset(hf_data) if isinstance(hf_data, str) else hf_data
        assert isinstance(ds, DatasetDict), (
            "hf_data must be a path to a huggingface DatasetDict in the hf hub or a "
            "DatasetDict object. If this is just a Dataset, pass it to `train_data`"
        )
        dd = ds
    else:
        dd[Split.train] = _convert_to_hf_dataset(train_data, labels)
        if val_data is not None:
            dd[Split.validation] = _convert_to_hf_dataset(val_data, labels)
        if test_data is not None:
            dd[Split.test] = _convert_to_hf_dataset(test_data, labels)
    return _validate_dataset_dict(dd, labels)


def _get_labels(dd: DatasetDict, labels: Optional[List[str]] = None) -> List[str]:
    """Gets the labels for this dataset from the dataset if not provided."""
    if labels is not None and isinstance(labels, (list, np.ndarray)):
        return list(labels)
    train_labels = dd[Split.train].features["label"]
    if hasattr(train_labels, "names"):
        return train_labels.names
    return sorted(set(dd[Split.train]["label"]))


def _log_dataset_dict(dd: DatasetDict) -> None:
    for key in dd:
        ds = dd[key]
        default_cols = ["text", "label", "id"]
        meta = [i for i in ds.features if i not in default_cols]
        dq.log_dataset(ds, meta=meta, split=key)


def auto(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    max_padding_length: int = 200,
    hf_model: str = "distilbert-base-uncased",
    labels: List[str] = None,
    project_name: str = None,
    run_name: str = None,
    wait: bool = True,
    _evaluation_metric: str = "accuracy",
) -> None:
    """Automatically gets insights on a text classification dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface transformer model, and
    provide Galileo insights via a link to the Galileo Console

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored
    :param train_data: Optional training data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param val_data: Optional validation data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param test_data: Optional test data to use. Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param max_padding_length: The max length for padding the input text
        during tokenization. Default 200
    :param hf_model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default distilbert-base-uncased
    :param labels: Optional list of labels for this dataset. If not provided, they
        will attempt to be extracted from the data
    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True
    :param _evaluation_metric: The metric to set for huggingface evaluation.
        This will simply control the metric huggingface uses to evaluate model
        performance.

    To see auto insights on a random, pre-selected dataset, simply run
    ```python
        from dataquality.auto.text_classification import auto

        auto()
    ```

    An example using `auto` with a hosted huggingface dataset
    ```python
        from dataquality.auto.text_classification import auto

        auto(hf_data="rungalileo/trec6")
    ```

    An example using `auto` with sklearn data as pandas dataframes
    ```python
        import pandas as pd
        from sklearn.datasets import fetch_20newsgroups
        from dataquality.auto.text_classification import auto

        # Load the newsgroups dataset from sklearn
        newsgroups_train = fetch_20newsgroups(subset='train')
        newsgroups_test = fetch_20newsgroups(subset='test')
        # Convert to pandas dataframes
        df_train = pd.DataFrame(
            {"text": newsgroups_train.data, "label": newsgroups_train.target}
        )
        df_test = pd.DataFrame(
            {"text": newsgroups_test.data, "label": newsgroups_test.target}
        )

        auto(
             train_data=df_train,
             test_data=df_test,
             labels=newsgroups_train.target_names,
             project_name="newsgroups_work",
             run_name="run_1_raw_data"
        )
    ```

    An example of using `auto` with a local CSV file with `text` and `label` columns
    ```python
    import pandas as pd
    from dataquality.auto.text_classification import auto

    auto(
         train_data="train.csv",
         test_data="test.csv",
         labels=newsgroups_train.target_names,
         project_name="newsgroups_work",
         run_name="run_1_raw_data"
    )
    ```
    """
    dd = _get_dataset_dict(hf_data, train_data, val_data, test_data, labels)
    labels = _get_labels(dd, labels)
    dq.login()
    dq.init(TaskType.text_classification, project_name=project_name, run_name=run_name)
    dq.set_labels_for_run(labels)
    _log_dataset_dict(dd)
    trainer, encoded_data = get_trainer(
        dd, labels, hf_model, max_padding_length, _evaluation_metric
    )
    watch(trainer)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore
    res = dq.finish(wait=wait) or {}
    # Try to open the console URL for them (won't work in colab)
    link = res.get("link")
    if link:
        try:
            webbrowser.open(link)
        except Exception:
            print(f"Click here to see your run! {link}")
