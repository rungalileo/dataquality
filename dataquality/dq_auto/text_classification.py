from typing import List, Optional, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict

import dataquality as dq
from dataquality import Analytics, ApiClient
from dataquality.dq_auto.base_data_manager import BaseDatasetManager
from dataquality.dq_auto.tc_trainer import get_trainer
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import add_val_data_if_missing, run_name_from_hf_dataset
from dataquality.utils.auto_trainer import do_train

a = Analytics(ApiClient, dq.config)
a.log_import("auto_tc")


class TCDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = [
        "rungalileo/newsgroups",
        "rungalileo/trec6",
        "rungalileo/conv_intent",
        "rungalileo/emotion",
        "rungalileo/amazon_polarity_30k",
        "rungalileo/sst2",
    ]

    def _convert_pandas_object_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts columns of object type to string type for huggingface

        Huggingface DataSets cannot handle mixed-type columns as columns due to Arrow.
        This casts those columns to strings
        """
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("str")
        return df

    def _add_class_label_to_dataset(
        self, ds: Dataset, labels: List[str] = None
    ) -> Dataset:
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

    def _convert_df_to_dataset(
        self, df: pd.DataFrame, labels: List[str] = None
    ) -> Dataset:
        """Converts a pandas dataframe to a well-formed huggingface dataset

        The main thing happening here is that we are taking the pandas label column and
        mapping it to a Dataset ClassLabel if possible. If not, it will get parsed later
        or a validation error will be thrown if not possible in `_validate_dataset_dict`
        """
        df_copy = self._convert_pandas_object_dtype(df.copy())
        # If there's no label column, we can't do any ClassLabel conversions. Validation
        # of the hf DatasetDict will handle this missing label column if it's an
        # issue. See `_validate_dataset_dict`
        ds = Dataset.from_pandas(df_copy)
        return self._add_class_label_to_dataset(ds, labels)

    def _validate_dataset_dict(
        self, clean_dd: DatasetDict, labels: List[str] = None
    ) -> DatasetDict:
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
        clean_dd = super()._validate_dataset_dict(clean_dd, labels)
        for key in list(clean_dd.keys()):
            ds = clean_dd.pop(key)
            assert "text" in ds.features, "Dataset must have column `text`"
            assert "label" in ds.features, "Dataset must have column `label`"
            if "id" not in ds.features:
                ds = ds.add_column("id", list(range(ds.num_rows)))
            if not isinstance(ds.features["label"], ClassLabel):
                ds = self._add_class_label_to_dataset(ds, labels)
            clean_dd[key] = ds
        return add_val_data_if_missing(clean_dd)


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
    project_name: str = "auto_tc",
    run_name: str = None,
    wait: bool = True,
    create_data_embs: bool = False,
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
    :param val_data: Optional validation data to use. The validation data is what is
        used for the evaluation dataset in huggingface, and what is used for early
        stopping. If not provided, but test_data is, that will be used as the evaluation
        set. If neither val nor test are available, the train data will be randomly
        split 80/20 for use as evaluation data.
        Can be one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param test_data: Optional test data to use. The test data, if provided with val,
        will be used after training is complete, as the held-out set. If no validation
        data is provided, this will instead be used as the evaluation set.
        Can be one of
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
    :param create_data_embs: Whether to create data embeddings for this run. Default
        False

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
    from dataquality.auto.text_classification import auto

    auto(
         train_data="train.csv",
         test_data="test.csv",
         project_name="data_from_local",
         run_name="run_1_raw_data"
    )
    ```
    """
    a.log_function("auto/tc")
    manager = TCDatasetManager()
    dd = manager.get_dataset_dict(hf_data, train_data, val_data, test_data, labels)
    labels = _get_labels(dd, labels)
    dq.login()
    if not run_name and isinstance(hf_data, str):
        run_name = run_name_from_hf_dataset(hf_data)
    dq.init(TaskType.text_classification, project_name=project_name, run_name=run_name)
    dq.set_labels_for_run(labels)
    _log_dataset_dict(dd)
    trainer, encoded_data = get_trainer(dd, labels, hf_model, max_padding_length)
    do_train(trainer, encoded_data, wait, create_data_embs)
