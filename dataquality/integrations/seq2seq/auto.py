from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import Trainer

import dataquality as dq
from dataquality import Analytics, ApiClient
from dataquality.dq_auto.base_data_manager import BaseDatasetManager
from dataquality.integrations.seq2seq.s2s_trainer import do_train, get_trainer
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import (
    add_val_data_if_missing,
    run_name_from_hf_dataset,
)

a = Analytics(ApiClient, dq.config)
a.log_import("auto_s2s")


def format_alpaca(sample: Dict[str, str]) -> Dict[str, str]:
    """Formats the alpaca dataset for seq2seq

    Example:
        >>> sample = {
        ...     "instruction": "Summarize the following paragraph",
        ...     "input": "The quick brown fox jumped over the lazy dog.",
        ...     "target": "The quick brown fox jumped over the lazy dog.",
        ... }
        >>> format_alpaca(sample)
        {
            "formatted_input": (
                "Human: Summarize the following paragraph "
                "Context: The quick brown fox jumped over the lazy dog."
            )
        }
    """
    instruction = f"Human: {sample['instruction']}"
    # By multiplying by a bool, we only add the context if it exists
    context = f"Context: {sample['input']}" * bool(sample["input"])
    return {
        "formatted_input": f"{instruction} {context}",
    }


class S2SDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = [
        "tatsu-lab/alpaca",
        # "billsum",
    ]

    DATASET_FORMAT_FNS = {
        "tatsu-lab/alpaca": format_alpaca,
    }

    DATASET_COLS = {
        "tatsu-lab/alpaca": {
            "input": "formatted_input",
            "target": "target",
        }
    }

    def _validate_dataset_dict(
        self,
        dd: DatasetDict,
        inference_names: List[str],
        labels: Optional[List[str]] = None,
    ) -> DatasetDict:
        """Validates the core components of the provided (or created) DatasetDict

        The DatasetDict that the user provides or that we create from the provided
        train/test/val data must have the following:
            * all keys must be one of our valid key names
            * it must have a `text` column
            * it must have a `label` column
                * if the `label` column isn't a ClassLabel, we convert it to one

        We then also convert the keys of the DatasetDict to our `Split` key enum so
        we can access it easier in the future
        """
        clean_dd = super()._validate_dataset_dict(dd, inference_names, labels)
        for key in list(clean_dd.keys()):
            ds = clean_dd.pop(key)
            # TODO: temporary, update
            ds = ds.select(range(100))

            if "id" not in ds.features:
                ds = ds.add_column("id", list(range(ds.num_rows)))
            clean_dd[key] = ds
        return add_val_data_if_missing(clean_dd, TaskType.seq2seq)


def _get_labels(dd: DatasetDict, labels: Optional[List[str]] = None) -> List[str]:
    """Gets the labels for this dataset from the dataset if not provided."""
    if labels is not None and isinstance(labels, (list, np.ndarray)):
        return list(labels)
    train_labels = dd[Split.train].features["label"]
    if hasattr(train_labels, "names"):
        return train_labels.names
    return sorted(set(dd[Split.train]["label"]))


def _log_dataset_dict(
    dd: DatasetDict, input_col: str = "text", target_col: str = "label"
) -> None:
    for key in dd:
        ds = dd[key]
        if key in Split.get_valid_keys():
            dq.log_dataset(ds, text=input_col, label=target_col, split=key)


def auto(
    hf_data: Optional[Union[DatasetDict, str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    num_train_epochs: int = 3,
    hf_model: str = "google/flan-t5-base",
    project_name: str = "auto_s2s",
    run_name: Optional[str] = None,
    wait: bool = True,
    max_input_tokens: Optional[int] = None,
    max_target_tokens: Optional[int] = None,
    create_data_embs: Optional[bool] = None,
    generation_splits: Optional[List[str]] = None,
) -> Trainer:
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
    :param hf_model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default distilbert-base-uncased
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
    manager = S2SDatasetManager()
    dd = manager.get_dataset_dict(
        hf_data,
        train_data,
        val_data,
        test_data,
    )
    dq.login()
    a.log_function("auto/tc")
    if not run_name and isinstance(hf_data, str):
        run_name = run_name_from_hf_dataset(hf_data)

    dq.init(TaskType.seq2seq, project_name=project_name, run_name=run_name)

    # We 'watch' in get_trainer, which must happen before logging datasets
    model, tokenizer, dataloaders = get_trainer(
        dd,
        hf_model,
        max_input_tokens,
        max_target_tokens,
        generation_splits,
    )
    input_col = "text"
    target_col = "label"
    if not run_name and isinstance(hf_data, str):
        input_col = manager.DATASET_COLS.get(hf_data, {}).get("input") or input_col
        target_col = manager.DATASET_COLS.get(hf_data, {}).get("target") or target_col

    _log_dataset_dict(dd, input_col=input_col, target_col=target_col)
    return do_train(model, dataloaders, num_train_epochs, wait, create_data_embs)
