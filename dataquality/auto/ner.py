from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict

import dataquality as dq
from dataquality import Analytics, ApiClient
from dataquality.auto.base_data_manager import BaseDatasetManager
from dataquality.auto.ner_trainer import get_trainer
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import (
    add_val_data_if_missing,
    do_train,
    run_name_from_hf_dataset,
)

a = Analytics(ApiClient, dq.config)
a.log_import("auto_ner")


class NERDatasetManager(BaseDatasetManager):
    DEMO_DATASETS = ["rungalileo/mit_movies", "rungalileo/wikiner_it", "wnut_17"]

    def _convert_dataset_to_hf_format(self, ds: Dataset) -> Dataset:
        """Converts the dataset to huggingface if necessary"""
        # Already in hf format
        if "tokens" in ds.features and (
            "tags" in ds.features or "ner_tags" in ds.features
        ):
            return ds
        if "text" not in ds.features or "spans" not in ds.features:
            raise GalileoException(
                "Data must be in either huggingface format "
                "(`tokens` and `tags`/`ner_tags`) or spacy format "
                "(`text` and `spans`). See help(auto) for more details and examples"
            )
        raise GalileoException("TODO support spacy format")
        # TODO: get code from @nikita-galileo for spacy -> huggingface

    def _validate_dataset_dict(
        self, dd: DatasetDict, labels: Optional[List[str]] = None
    ) -> DatasetDict:
        """Validates the core components of the provided (or created) DatasetDict)

        If in spacy format, this will convert it to huggingface format. See `auto` for
        details.

        The DatasetDict that the user provides or that we create from the provided
        train/test/val data must have the following:
            * all keys must be one of our valid key names
            * it must have a `tokens` column
            * it must have a `tags` or `ner_tags column

        We then also convert the keys of the DatasetDict to our `Split` key enum so
        we can access it easier in the future
        """
        super()._validate_dataset_dict(dd, labels)
        for key in list(dd.keys()):
            ds = dd.pop(key)
            ds = self._convert_dataset_to_hf_format(ds)
            # Use the split Enums
            dd[Split[key]] = ds
        return add_val_data_if_missing(dd)


def auto(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    hf_model: str = "distilbert-base-uncased",
    labels: List[str] = None,
    project_name: str = "auto_ner",
    run_name: str = None,
    wait: bool = True,
) -> None:
    """Automatically gets insights on an NER or Token Classification dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface token classification model, and
    provide Galileo insights via a link to the Galileo Console

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    The data can be provided in 1 of 2 ways:
    * `huggingface` format: A dataset with `tokens` and (`ner_tags` or `tags`) columns
        See example: https://huggingface.co/datasets/rungalileo/mit_movies

        MIT Movies dataset in huggingface format

        tokens	                                            ner_tags
        [what, is, a, good, action, movie, that, is, r...	[0, 0, 0, 0, 7, 0, ...
        [show, me, political, drama, movies, with, jef...	[0, 0, 7, 8, 0, 0, ...
        [what, are, some, good, 1980, s, g, rated, mys...	[0, 0, 0, 0, 5, 6, ...
        [list, a, crime, film, which, director, was, d...	[0, 0, 7, 0, 0, 0, ...
        [is, there, a, thriller, movie, starring, al, ...	[0, 0, 0, 7, 0, 0, ...
        ...                                               ...                      ...

    * `spacy` format: This is the classic NER format for spacy models. Two columns
        `text` and `spans` are required. `spans` is a JSON object as a list of spans
        with fields `start`, `end`, and `label`

        MIT Movies dataset in spacy format
        text	                                            spans
        what is a good action movie that is rated pg 1...	[{'start': 15, 'end': 21, 'label': 'GENRE'}, {...  # noqa: 3501
        show me political drama movies with jeff danie...	[{'start': 8, 'end': 23, 'label': 'GENRE'}, {'...  # noqa: 3501
        what are some good 1980 s g rated mystery movi...	[{'start': 19, 'end': 25, 'label': 'YEAR'}, {'...  # noqa: 3501
        list a crime film which director was david lean .	[{'start': 7, 'end': 12, 'label': 'GENRE'}, {'...  # noqa: 3501
        is there a thriller movie starring al pacino .	    [{'start': 11, 'end': 19, 'label': 'GENRE'}, {...  # noqa: 3501
        ...	                                                ...


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
    :param labels: Optional list of labels for this dataset. If not provided, they
        will attempt to be extracted from the data
    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True

    To see auto insights on a random, pre-selected dataset, simply run
    ```python
        from dataquality.auto.ner import auto

        auto()
    ```

    An example using `auto` with a hosted huggingface dataset
    ```python
        from dataquality.auto.text_classification import auto

        auto(hf_data="rungalileo/mit_movies")
    ```

    An example using `auto` with sklearn data as pandas dataframes
    ```python
        import pandas as pd
        from dataquality.auto.ner import auto

        TODO EXAMPLE FOR NER FROM PANDAS DFs

        auto(
             train_data=df_train,
             test_data=df_test,
             labels=TODO,
             project_name="TODO",
             run_name="run_1_raw_data"
        )
    ```

    An example of using `auto` with a local CSV file with `text` and `label` columns
    ```python
    from dataquality.auto.ner import auto

    auto(
         train_data="train.csv",
         test_data="test.csv",
         project_name="data_from_local",
         run_name="run_1_raw_data"
    )
    ```
    """
    a.log_function("auto/ner")
    manager = NERDatasetManager()
    dd = manager.get_dataset_dict(hf_data, train_data, val_data, test_data)
    dq.login()
    if not run_name and isinstance(hf_data, str):
        run_name = run_name_from_hf_dataset(hf_data)
    dq.init(TaskType.text_ner, project_name=project_name, run_name=run_name)
    trainer, encoded_data = get_trainer(dd, hf_model, labels)
    do_train(trainer, encoded_data, wait)
