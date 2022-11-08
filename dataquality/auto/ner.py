import webbrowser
from typing import List, Union

import pandas as pd
from datasets import Dataset, DatasetDict

import dataquality as dq
from dataquality.auto.ner_trainer import get_trainer
from dataquality.exceptions import GalileoException
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import load_data_from_str, try_load_dataset_dict

DEMO_DATASETS = ["rungalileo/mit_movies", "rungalileo/wikiner_it", "wnut_17"]


def _convert_dataset_to_hf_format(ds: Dataset) -> Dataset:
    """Converts the dataset to huggingface if necessary"""
    # Already in hf format
    if "tokens" in ds.features and ("tags" in ds.features or "ner_tags" in ds.features):
        return ds
    if "text" not in ds.features or "spans" not in ds.features:
        raise GalileoException(
            "Data must be in either huggingface format "
            "(`tokens` and `tags`/`ner_tags`) or spacy format (`text` and `spans`). "
            "See help(auto) for more details and examples"
        )
    raise GalileoException("TODO support spacy format")
    # TODO: get code from @nikita-galileo for spacy -> huggingface


def _convert_to_hf_dataset(data: Union[pd.DataFrame, Dataset, str]) -> Dataset:
    """Loads the data into a huggingface DataSet.

    Data can be one of Dataset, pandas df, str. If str, it's either a path to a local
    file or a path to a remote huggingface Dataset that we load with `load_dataset`
    """
    if isinstance(data, Dataset):
        ds = data
    elif isinstance(data, pd.DataFrame):
        ds = Dataset.from_pandas(data)
    elif isinstance(data, str):
        ds = load_data_from_str(data)
        if isinstance(ds, pd.DataFrame):
            ds = Dataset.from_pandas(ds)
    else:
        raise GalileoException(
            "Dataset must be one of pandas df, huggingface Dataset, or string path"
        )
    return ds


def _validate_dataset_dict(dd: DatasetDict) -> DatasetDict:
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
    valid_keys = Split.get_valid_keys()
    for key in list(dd.keys()):
        assert (
            key in valid_keys
        ), f"All keys of dataset must be one of {valid_keys}. Found {list(dd.keys())}"
        ds = dd.pop(key)
        ds = _convert_dataset_to_hf_format(ds)
        # Use the split Enums
        dd[Split[key]] = ds
    return dd


def _get_dataset_dict(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
) -> DatasetDict:
    """Creates and/or validates the DatasetDict provided by the user.

    If the user provides a DatasetDict, we simply validate it. Otherwise, we
    parse a combination of the parameters provided, generate a DatasetDict of their
    training data, and validate that.
    """
    dd = try_load_dataset_dict(DEMO_DATASETS, hf_data, train_data) or DatasetDict()
    if not dd:
        dd[Split.train] = _convert_to_hf_dataset(train_data)
        if val_data is not None:
            dd[Split.validation] = _convert_to_hf_dataset(val_data)
        if test_data is not None:
            dd[Split.test] = _convert_to_hf_dataset(test_data)
    return _validate_dataset_dict(dd)


def auto(
    hf_data: Union[DatasetDict, str] = None,
    train_data: Union[pd.DataFrame, Dataset, str] = None,
    val_data: Union[pd.DataFrame, Dataset, str] = None,
    test_data: Union[pd.DataFrame, Dataset, str] = None,
    hf_model: str = "distilbert-base-uncased",
    labels: List[str] = None,
    project_name: str = None,
    run_name: str = None,
    wait: bool = True,
    _evaluation_metric: str = "seqeval",
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
    dd = _get_dataset_dict(hf_data, train_data, val_data, test_data)
    dq.login()
    dq.init(TaskType.text_ner, project_name=project_name, run_name=run_name)
    trainer, encoded_data = get_trainer(dd, hf_model, _evaluation_metric, labels)
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
