from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict

from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import get_task_type_from_data

AUTO_PROJECT_NAME = {
    TaskType.text_classification: "auto_tc",
    TaskType.text_ner: "auto_ner",
}


def auto(
    hf_data: Optional[Union[DatasetDict, str]] = None,
    hf_inference_names: Optional[List[str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    inference_data: Optional[Dict[str, Union[pd.DataFrame, Dataset, str]]] = None,
    max_padding_length: int = 200,
    hf_model: str = "distilbert-base-uncased",
    num_train_epochs: int = 15,
    labels: Optional[List[str]] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    wait: bool = True,
    create_data_embs: Optional[bool] = None,
    early_stopping: bool = True,
) -> None:
    """Automatically gets insights on a text classification or NER dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface transformer model, and
    provide Galileo insights via a link to the Galileo Console

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored.
    :param hf_inference_names: Use this param alongside `hf_data` if you have splits
        you'd like to consider as inference. A list of key names in `hf_data`
        to be run as inference runs after training. Any keys set must exist in `hf_data`
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
    :param inference_data: User this param to include inference data alongside the
        `train_data` param. If you are passing data via the `hf_data` parameter, you
        should use the `hf_inference_names` param. Optional inference datasets to run
        with after training completes. The structure is a dictionary with the
        key being the inference name and the value one of
        * Pandas dataframe
        * Huggingface dataset
        * Path to a local file
        * Huggingface dataset hub path
    :param max_padding_length: The max length for padding the input text
        during tokenization. Default 200
    :param hf_model: The pretrained AutoModel from huggingface that will be used to
        tokenize and train on the provided data. Default distilbert-base-uncased
    :param num_train_epochs: The number of epochs to train for (early stopping will
        always be active). Default 15
    :param labels: Optional list of labels for this dataset. If not provided, they
        will attempt to be extracted from the data
    :param project_name: Optional project name. If not set, a random name will
        be generated
    :param run_name: Optional run name for this data. If not set, a random name will
        be generated
    :param wait: Whether to wait for Galileo to complete processing your run.
        Default True
    :param create_data_embs: Whether to create data embeddings for this run. If True,
        Sentence-Transformers will be used to generate data embeddings for this dataset
        and uploaded with this run. You can access these embeddings via
        `dq.metrics.get_data_embeddings` in the `emb` column or
        `dq.metrics.get_dataframe(..., include_data_embs=True)` in the `data_emb` col
        Only available for TC currently. NER coming soon. Default True if a GPU is
        available, else default False.
    :param early_stopping: Whether to use early stopping. Default True

    For text classification datasets, the only required columns are `text` and `label`

    For NER, the required format is the huggingface standard format of `tokens` and
    `tags` (or `ner_tags`).
    See example: https://huggingface.co/datasets/rungalileo/mit_movies

        MIT Movies dataset in huggingface format

    .. code-block:: python

        tokens	                                            ner_tags
        [what, is, a, good, action, movie, that, is, r...	[0, 0, 0, 0, 7, 0, ...
        [show, me, political, drama, movies, with, jef...	[0, 0, 7, 8, 0, 0, ...
        [what, are, some, good, 1980, s, g, rated, mys...	[0, 0, 0, 0, 5, 6, ...
        [list, a, crime, film, which, director, was, d...	[0, 0, 7, 0, 0, 0, ...
        [is, there, a, thriller, movie, starring, al, ...	[0, 0, 0, 7, 0, 0, ...
        ...                                               ...                      ...


    To see auto insights on a random, pre-selected dataset, simply run

    .. code-block:: python

        import dataquality as dq

        dq.auto()


    An example using `auto` with a hosted huggingface text classification dataset

    .. code-block:: python

        import dataquality as dq

        dq.auto(hf_data="rungalileo/trec6")


    Similarly, for NER

    .. code-block:: python

        import dataquality as dq

        dq.auto(hf_data="conll2003")


    An example using `auto` with sklearn data as pandas dataframes

    .. code-block:: python

        import dataquality as dq
        import pandas as pd
        from sklearn.datasets import fetch_20newsgroups

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

        dq.auto(
             train_data=df_train,
             test_data=df_test,
             labels=newsgroups_train.target_names,
             project_name="newsgroups_work",
             run_name="run_1_raw_data"
        )


    An example of using `auto` with a local CSV file with `text` and `label` columns

    .. code-block:: python

        import dataquality as dq

        dq.auto(
            train_data="train.csv",
            test_data="test.csv",
            project_name="data_from_local",
            run_name="run_1_raw_data"
        )
    """
    import datasets
    import transformers

    transformers.logging.enable_progress_bar()
    datasets.logging.enable_progress_bar()

    # We need to import auto down here instead of at the top of the file like normal
    # because we simultaneously want analytic tracking on the files we import while
    # wanting dq.auto as a top level function. If we have these imports at the top,
    # and make dq.auto() available, then auto_tc and auto_ner will both always be
    # imported as soon as dataquality is imported. Also, transformers_trainer and
    # pytorch (which auto depends on) will be immediately imported. The only way to
    # avoid that is by having the imports only be made selectively when auto is called
    if hf_data is None and train_data is None:
        from dataquality.dq_auto.text_classification import auto as auto_tc

        auto_tc(num_train_epochs=num_train_epochs)
        return
    task_type = get_task_type_from_data(hf_data, train_data)
    # We cannot use a common list of *args or **kwargs here because mypy screams :(
    if task_type == TaskType.text_classification:
        from dataquality.dq_auto.text_classification import auto as auto_tc

        return auto_tc(
            hf_data=hf_data,
            hf_inference_names=hf_inference_names,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            inference_data=inference_data,
            max_padding_length=max_padding_length,
            hf_model=hf_model,
            labels=labels,
            project_name=project_name or AUTO_PROJECT_NAME[task_type],
            run_name=run_name,
            wait=wait,
            create_data_embs=create_data_embs,
            num_train_epochs=num_train_epochs,
            early_stopping=early_stopping,
        )
    elif task_type == TaskType.text_ner:
        from dataquality.dq_auto.ner import auto as auto_ner

        return auto_ner(
            hf_data=hf_data,
            hf_inference_names=hf_inference_names,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            inference_data=inference_data,
            hf_model=hf_model,
            labels=labels,
            project_name=project_name or AUTO_PROJECT_NAME[task_type],
            run_name=run_name,
            wait=wait,
            num_train_epochs=num_train_epochs,
            early_stopping=early_stopping,
        )
    else:
        raise Exception("auto is only supported for text classification and NER!")
