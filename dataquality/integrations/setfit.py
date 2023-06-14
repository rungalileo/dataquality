import contextlib
import io
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset, DatasetDict

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.dq_auto.text_classification import (
    TCDatasetManager,
    _get_labels,
    _log_dataset_dict,
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import run_name_from_hf_dataset
from dataquality.utils.patcher import PatchManager
from dataquality.utils.setfit import (
    SetFitModelHook,
    _apply_column_mapping,
    _prepare_config,
    _setup_patches,
    get_trainer,
    log_preds_setfit,
    validate_setfit,
)

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("setfit")


if TYPE_CHECKING:
    from setfit import SetFitModel, SetFitTrainer


def unwatch(setfit_obj: Optional[Union["SetFitModel", "SetFitTrainer"]]) -> None:
    """Unpatch SetFit model by replacing predict_proba function with original
    function.
    :param setfit_obj: SetFitModel or SetFitTrainer
    """
    a.log_function("setfit/unwatch")
    setfitmanager = PatchManager()
    setfitmanager.unpatch()
    helper_data = dq.get_data_logger().logger_config.helper_data
    if helper_data:
        helper_data.clear()


def watch(
    setfit: Union["SetFitModel", "SetFitTrainer"],
    labels: Optional[List[str]] = None,
    project_name: str = "",
    run_name: str = "",
    finish: bool = True,
    wait: bool = False,
    batch_size: Optional[int] = None,
    meta: Optional[List] = None,
    validate_before_training: bool = True,
) -> Callable:
    """Watch a SetFit model or trainer and extract model outputs for dataquality.
    Returns a function that can be used to evaluate the model on a dataset.
    :param setfit: SetFit model or trainer
    :param labels: list of labels
    :param project_name: name of project
    :param run_name: name of run
    :param finish: whether to run dq.finish after evaluation
    :param wait: whether to wait for dq.finish
    :param batch_size: batch size for evaluation
    :return: dq_evaluate function
    """
    a.log_function("setfit/watch")

    from setfit import SetFitTrainer

    if not dq.config.task_type:
        init_kwargs: Dict[str, Any] = {}
        if project_name:
            init_kwargs["project_name"] = project_name
        if run_name:
            init_kwargs["run_name"] = run_name
        dq.init("text_classification", **init_kwargs)
        print("dataquality initialized on SetFitTrainer/SetFitModel")

    labels = labels or dq.get_data_logger().logger_config.labels
    _prepare_config()
    if isinstance(setfit, SetFitTrainer):
        if validate_before_training:
            f_err = io.StringIO()
            f_out = io.StringIO()
            print("Validating SetFit model before training...")
            with contextlib.redirect_stderr(f_err), contextlib.redirect_stdout(f_out):
                validate_setfit(
                    setfit,
                    labels,
                    batch_size=batch_size,
                    meta=meta,
                )

        _setup_patches(
            setfit,
            labels,
            finish=finish,
            wait=wait,
            batch_size=batch_size,
            meta=meta,
        )
        return evaluate(setfit.model)
    else:
        model = setfit
        assert labels and len(
            labels
        ), "Labels must be set (watch(trainer, labels=[...]))"
        dq.set_labels_for_run(labels)
        return evaluate(model)


def evaluate(
    model: "SetFitModel",
) -> Callable:
    """Watch SetFit model by replacing predict_proba function with SetFitModelHook.
    :param model: SetFit model
    :return: SetFitModelHook object"""
    dq_hook = SetFitModelHook(model)
    dq_store = dq_hook.store

    def dq_evaluate(
        dataset: Dataset,
        split: Split,
        meta: Optional[List] = None,
        inference_name: Optional[str] = None,
        column_mapping: Optional[Dict] = None,
        batch_size: int = 64,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate SetFit model and log input and output to Galileo.
        :param batch: batch of data as a dictionary
        :param split: split of data (training, validation, test, inference)
        :param meta: columns that should be logged as metadata
        :param inference_name: inference name (if split is inference, must be provided)
        :param column_mapping: mapping of column names (if different from default)
        :return: output of SetFitModel.predict_proba function"""
        a.log_function("setfit/evaluate")

        column_mapping = column_mapping or dict(
            text="text",
            id="id",
            label="label",
        )

        if column_mapping is not None:
            dataset = _apply_column_mapping(dataset, column_mapping)
        if "id" not in dataset.features:
            dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True)
        if epoch is not None:
            dq.set_epoch(epoch)
        cur_epoch = get_data_logger().logger_config.cur_epoch
        return log_preds_setfit(
            model=model,
            dataset=dataset,
            dq_store=dq_store,
            batch_size=batch_size,
            split=split,
            inference_name=inference_name,
            meta=meta,
            epoch=cur_epoch,
        )

    return dq_evaluate


def auto(
    hf_data: Optional[Union[DatasetDict, str]] = None,
    hf_inference_names: Optional[List[str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    inference_data: Optional[Dict[str, Union[pd.DataFrame, Dataset, str]]] = None,
    max_padding_length: int = 200,
    num_train_epochs: int = 15,
    hf_model: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    labels: Optional[List[str]] = None,
    project_name: str = "auto_tc",
    run_name: Optional[str] = None,
    wait: bool = True,
    create_data_embs: Optional[bool] = None,
) -> "SetFitTrainer":
    """Automatically gets insights on a text classification dataset

    Given either a pandas dataframe, file_path, or huggingface dataset path, this
    function will load the data, train a huggingface transformer model, and
    provide Galileo insights via a link to the Galileo Console

    One of `hf_data`, `train_data` should be provided. If neither of those are, a
    demo dataset will be loaded by Galileo for training.

    :param hf_data: Union[DatasetDict, str] Use this param if you have huggingface
        data in the hub or in memory. Otherwise see `train_data`, `val_data`,
        and `test_data`. If provided, train_data, val_data, and test_data are ignored
    :param hf_inference_names: A list of key names in `hf_data` to be run as inference
        runs after training. If set, those keys must exist in `hf_data`
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
    :param inference_data: Optional inference datasets to run with after training
        completes. The structure is a dictionary with the key being the infeerence name
        and the value one of
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
    manager = TCDatasetManager()
    dd = manager.get_dataset_dict(
        hf_data,
        hf_inference_names,
        train_data,
        val_data,
        test_data,
        inference_data,
        labels,
    )
    labels = _get_labels(dd, labels)
    dq.login()
    a.log_function("auto/tc")
    if not run_name and isinstance(hf_data, str):
        run_name = run_name_from_hf_dataset(hf_data)
    dq.init(TaskType.text_classification, project_name=project_name, run_name=run_name)
    dq.set_labels_for_run(labels)
    _log_dataset_dict(dd)
    trainer, encoded_data = get_trainer(
        dd, labels, hf_model, max_padding_length, num_train_epochs
    )
    return do_train(trainer, encoded_data, wait, create_data_embs)


def do_train(
    trainer: "SetFitTrainer",
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> "SetFitTrainer":
    dq_evaluate = watch(trainer, finish=False)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        dq_evaluate(
            encoded_data[Split.test],
            split=Split.test,
            # for inference set the split to inference
            # and pass an inference_name="inference_run_1"
        )

    inf_names = [k for k in encoded_data if k not in Split.get_valid_keys()]
    for inf_name in inf_names:
        dq_evaluate(
            encoded_data[inf_name],
            split=Split.inference,  # type: ignore
            inference_name=inf_name,  # type: ignore
        )

    dq.finish(wait=wait, create_data_embs=create_data_embs)
    return trainer
