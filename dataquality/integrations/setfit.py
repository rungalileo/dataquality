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
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auto import _apply_column_mapping, run_name_from_hf_dataset
from dataquality.utils.patcher import PatchManager
from dataquality.utils.setfit import (
    SetFitModelHook,
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
    validate_before_training: bool = False,
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
    :param meta: meta data for evaluation
    :param validate_before_training: whether to do a testrun before training
    :return: dq_evaluate function
    """
    a.log_function("setfit/watch")

    from setfit import SetFitTrainer

    pm = PatchManager()
    pm.unpatch()
    # If dq.init has been previously called, we don't need to call it again
    # To detect this we check the paramater and the dq.config.task_type and
    # no project_name
    if project_name or dq.config.task_type != TaskType.text_classification:
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
    setfit_model: Union[
        "SetFitModel", str
    ] = "sentence-transformers/paraphrase-mpnet-base-v2",
    hf_data: Optional[Union[DatasetDict, str]] = None,
    hf_inference_names: Optional[List[str]] = None,
    train_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    val_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    test_data: Optional[Union[pd.DataFrame, Dataset, str]] = None,
    inference_data: Optional[Dict[str, Union[pd.DataFrame, Dataset, str]]] = None,
    labels: Optional[List[str]] = None,
    project_name: str = "auto_tc_setfit",
    run_name: Optional[str] = None,
    training_args: Optional[Dict[str, Any]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    wait: bool = True,
    create_data_embs: Optional[bool] = None,
) -> Union["SetFitModel", "SetFitTrainer"]:
    """Automatically processes and generates insights on a text classification dataset.

    Given a pandas dataframe, a file path, or a Huggingface dataset path, this
    function will load the data, train a Huggingface transformer model, and
    provide insights via a link to the Console.

    At least one of `hf_data`, `train_data` should be provided. If neither of
    those are, a demo dataset will be used for training.

    Parameters
    ----------
    setfit : SetFitModel or Huggingface model name
        Computes text embeddings for a given text dataset with the model.
        If a string is provided, it will be used to load a Huggingface model
        and train it on the data.
    hf_data : Union[DatasetDict, str], optional
        Use this parameter if you have Huggingface data in the hub or in memory.
        Otherwise see `train_data`, `val_data`, and `test_data`. If provided,
        train_data, val_data, and test_data are ignored.
    hf_inference_names : list of str, optional
        A list of key names in `hf_data` to be run as inference
        runs after training. If set, those keys must exist in `hf_data`.
    train_data : pandas.DataFrame, Dataset, str, optional
        Training data to use. Can be a pandas dataframe, a Huggingface dataset,
        path to a local file, or Huggingface dataset hub path.
    val_data : pandas.DataFrame, Dataset, str, optional
        Validation data to use for evaluation and early stopping. If not provided,
        but test_data is, that will be used as the evaluation set. If neither val_data
        nor test_data are available, the train data will be split randomly in
        80/20 ratio.
    test_data : pandas.DataFrame, Dataset, str, optional
        Test data to use. If provided with val_data, will be used after training
        is complete,as the held-out set. If no validation data is provided,
        this will instead be used as the evaluation set.
    inference_data : dict, optional
        Optional inference datasets to run after training. The structure is a dictionary
        with the key being the inference name and the value being a pandas dataframe, a
        Huggingface dataset, path to a local file, or Huggingface dataset hub path.
    labels : list of str, optional
        List of labels for this dataset. If not provided, they will attempt to
        be extracted from the data.
    project_name : str, optional
        Project name. If not set, a random name will be generated.
        Default is "auto_tc_setfit".
    run_name : str, optional
        Run name for this data. If not set, a random name will be generated.
    training_args : dict, optional
        A dictionary of arguments for the SetFitTrainer. It allows you
        to customize training configuration such as learning rate,
        batch size, number of epochs, etc.
    column_mapping : dict, optional
        A dictionary of column names to use for the provided data.
        Needs to map to the following keys: "text", "id", "label".
    wait : bool, optional
        Whether to wait for the processing of your run to complete. Default is True.
    create_data_embs : bool, optional
        Whether to create data embeddings for this run. Default is None.

    Returns
    -------
    SetFitModel or SetFitTrainer
        A SetFitTrainer instance trained on the provided dataset.

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

        auto(model=model,
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
         setfit_model="sentence-transformers/paraphrase-mpnet-base-v2",
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
        column_mapping,
    )
    labels = _get_labels(dd, labels)
    dq.login()
    a.log_function("setfit/auto")

    if not run_name:
        run_name = run_name_from_hf_dataset(hf_data or "setfit_auto")
    dq.init(TaskType.text_classification, project_name=project_name, run_name=run_name)
    dq.set_labels_for_run(labels)
    if isinstance(setfit_model, str):
        # Load the model and train it
        trainer, encoded_data = get_trainer(dd, setfit_model, training_args)
        return do_train(
            trainer,
            encoded_data,
            wait,
            create_data_embs,
        )
    else:
        # Don't train, just evaluate
        return do_model_eval(setfit_model, dd, wait, create_data_embs)


def do_model_eval(
    model: "SetFitModel",
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> "SetFitModel":
    dq_evaluate = watch(
        model,
        finish=False,
    )
    for split in [Split.train, Split.test, Split.val]:
        if split in encoded_data:
            dq_evaluate(
                encoded_data[split],
                split=split,
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
    return model


def do_train(
    trainer: "SetFitTrainer",
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> "SetFitTrainer":
    watch(trainer, finish=False)

    trainer.train()
    dq_evaluate = watch(trainer, finish=False)
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
