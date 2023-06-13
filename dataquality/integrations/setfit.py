from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

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
from dataquality.utils.patcher import Patch, PatchManager
from dataquality.utils.setfit import get_trainer, log_preds_setfit

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("setfit")


if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from setfit import SetFitModel, SetFitTrainer


def _apply_column_mapping(
    dataset: "Dataset", column_mapping: Dict[str, str]
) -> "Dataset":
    """
    Applies the provided column mapping to the dataset, renaming columns accordingly.
    Extra features not in the column mapping are prefixed with `"feat_"`.
    """
    if type(dataset) == dict:
        from datasets import Dataset

        dataset = Dataset.from_dict(dataset)
    dataset = dataset.rename_columns(
        {
            **column_mapping,
            **{
                col: f"feat_{col}"
                for col in dataset.column_names
                if col not in column_mapping
            },
        }
    )
    dset_format = dataset.format
    dataset = dataset.with_format(
        type=dset_format["type"],
        columns=dataset.column_names,
        output_all_columns=dset_format["output_all_columns"],
        **dset_format["format_kwargs"],
    )
    return dataset


class SetFitModelHook(Patch):
    """Hook to SetFit model to store input and output of predict_proba function."""

    name = "setfit_predict"

    def __init__(
        self,
        setfit_model: Any,
        store: Optional[Dict] = None,
        func_name: str = "predict_proba",
        n_labels: Optional[int] = None,
    ) -> None:
        """
        Hook to SetFit model to store input and output of predict_proba function.
        :param setfit_model: SetFit model
        :param store: dictionary to store input and output
        :param func_name: name of function to hook
        :param n_labels: number of labels
        """
        self.in_cls = setfit_model
        if store is not None:
            self.store = store
        else:
            self.store = {}

        self.cls_name = "model_head"
        self.func_name_predict = func_name
        self.n_labels = n_labels
        self.patch()

    def _patch(self) -> "Patch":
        """Setup hook to SetFit model by replacing predict_proba function with self."""
        self.old_model = getattr(self.in_cls, self.cls_name)
        old_fn = getattr(self.old_model, self.func_name_predict)
        if hasattr(old_fn, "is_patch"):
            old_fn.unpatch()
            old_fn = getattr(self.old_model, self.func_name_predict)
        self.old_fn = old_fn
        setattr(self.old_model, self.func_name_predict, self)
        _PatchSetFitModel(
            self.in_cls,
        )
        self.store["hook"] = self
        return self

    def __call__(self, *args: Tuple, **kwargs: Dict) -> Any:
        """Call predict_proba function and store input and output.
        :param args: arguments of predict_proba function
        :param kwargs: keyword arguments of predict_proba function
        :return: output of predict_proba function"""
        self.store["input_args"] = args
        self.store["input_kwargs"] = kwargs
        output = self.old_fn(*args, **kwargs)
        if self.cls_name == "predict":
            assert self.n_labels is not None, "n_labels must be set"
            self.store["output"] = np.eye(self.n_labels)[output]
        self.store["output"] = output
        return output

    def _unpatch(self) -> None:
        """Unpatch SetFit model by replacing predict_proba
        function with old function."""
        setattr(self.old_model, self.func_name_predict, self.old_fn)


class _PatchSetFitModel(Patch):
    """Patch to SetFit model to unpatch when calling save_pretrained function."""

    name = "setfit_save_pretrained"

    def __init__(
        self, setfit_model: "SetFitModel", function_name: str = "save_pretrained"
    ) -> None:
        """Patch to SetFit model to unpatch when calling save_pretrained function.
        :param setfit_model: SetFit model
        :param function_name: name of function to patch ('save_pretrained')
        """
        self.model = setfit_model
        self.function_name = function_name
        self.patch()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call unpatch SetFit model and then the save_pretrained function."""
        self.unpatch()
        return self.old_fn(*args, **kwds)

    def _patch(self) -> "Patch":
        """Patch SetFit model by replacing save_pretrained function with self."""
        old_fn = getattr(self.model, self.function_name)
        if hasattr(old_fn, "is_patch"):
            old_fn.unpatch()
            old_fn = getattr(self.model, self.function_name)
        self.old_fn = old_fn
        setattr(self.model, self.function_name, self)
        return self

    def _unpatch(self) -> None:
        """Unpatch SetFit model by replacing save_pretrained"""
        setattr(self.model, self.function_name, self.old_fn)


class _PatchSetFitTrainer(Patch):
    """Patch to SetFit trainer to run dataquality after training."""

    name = "setfit_trainer"

    def __init__(
        self,
        setfit_trainer: "SetFitTrainer",
        labels: List[str] = [],
        finish: bool = True,
        wait: bool = False,
        batch_size: Optional[int] = None,
        meta: Optional[List] = None,
    ) -> None:
        """Patch to SetFit trainer to run dataquality after training.
        :param setfit_trainer: SetFit trainer
        :param project_name: name of project
        :param run_name: name of run
        :param labels: list of labels
        :param finish: whether to run dq.finish after evaluation
        :param wait: whether to wait for dq.finish
        """
        self.trainer = setfit_trainer
        self.function_name = "train"
        self.patch()
        self.labels = labels
        self.finish = finish
        self.wait = wait
        self.batch_size = batch_size
        self.meta = meta

    def _patch(self) -> "Patch":
        """Patch SetFit trainer by replacing train function with self."""
        old_fn = getattr(self.trainer, self.function_name)
        if hasattr(old_fn, "is_patch"):
            old_fn.unpatch()
            old_fn = getattr(self.trainer, self.function_name)
        self.old_fn = old_fn
        setattr(self.trainer, self.function_name, self)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call train function and run dataquality after training."""
        batch_size = kwargs.get("batch_size", self.trainer.batch_size)
        if batch_size is not None and len(args) > 0:
            batch_size = args[1]
        # If batch_size is set in watch function, override the batch_size
        if self.batch_size is not None:
            batch_size = self.batch_size

        res = self.old_fn(*args, **kwargs)
        model = self.trainer.model
        dq_hook = SetFitModelHook(model)
        dq_store = dq_hook.store
        train_dataset = self.trainer.train_dataset
        eval_dataset = self.trainer.eval_dataset

        if self.trainer.column_mapping is not None:
            train_dataset = self.trainer._apply_column_mapping(
                train_dataset, self.trainer.column_mapping
            )

        labels: List = self.labels
        if not labels:
            labels = dq.get_data_logger().logger_config.labels
        if not labels:
            labels = getattr(train_dataset.features.get("label", {}), "names", [])
        assert len(labels), "Labels must be set (watch(trainer, labels=[...]))"
        dq.set_labels_for_run(labels)
        if eval_dataset is not None:
            if self.trainer.column_mapping is not None:
                eval_dataset = self.trainer._apply_column_mapping(
                    eval_dataset, self.trainer.column_mapping
                )

        for split, dataset in zip(
            [Split.training, Split.validation], [train_dataset, eval_dataset]
        ):
            if dataset is None:
                continue
            if "id" not in dataset.features:
                dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True)

            log_preds_setfit(
                model=model,
                dataset=dataset,
                dq_store=dq_store,
                batch_size=batch_size,
                split=split,
                meta=self.meta,
            )

        if self.finish:
            dq.finish(wait=self.wait)

        return res

    def _unpatch(self) -> None:
        """Unpatch SetFit trainer by replacing the patched train function with
        original function."""
        setattr(self.trainer, self.function_name, self.old_fn)


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
) -> Callable[
    ["Dataset", Split, Optional[List], Optional[str], Optional[Dict], int], torch.Tensor
]:
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

    labels = labels or []
    model = setfit

    setfitmanager = PatchManager()

    # If the user has already logged input data, skip it during evaluate
    logger_config = dq.get_data_logger().logger_config
    for split in ["training", "validation", "test", "inference"]:
        split_key = f"setfit_skip_input_log_{split}"
        logger_config.helper_data[split_key] = getattr(logger_config, f"{split}_logged")

    if isinstance(setfit, SetFitTrainer):
        patched_trainer = _PatchSetFitTrainer(
            setfit,
            labels=labels,
            finish=finish,
            wait=wait,
            batch_size=batch_size,
            meta=meta,
        )
        setfitmanager.add_patch(patched_trainer)
        return evaluate(setfit.model)
    else:
        if not labels:
            labels = dq.get_data_logger().logger_config.labels
        assert labels and len(
            labels
        ), "Labels must be set (watch(trainer, labels=[...]))"
        dq.set_labels_for_run(labels)
        return evaluate(model)


def evaluate(
    model: "SetFitModel",
) -> Callable[
    ["Dataset", Split, Optional[List], Optional[str], Optional[Dict], int], torch.Tensor
]:
    """Watch SetFit model by replacing predict_proba function with SetFitModelHook.
    :param model: SetFit model
    :return: SetFitModelHook object"""
    dq_hook = SetFitModelHook(model)
    dq_store = dq_hook.store

    def dq_evaluate(
        dataset: "Dataset",
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
    hf_data: Optional[Union["DatasetDict", str]] = None,
    hf_inference_names: Optional[List[str]] = None,
    train_data: Optional[Union[pd.DataFrame, "Dataset", str]] = None,
    val_data: Optional[Union[pd.DataFrame, "Dataset", str]] = None,
    test_data: Optional[Union[pd.DataFrame, "Dataset", str]] = None,
    inference_data: Optional[Dict[str, Union[pd.DataFrame, "Dataset", str]]] = None,
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
            split=Split.test,  # type: ignore
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
