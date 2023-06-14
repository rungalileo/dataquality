import contextlib
import io
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import torch

import dataquality as dq
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core.log import get_data_logger
from dataquality.schemas.split import Split
from dataquality.utils.patcher import PatchManager
from dataquality.utils.setfit import (
    SetFitModelHook,
    _apply_column_mapping,
    _prepare_config,
    _setup_patches,
    log_preds_setfit,
    validate_setfit,
)

a = Analytics(ApiClient, dq.config)  # type: ignore
a.log_import("setfit")


if TYPE_CHECKING:
    from datasets import Dataset
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

    init_kwargs: Dict[str, Any] = {}
    if not dq.config.task_type:
        if project_name:
            init_kwargs["project_name"] = project_name
        if run_name:
            init_kwargs["run_name"] = run_name
        dq.init("text_classification", **init_kwargs)

    labels = labels or []
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
