import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch import Tensor

import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.patcher import Patch, PatchManager

BATCH_LOG_SIZE = 10_000

if TYPE_CHECKING:
    from setfit import SetFitModel, SetFitTrainer


@dataclass
class DataSampleLogArgs:
    split: Split
    inference_name: Optional[str] = None
    meta: Optional[Dict] = None
    texts: List[str] = field(default_factory=list)
    ids: List[int] = field(default_factory=list)
    labels: List = field(default_factory=list)

    def clear(self) -> None:
        """Resets the arrays of the class."""
        self.texts.clear()
        self.ids.clear()
        self.labels.clear()


def log_preds_setfit(
    model: "SetFitModel",
    dataset: Dataset,
    split: Split,
    dq_store: Dict,
    batch_size: int,
    meta: Optional[List] = None,
    inference_name: Optional[str] = None,
    return_preds: bool = False,
    epoch: Optional[int] = None,
) -> Tensor:
    """Logs the data samples and model outputs for a SetFit model.
    :param model: The SetFit model
    :param dataset: The dataset in the form of a HuggingFace Dataset
    :param split: The split of the data samples (for example "training")
    :param dq_store: The dataquality store
    :param batch_size: The batch size
    :param inference_name: The name of the inference (for example "inference_run_1")
    :param return_preds: Whether to return the predictions
    :return: The predictions
    """
    text_col = "text"
    id_col = "id"
    label_col = "label"
    preds: List[Tensor] = []
    log_args: DataSampleLogArgs = DataSampleLogArgs(split=split)
    inference_dict: Dict[str, str] = {}
    if inference_name is not None:
        log_args.inference_name = inference_name
        inference_dict["inference_name"] = inference_name

    logger_config = dq.get_data_logger().logger_config
    labels = logger_config.labels
    epoch = epoch or 0

    # Check if the data should be logged by checking if the split is in the
    # input_data_logged
    skip_logging = logger_config.helper_data[f"setfit_skip_input_log_{split}"]
    # Iterate over the dataset in batches and log the data samples
    # and model outputs
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        assert text_col in batch, f"column '{text_col}' must be in batch"
        assert id_col in batch, f"column '{id_col}' text must be in batch"

        if inference_name is None and not skip_logging:
            assert label_col in batch, f"column '{label_col}' must be in batch"
            log_args.labels += [labels[label] for label in batch[label_col]]

        pred = model.predict_proba(batch[text_col])
        if return_preds:
            preds.append(pred)
        # 🔭🌕 Galileo logging
        if not skip_logging:
            log_args.texts += batch[text_col]
            log_args.ids += batch[id_col]
            if meta is not None:
                log_args.meta = {
                    meta_col: batch[f"feat_{meta_col}"] for meta_col in meta
                }
            if len(log_args.texts) >= BATCH_LOG_SIZE:
                dq.log_data_samples(**asdict(log_args))
                log_args.clear()
        # 🔭🌕 Galileo logging
        dq.log_model_outputs(
            ids=batch[id_col],
            probs=dq_store["output"],
            embs=dq_store["input_args"][0],
            split=split,
            epoch=epoch,
            **inference_dict,  # type: ignore
        )

    # Log any leftovers
    if log_args and not skip_logging:
        dq.log_data_samples(**asdict(log_args))
    if not return_preds:
        return torch.tensor([])
    return torch.concat(preds)


def _prepare_config() -> None:
    """
    Prepares the config for the SetFit model.
    If the user has already logged input data, skip it during evaluate
    """
    logger_config = dq.get_data_logger().logger_config
    for split in ["training", "validation", "test", "inference"]:
        split_key = f"setfit_skip_input_log_{split}"
        logger_config.helper_data[split_key] = getattr(logger_config, f"{split}_logged")


def _setup_patches(
    setfit: Union["SetFitModel", "SetFitTrainer"],
    labels: List[str],
    finish: bool = True,
    wait: bool = False,
    batch_size: Optional[int] = None,
    meta: Optional[List] = None,
) -> None:
    """Sets up the patches for a SetFit model.
    :param setfit: The SetFit model or trainer
    :param labels: The labels of the model
    :param finish: Whether to finish the run
    :param wait: Whether to wait for the run to finish
    :param batch_size: The batch size
    :param meta: The meta columns
    """

    setfitmanager = PatchManager()
    patched_trainer = _PatchSetFitTrainer(
        setfit,
        labels=labels,
        finish=finish,
        wait=wait,
        batch_size=batch_size,
        meta=meta,
    )
    setfitmanager.add_patch(patched_trainer)


def validate_setfit(
    setfit: Union["SetFitModel", "SetFitTrainer"],
    labels: List[str],
    batch_size: Optional[int] = None,
    meta: Optional[List] = None,
) -> None:
    """Validates a SetFit model.
    :param setfit: The SetFit model or trainer
    :param labels: The labels of the model
    :param wait: Whether to wait for the run to finish
    :param batch_size: The batch size
    :param meta: The meta columns
    """
    from setfit import sample_dataset

    # Store the current project and run name
    dq_project_name = dq.config.current_project_name
    dq_run_name = dq.config.current_run_name
    # Create a random project and run name to avoid collisions
    random_id = str(uuid.uuid4())
    dq.init(
        "text_classification",
        project_name="validate_project_name",
        run_name=random_id,
    )
    _prepare_config()
    _setup_patches(
        setfit,
        labels,
        finish=False,
        batch_size=batch_size,
        meta=meta,
    )
    train_dataset = setfit.train_dataset
    eval_dataset = setfit.eval_dataset
    # Sample the dataset to speed up the test
    setfit.train_dataset = sample_dataset(setfit.train_dataset, num_samples=2)
    setfit.eval_dataset = sample_dataset(setfit.eval_dataset, num_samples=2)
    setfit.train(num_epochs=1)
    setfit.evaluate()
    c = dq.get_data_logger(TaskType.text_classification)
    # Mock the finish with upload
    c.upload()
    c._cleanup()
    dq.core.init.delete_run("validate_project_name", random_id)
    # Restore the original datasets, project and run name
    setfit.train_dataset = train_dataset
    setfit.eval_dataset = eval_dataset
    PatchManager().unpatch()
    dq.init(
        "text_classification",
        project_name=dq_project_name,
        run_name=dq_run_name,
    )
    _prepare_config()


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


def get_trainer(
    dd: DatasetDict,
    model_checkpoint: str,
    training_args: Optional[Dict[str, Any]],
) -> Tuple["SetFitTrainer", DatasetDict]:
    from sentence_transformers.losses import CosineSimilarityLoss
    from setfit import SetFitModel, SetFitTrainer

    # Used to properly seed the model
    def model_init() -> Any:
        return SetFitModel.from_pretrained(model_checkpoint)

    has_val = Split.validation in dd
    setfit_args = {
        "loss_class": CosineSimilarityLoss,
        "num_iterations": 20,
    }
    if training_args is not None:
        setfit_args.update(training_args)
    trainer = SetFitTrainer(
        model=model_init(),
        train_dataset=dd[Split.training],
        eval_dataset=dd[Split.validation] if has_val else None,
        **setfit_args,
    )
    return trainer, dd
