from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.utils.patcher import Cleanup, Patch, PatchManager, RefManager

if TYPE_CHECKING:
    from datasets import Dataset
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
        project_name: str,
        run_name: str,
        labels: List[str] = [],
        finish: bool = True,
        wait: bool = False,
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
        self.project_name = project_name
        self.run_name = run_name

    def _patch(self) -> "Patch":
        """Patch SetFit trainer by replacing train function with self."""
        old_fn = getattr(self.trainer, self.function_name)
        if hasattr(old_fn, "is_patch"):
            old_fn.unpatch()
            old_fn = getattr(self.trainer, self.function_name)
        self.old_fn = old_fn
        setattr(self.trainer, self.function_name, self)
        return self

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Call train function and run dataquality after training."""
        batch_size = kwds.get("batch_size", self.trainer.batch_size)
        if batch_size is not None and len(args) > 0:
            batch_size = args[1]

        res = self.old_fn(*args, **kwds)
        model = self.trainer.model
        dq_hook = SetFitModelHook(model)
        dq_store = dq_hook.store
        train_dataset = self.trainer.train_dataset
        eval_dataset = self.trainer.eval_dataset

        if self.trainer.column_mapping is not None:
            train_dataset = self.trainer._apply_column_mapping(
                train_dataset, self.trainer.column_mapping
            )
        if not dq.config.task_type:
            init_kwargs: Dict[str, Any] = {}
            if self.project_name:
                init_kwargs["project_name"] = self.project_name
            if self.run_name:
                init_kwargs["run_name"] = self.run_name

            dq.init("text_classification", **init_kwargs)
        labels: Any = self.labels
        if not labels:
            labels = dq.get_data_logger().logger_config.labels
        if not labels:
            labels = getattr(train_dataset.features.get("label", {}), "names", None)
        assert labels, "Labels must be set (watch(trainer, labels=[...]))"
        dq.set_labels_for_run(labels)
        datasets = [train_dataset]
        if eval_dataset is not None:
            if self.trainer.column_mapping is not None:
                eval_dataset = self.trainer._apply_column_mapping(
                    eval_dataset, self.trainer.column_mapping
                )
            datasets.append(eval_dataset)
        for split in [Split.training, Split.validation]:
            if split == Split.training:
                dataset = train_dataset
            else:
                dataset = eval_dataset
            if dataset is None:
                continue
            if "id" not in dataset.features:
                dataset = dataset.map(lambda x, idx: {"id": idx}, with_indices=True)
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i : i + batch_size]
                model.predict_proba(batch["text"])
                # 🔭🌕 Galileo logging
                dq.log_data_samples(
                    texts=batch["text"],
                    ids=batch["id"],
                    labels=[labels[label_id] for label_id in batch["label"]],
                    split=split,
                )
                # 🔭🌕 Galileo logging
                dq.log_model_outputs(
                    ids=batch["id"],
                    probs=dq_store["output"],
                    embs=dq_store["input_args"][0],
                    split=split,
                    epoch=0,
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
    setfitmanager = PatchManager()
    setfitmanager.unpatch()


def watch(
    setfit: Union["SetFitModel", "SetFitTrainer"],
    labels: List[str] = [],
    project_name: str = "",
    run_name: str = "",
    finish: bool = True,
    wait: bool = False,
) -> Optional[Callable]:
    """Watch SetFit model by replacing predict_proba function with SetFitModelHook.
    :param model: SetFit model"""
    model = setfit

    setfitmanager = PatchManager()

    if setfit.__class__.__name__ == "SetFitTrainer":
        model = setfit.model
        patched_trainer = _PatchSetFitTrainer(
            setfit,
            labels=labels,
            finish=finish,
            wait=wait,
            run_name=run_name,
            project_name=project_name,
        )
        setfitmanager.add_patch(patched_trainer)
        return None
    else:
        return evaluate(model)


def evaluate(model: "SetFitModel") -> Callable:
    """Watch SetFit model by replacing predict_proba function with SetFitModelHook.
    :param model: SetFit model
    :return: SetFitModelHook object"""
    dq_hook = SetFitModelHook(model)
    dq_store = dq_hook.store
    labels = dq.get_data_logger().logger_config.labels

    helper_data = dq.get_data_logger().logger_config.helper_data

    # Unpatch SetFit model after logging (when finished is called)
    cleanup_manager = RefManager(dq_hook.unpatch)
    helper_data["cleaner"] = Cleanup(cleanup_manager)

    def dq_evaluate(
        dataset: "Dataset",
        split: Split,
        inference_name: Optional[str] = None,
        column_mapping: Optional[Dict] = {
            "text": "text",
            "id": "id",
            "label": "label",
        },
        batch_size: int = 64,
    ) -> torch.Tensor:
        """Evaluate SetFit model and log input and output to Galileo.
        :param batch: batch of data as a dictionary
        :param split: split of data (training, validation, test, inference)
        :param inference_name: inference name (if split is inference, must be provided)
        :param column_mapping: mapping of column names (if different from default)
        :return: output of SetFitModel.predict_proba function"""

        text_col = "text"
        id_col = "id"
        label_col = "label"
        if column_mapping is not None:
            dataset = _apply_column_mapping(dataset, column_mapping)
        preds: List[torch.Tensor] = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]

            assert text_col in batch, f"column '{text_col}' must be in batch"
            assert id_col in batch, f"column '{id_col}' text must be in batch"

            pred = model.predict_proba(batch[text_col])
            preds.append(pred)
            # 🔭🌕 Galileo logging
            log_args = dict(texts=batch["text"], ids=batch[id_col], split=split)
            inference_dict: Dict[str, str] = {}
            if inference_name is not None:
                log_args["inference_name"] = inference_name
                inference_dict["inference_name"] = inference_name
            else:
                assert label_col in batch, f"column '{label_col}' must be in batch"
                log_args["labels"] = [labels[label] for label in batch[label_col]]

            dq.log_data_samples(**log_args)
            # 🔭🌕 Galileo logging
            dq.log_model_outputs(
                ids=batch[id_col],
                probs=dq_store["output"],
                embs=dq_store["input_args"][0],
                split=split,
                epoch=0,
                **inference_dict,  # type: ignore
            )

        return torch.concat(preds)

    return dq_evaluate
