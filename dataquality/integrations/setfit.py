from typing import Any, Dict, Optional, Tuple

import numpy as np
from torch import Tensor

import dataquality as dq
from dataquality.schemas.split import Split
from dataquality.utils.cleanup import Cleanup, RefManager


class SetFitModelHook:
    def __init__(
        self,
        setfit_model: Any,
        store: Dict = None,
        func_name="predict_proba",
        n_labels=None,
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

        self.func_name = "model_head"
        self.func_name_predict = func_name
        self.n_labels = n_labels
        self.setup()

    def setup(self) -> None:
        """Setup hook to SetFit model by replacing predict_proba function with self."""

        self.old_model = getattr(self.in_cls, self.func_name)
        self.old_func = getattr(self.old_model, self.func_name_predict)
        setattr(self.old_model, self.func_name_predict, self)
        self.store["hook"] = self

    def __call__(self, *args: Tuple, **kwargs: Dict) -> Any:
        """Call predict_proba function and store input and output.
        :param args: arguments of predict_proba function
        :param kwargs: keyword arguments of predict_proba function
        :return: output of predict_proba function"""
        self.store["input_args"] = args
        self.store["input_kwargs"] = kwargs
        output = self.old_func(*args, **kwargs)
        if self.func_name == "predict":
            self.store["output"] = np.eye(self.n_values)[output]
        self.store["output"] = output
        return output

    def unpatch(self):
        """Unpatch SetFit model by replacing predict_proba
        function with old function."""
        setattr(self.old_model, self.func_name_predict, self.old_func)


def watch(model: Any) -> Dict:
    """Watch SetFit model by replacing predict_proba function with SetFitModelHook.
    :param model: SetFit model
    :return: SetFitModelHook object"""
    dq_hook = SetFitModelHook(model)
    dq_store = dq_hook.store
    labels = dq.get_data_logger().logger_config.labels

    def dq_evaluate(
        batch: Dict,
        split: Split,
        inference_name: Optional[str] = None,
        column_mapping: Optional[Dict] = {
            "text": "text",
            "id": "id",
            "label": "label",
        },
    ) -> Tensor:
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
            text_col = column_mapping[text_col]
            id_col = column_mapping[id_col]
            label_col = column_mapping[label_col]

        assert text_col in batch, f"column '{text_col}' must be in batch"
        assert id_col in batch, f"column '{id_col}' text must be in batch"

        preds = model.predict_proba(batch[text_col])
        # 🔭🌕 Galileo logging
        log_args = dict(texts=batch["text"], ids=batch[id_col], split=split)
        if inference_name is not None:
            log_args["inference_name"] = inference_name
            inference_dict = {"inference_name": inference_name}
        else:
            assert label_col in batch, f"column '{label_col}' must be in batch"
            log_args["labels"] = [labels[label] for label in batch[label_col]]
            inference_dict = {}
        helper_data = dq.get_data_logger().logger_config.helper_data

        # Unpatch SetFit model after logging (when finished is called)
        cleanup_manager = RefManager(dq_hook.unpatch)
        helper_data["cleaner"] = Cleanup(cleanup_manager)

        dq.log_data_samples(**log_args)
        # 🔭🌕 Galileo logging
        dq.log_model_outputs(
            ids=batch[id_col],
            logits=dq_store["output"],
            embs=dq_store["input_args"][0],
            split=split,
            epoch=0,
            **inference_dict,
        )
        return preds

    return dq_evaluate
