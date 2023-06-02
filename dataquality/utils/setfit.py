from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import Tensor

import dataquality as dq
from dataquality.schemas.split import Split

BATCH_LOG_SIZE = 10_000

if TYPE_CHECKING:
    from datasets import Dataset
    from setfit import SetFitModel


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
    dataset: "Dataset",
    split: Split,
    dq_store: Dict,
    batch_size: int,
    meta: Optional[List] = None,
    inference_name: Optional[str] = None,
    return_preds: bool = False,
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
            epoch=0,
            **inference_dict,  # type: ignore
        )

    # Log any leftovers
    if log_args and not skip_logging:
        dq.log_data_samples(**asdict(log_args))
    if not return_preds:
        return torch.tensor([])
    return torch.concat(preds)
