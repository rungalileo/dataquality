from dataclasses import asdict, dataclass
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
class LogArgs:
    texts: List[str]
    ids: List[int]
    split: Split
    inference_name: Optional[str]
    labels: List

    def __init__(
        self,
        split: Split,
        inference_name: Optional[str] = None,
    ) -> None:
        self.texts = []
        self.ids = []
        self.labels = []
        self.split = split
        self.inference_name = inference_name

    def clear(self) -> None:
        self.texts.clear()
        self.ids.clear()
        self.labels.clear()


def run_model_predictions(
    model: "SetFitModel",
    dataset: "Dataset",
    split: Split,
    dq_store: Dict,
    batch_size: int,
    inference_name: Optional[str] = None,
) -> Tensor:
    text_col = "text"
    id_col = "id"
    label_col = "label"
    preds: List[Tensor] = []
    log_args: LogArgs = LogArgs(split=split)
    inference_dict: Dict[str, str] = {}
    if inference_name is not None:
        log_args.inference_name = inference_name
        inference_dict["inference_name"] = inference_name

    labels = dq.get_data_logger().logger_config.labels

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        assert text_col in batch, f"column '{text_col}' must be in batch"
        assert id_col in batch, f"column '{id_col}' text must be in batch"

        if inference_name is None:
            assert label_col in batch, f"column '{label_col}' must be in batch"
            log_args.labels += [labels[label] for label in batch[label_col]]
        pred = model.predict_proba(batch[text_col])
        preds.append(pred)
        # 🔭🌕 Galileo logging
        log_args.texts += batch[text_col]
        log_args.ids += batch[id_col]

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
    # Any leftovers
    if log_args:
        dq.log_data_samples(**asdict(log_args))

    return torch.concat(preds)
