from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.hf import tokenize_and_log_dataset
from dataquality.schemas.hf import HFCol
from dataquality.schemas.split import Split
from dataquality.utils.helpers import mps_available

try:
    # For NER training, there is only 1 evaluation tool
    # https://huggingface.co/course/chapter7/2#metrics
    import evaluate

    metric = evaluate.load("seqeval")
except ImportError:
    raise GalileoException(
        "⚠️ Huggingface evaluate library not installed "
        "please run `pip install dataquality[evaluate]` "
        "to enable metrics computation."
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    """Metrics computation for token classification

    Taken directly from the docs https://huggingface.co/course/chapter7/2#metrics
    and updated for typing
    """
    str_labels = dq.get_model_logger().logger_config.labels
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [str_labels[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [str_labels[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_trainer(
    dd: DatasetDict,
    model_checkpoint: str,
    num_train_epochs: int,
    labels: Optional[List[str]] = None,
    early_stopping: bool = True,
) -> Tuple[Trainer, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    default_cols = set(HFCol.get_fields())
    meta = [c for c in dd[Split.train].features if c not in default_cols]
    encoded_datasets = tokenize_and_log_dataset(
        dd, tokenizer, label_names=labels, meta=meta
    )
    # FIXME: Why do I need this? `tokenize_and_log_dataset` returns a DatasetDict
    #  not an Optional[DatasetDict]...?
    assert isinstance(encoded_datasets, DatasetDict)

    # Used to properly seed the model
    def model_init() -> Any:
        return AutoModelForTokenClassification.from_pretrained(
            model_checkpoint, num_labels=len(dq.get_model_logger().logger_config.labels)
        )

    batch_size = 64
    has_val = Split.validation in encoded_datasets
    eval_strat = IntervalStrategy.EPOCH if has_val else IntervalStrategy.NO
    load_best_model = has_val  # Can only load the best model if we have validation data

    args = TrainingArguments(
        "finetuned",
        evaluation_strategy=eval_strat,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=load_best_model,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy=IntervalStrategy.EPOCH,
        logging_strategy=IntervalStrategy.EPOCH,
        logging_dir="./logs",
        seed=42,
        use_mps_device=mps_available(),
    )

    callbacks = []
    if early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))
    # We pass huggingface datasets here but typing expects torch datasets, so we ignore
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_datasets[Split.train],  # type: ignore
        eval_dataset=encoded_datasets.get(Split.validation),  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=callbacks,
    )
    return trainer, encoded_datasets
