from functools import partial
from typing import Dict, List, Optional, Tuple

import evaluate
import numpy as np
from datasets import DatasetDict
from evaluate import EvaluationModule
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality.integrations.hf import tokenize_and_log_dataset
from dataquality.schemas.hf import HFCol
from dataquality.schemas.split import Split


def compute_metrics(metric: EvaluationModule, eval_pred: EvalPrediction) -> Dict:
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [labels[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return metric.compute(predictions=true_predictions, references=true_labels)


def get_trainer(
    dd: DatasetDict,
    model_checkpoint: str,
    evaluation_metric: str,
    labels: Optional[List[str]] = None,
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

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint, num_labels=len(dq.get_model_logger().logger_config.labels)
    )

    # Training arguments and training part
    metric = evaluate.load(evaluation_metric)
    # We use the users chosen evaluation metric by preloading it into the partial
    compute_metrics_partial = partial(compute_metrics, metric)
    batch_size = 64
    has_val = Split.validation in encoded_datasets
    eval_strat = IntervalStrategy.EPOCH if has_val else IntervalStrategy.NO
    load_best_model = has_val  # Can only load the best model if we have validation data

    args = TrainingArguments(
        "finetuned",
        evaluation_strategy=eval_strat,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=evaluation_metric,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="./logs",
        seed=42,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    # We pass huggingface datasets here but typing expects torch datasets, so we ignore
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets[Split.train],  # type: ignore
        eval_dataset=encoded_datasets.get(Split.validation),  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_partial,
        data_collator=data_collator,
    )
    return trainer, encoded_datasets
