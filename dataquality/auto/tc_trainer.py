from typing import Dict, List, Tuple

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    EvalPrediction,
    IntervalStrategy,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from dataquality.schemas.split import Split


# Taken from the docs of the trainer module:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/
# text-classification/run_glue.py#L434
def preprocess_function(
    input_data: Dataset, tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    return tokenizer(
        input_data["text"], padding="max_length", max_length=200, truncation=True
    )


def compute_metrics(eval_pred: EvalPrediction) -> Dict:
    metric = evaluate.load("accuracy")
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(
    dd: DatasetDict, labels: List[str], model_checkpoint: str
) -> Tuple[Trainer, Dict[str, Dataset]]:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    encoded_datasets = {}
    for key in dd:
        encoded_datasets[key] = dd[key].map(
            lambda x: preprocess_function(x, tokenizer), batched=True
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(labels)
    )

    # Training arguments and training part
    metric_name = "accuracy"
    batch_size = 64
    has_val = Split.validation in encoded_datasets
    eval_strat = IntervalStrategy.EPOCH if has_val else IntervalStrategy.NO
    load_best_model = has_val  # Can only load the best model if we have validation data
    args = TrainingArguments(
        "finetuned",
        evaluation_strategy=eval_strat,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=load_best_model,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        report_to=["all"],
        seed=42,
    )

    # We pass huggingface datasets here but typing expects torch datasets, so we ignore
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets[Split.train],  # type: ignore
        eval_dataset=encoded_datasets.get(Split.validation),  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer, encoded_datasets
