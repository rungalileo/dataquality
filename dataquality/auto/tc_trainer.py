from typing import Dict, List, Tuple

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


# Taken from the docs of the trainer module:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/
# text-classification/run_glue.py#L434
def preprocess_function(
    input_data: Dataset, tokenizer: PreTrainedTokenizerBase
) -> BatchEncoding:
    return tokenizer(
        input_data["text"], padding="max_length", max_length=201, truncation=True
    )


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> float:
    metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)


def get_trainer(
    dd: DatasetDict, labels: List[str]
) -> Tuple[Trainer, Dict[str, Dataset]]:
    model_checkpoint = "microsoft/xtremedistil-l6-h256-uncased"
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
    args = TrainingArguments(
        "finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        report_to="all",
        seed=42,
        data_seed=42,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets.get("validation"),
        # test_dataset=encoded_datasets.get("test"), TODO: test_dataset for predict?
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer, encoded_datasets
