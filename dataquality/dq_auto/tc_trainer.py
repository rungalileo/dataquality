from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
from dataquality.utils.helpers import mps_available

EVAL_METRIC = "f1"

try:
    import evaluate
    from evaluate import EvaluationModule
except ImportError:
    raise GalileoException(
        "⚠️ Huggingface evaluate library not installed "
        "please run `pip install dataquality[evaluate]` "
        "to enable metrics computation."
    )


# Taken from the docs of the trainer module:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/
# text-classification/run_glue.py#L434
def preprocess_function(
    input_data: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> BatchEncoding:
    return tokenizer(
        input_data["text"], padding="max_length", max_length=max_length, truncation=True
    )


def compute_metrics(metric: EvaluationModule, eval_pred: EvalPrediction) -> Dict:
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=1)
    return metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )


def get_trainer(
    dd: DatasetDict,
    labels: List[str],
    model_checkpoint: str,
    max_padding_length: int,
    num_train_epochs: int,
) -> Tuple[Trainer, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    encoded_datasets = dd.map(
        lambda x: preprocess_function(x, tokenizer, max_padding_length), batched=True
    )

    # Used to properly seed the model
    def model_init() -> Any:
        return AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=len(labels)
        )

    # Training arguments and training part
    metric = evaluate.load(EVAL_METRIC)
    # We use the users chosen evaluation metric by preloading it into the partial
    compute_metrics_partial = partial(compute_metrics, metric)
    batch_size = 64
    has_val = Split.validation in encoded_datasets
    eval_strat = IntervalStrategy.EPOCH if has_val else IntervalStrategy.NO
    load_best_model = has_val  # Can only load the best model if we have validation data
    args = TrainingArguments(
        "finetuned",
        evaluation_strategy=eval_strat,
        save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=load_best_model,
        push_to_hub=False,
        report_to=["all"],
        seed=42,
        use_mps_device=mps_available(),
    )

    # We pass huggingface datasets here but typing expects torch datasets, so we ignore
    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=encoded_datasets[Split.train],  # type: ignore
        eval_dataset=encoded_datasets.get(Split.validation),  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_partial,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    return trainer, encoded_datasets
