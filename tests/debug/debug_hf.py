import dataquality
import copy
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality import config
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.task_type import TaskType


model_name =  "microsoft/xtremedistil-l6-h256-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_name = "emotion"
dataset = load_dataset(dataset_name)
metric = load_metric("accuracy")
def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", max_length=201, truncation=True
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 🔭🌕 Galileo logging
ds = dataset.map(lambda x,idx: {"id":idx}, with_indices=True)

train_dataset = ds["train"].select(range(4000)) 
test_dataset = ds["test"].select(range(2000))

encoded_train_dataset = train_dataset.map(lambda x: preprocess_function(x,tokenizer),batched=True) 
encoded_test_dataset =  test_dataset.map(lambda x: preprocess_function(x,tokenizer),batched=True) 
model = AutoModelForSequenceClassification.from_pretrained(
  model_name ,num_labels=train_dataset.features["label"].num_classes
)

#Training arguments and training part
metric_name = "accuracy"
batch_size= 16
args = TrainingArguments(
    f"finetuned",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
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
    #data_seed=42
    )

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

watch(trainer)
trainer.train()