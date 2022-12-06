import hashlib
from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import vaex
from datasets import load_dataset
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

import dataquality as dq
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCATION

food = load_dataset("sasha/dog-food")
food["train"] = food["train"].select(range(200))
food["test"] = food["test"].select(range(64))
food["train"] = food["train"].map(lambda x, idx: {"id": idx}, with_indices=True)
food["test"] = food["test"].map(lambda x, idx: {"id": idx}, with_indices=True)

labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model_name = "facebook/deit-tiny-patch16-224"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)
_transforms = Compose(
    [RandomResizedCrop(feature_extractor.size), ToTensor(), normalize]
)


def transforms(examples):
    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples


food = food.with_transform(transforms)
data_collator = DefaultDataCollator()

model = AutoModelForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_cv_hf(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    set_test_config(task_type=TaskType.text_classification)
    dq.set_labels_for_run(labels)
    train_df = food["train"].to_pandas()
    test_df = food["test"].to_pandas()

    train_df["text"] = train_df["image"].apply(
        lambda x: hashlib.sha256(x.get("bytes")).hexdigest()
    )
    test_df["text"] = test_df["image"].apply(
        lambda x: hashlib.sha256(x.get("bytes")).hexdigest()
    )
    train_df["label"] = train_df["label"].astype(str).map(id2label)
    test_df["label"] = test_df["label"].astype(str).map(id2label)
    dq.log_dataset(train_df[["id", "text", "label"]], split="train")
    dq.log_dataset(test_df[["id", "text", "label"]], split="validation")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=food["train"],
        eval_dataset=food["test"],
        tokenizer=feature_extractor,
    )
    watch(trainer)
    trainer.train()
    trainer.evaluate()
    ThreadPoolManager.wait_for_threads()

    assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5")) == len(train_df)
    assert len(vaex.open(f"{LOCATION}/test/**/*.hdf5")) == len(test_df)
    dq.finish()
