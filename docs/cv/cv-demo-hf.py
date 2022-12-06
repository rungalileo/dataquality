import base64
import hashlib
from io import BytesIO
from typing import Any, Callable, Generator
from unittest.mock import MagicMock, patch

import vaex
from datasets import load_dataset
from PIL import Image
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

B64_CONTENT_TYPE_DELIMITER = ";base64,"


def _b64_image_data_prefix(mimetype: str) -> bytes:
    return f"data:{mimetype}{B64_CONTENT_TYPE_DELIMITER}".encode("utf-8")


def _img_to_b64(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format=img.format)
    return base64.b64encode(img_bytes.getvalue())


def _bytes_to_img(b: bytes) -> Image:
    return Image.open(BytesIO(b))


def _raw_bytes_to_b64_str(b: bytes) -> str:
    img = _bytes_to_img(b=b)
    prefix = _b64_image_data_prefix(mimetype=img.get_format_mimetype())
    data = _img_to_b64(img=img)
    return (prefix + data).decode("utf-8")


def _main() -> None:

    dq.login()

    dq.init(
        task_type="image_classification",
        project_name="anthony-cv-testing",
        run_name="food-101-0",
    )

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

    dq.set_labels_for_run(labels)
    train_df = food["train"].to_pandas()
    test_df = food["test"].to_pandas()

    train_df["text"] = train_df["image"].apply(
        lambda x: _raw_bytes_to_b64_str(b=x["bytes"])
    )
    test_df["text"] = train_df["image"].apply(
        lambda x: _raw_bytes_to_b64_str(b=x["bytes"])
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
    # assert len(vaex.open(f"./training/0/*.hdf5")) == len(train_df)
    # assert len(vaex.open(f"{LOCATION}/validation/0/*.hdf5")) == len(test_df)
    dq.finish()


if __name__ == "__main__":
    _main()
