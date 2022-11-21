import base64
import os
import pdb
from io import BytesIO
from typing import Dict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor
from transformers import (
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

import dataquality as dq
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split

# from transformers.data.data_collator import DefaultDataCollator


#
#   dq
#
dq.login()

dq.init(
    task_type="image_classification",
    project_name="anthony-cv-testing",
    run_name="food-101-0",
)

#
#   Preparing the dataset
#
cache_dir = f'{os.environ["HOME"]}/.transformers/data'
food = load_dataset("food101", split="train", cache_dir=cache_dir, streaming=True)
# food = load_dataset("sasha/dog-food", split="train", cache_dir=cache_dir, streaming=True)
food_list = list(food.take(10))  # type: ignore

food = Dataset.from_list(food_list)
food = food.train_test_split(test_size=0.2)

print(food["train"][0])

df_train = pd.DataFrame(food["train"]["label"]).reset_index()
df_train["text"] = "img_" + df_train["index"].astype(str)
df_train = df_train.rename(columns={0: "label", "index": "id"})

df_test = pd.DataFrame(food["test"]["label"]).reset_index()
df_test["text"] = "img_" + df_test["index"].astype(str)
df_test = df_test.rename(columns={0: "label", "index": "id"})

food = food.map(lambda x, idx: {"id": idx}, with_indices=True)

print("dataset preparing finished, logging dataset")

dq.log_dataset(df_train, split=Split.train)
dq.log_dataset(df_test, split=Split.test)

#
#   Setting the labels
#
labels = list(set(food["train"]["label"]))

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[str(label)] = str(i)
    id2label[str(i)] = str(label)

dq.set_labels_for_run(list(label2id.values()))

#
#   Building and running the model
#
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)
normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)
_transforms = Compose(
    [RandomResizedCrop(feature_extractor.size), ToTensor(), normalize]
)

B64_CONTENT_TYPE_DELIMITER = ";base64,"


def _b64_image_data_prefix(mimetype: str) -> bytes:
    return f"data:{mimetype}{B64_CONTENT_TYPE_DELIMITER}".encode("utf-8")


def _img_to_b64(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format=img.format)
    return base64.b64encode(img_bytes.getvalue())


def transforms(examples: Dict) -> Dict:
    # for img in examples["image"]:
    #     print(type(img))
    #     print(type(_transforms(list(img.convert("RGB").getdata()))))
    #     pdb.set_trace()

    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]

    # examples["img_tn"] = # thumbnail b64data
    del examples["image"]

    return examples


_food = food.with_transform(transforms)
data_collator = DefaultDataCollator()


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        print(type(example["pixel_values"][0]))
        images.append(example["pixel_values"])
        labels.append(example["labels"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


dataloader = DataLoader(_food, collate_fn=collate_fn, batch_size=4)


model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)


def _test(*args, **kwargs) -> None:
    print(args, kwargs)


next(model.children()).register_forward_hook(_test)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=1,
    # fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=_food["train"],
    eval_dataset=_food["test"],
    tokenizer=feature_extractor,
)

watch(trainer, layer=model.vit)
trainer.train()

dq.finish()
