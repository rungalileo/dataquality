import hashlib
from typing import Any, Callable, Generator

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
from dataquality.utils.thread_pool import ThreadPoolManager

food = load_dataset("sasha/dog-food")
# food["train"] = food["train"].select(range(300))
# food["test"] = food["test"].select(range(64))
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
