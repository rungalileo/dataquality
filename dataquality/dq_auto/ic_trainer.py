from functools import partial
from typing import Dict, List, Tuple

import evaluate
from evaluate import EvaluationModule
import numpy as np
from datasets import Dataset, DatasetDict

from dataquality.schemas.split import Split
from dataquality.utils.helpers import mps_available

import torch
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)


EVAL_METRIC = "accuracy"


def preprocess_function(input_data: Dataset):
    pass

# TODO: add type
def data_collator(examples) -> Dict:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(metric: EvaluationModule, eval_pred: EvalPrediction) -> Dict:
    return metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)


# TODO: add types + use image_col_name
def add_transforms(examples, image_col_name: str, transforms):
    examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples[image_col_name]]
    del examples[image_col_name]
    return examples


def get_trainer(
    dd: DatasetDict,
    labels: List[str],
    model_checkpoint: str,
) -> Tuple[Trainer, DatasetDict]:

    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Get model and pre-processor
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageClassification.from_pretrained(model_checkpoint, ignore_mismatched_sizes = True, label2id=label2id, id2label=id2label)#.to(device) # num_labels=len(labels) instead of ignore ?

    # Set the augmentation TODO: put in preprocess function
    image_col_name = "image" # TODO: find this
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = ( 
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    ) 
    # del image_processor # TODO: can we get/estimate the size from somewhere else ? Just has do be consistent, not necessarily the one used here. 224 by default ?
    transforms_train = Compose([RandomResizedCrop(size), ToTensor(), normalize]) # TODO: add more
    dd[Split.train].set_transform(lambda examples: add_transforms(examples, image_col_name, transforms_train))
    transforms_eval = Compose([RandomResizedCrop(size), ToTensor()])
    dd[Split.validation].set_transform(lambda examples: add_transforms(examples, image_col_name, transforms_eval)) # TODO: loop over non training splits
    dd[Split.test].set_transform(lambda examples: add_transforms(examples, image_col_name, transforms_eval)) # TODO: loop over non training splits

    # Training arguments and training part
    metric = evaluate.load(EVAL_METRIC)
    # We use the users chosen evaluation metric by preloading it into the partial
    compute_metrics_partial = partial(compute_metrics, metric)
    batch_size = 64
    has_val = Split.validation in dd
    eval_strat = IntervalStrategy.EPOCH if has_val else IntervalStrategy.NO
    load_best_model = has_val  # Can only load the best model if we have validation data
    training_args = TrainingArguments(
        "finetuned",
        evaluation_strategy=eval_strat,
        remove_unused_columns=False,
        # save_strategy=IntervalStrategy.EPOCH,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3, # TODO: + add callback
        weight_decay=0.01,
        # load_best_model_at_end=load_best_model,
        push_to_hub=False,
        report_to=["all"],
        seed=42,
        # use_mps_device=mps_available(),
    )

    # We pass huggingface datasets here but typing expects torch datasets, so we ignore
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dd[Split.train],  # type: ignore
        eval_dataset=dd.get(Split.validation),  # type: ignore
        data_collator=data_collator,
        # tokenizer=image_processor,
        compute_metrics=compute_metrics_partial,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    return trainer, dd#, encoded_datasets
