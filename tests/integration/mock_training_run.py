"""Use dataquality client as if we were training a model without training a model
Usage: `python model_training_run.py`
To change datasets change DATASET flag to something in this s3 bucket:
https://s3.console.aws.amazon.com/s3/buckets/galileo-ml-train?region=us-west-
1&prefix=datasets/original/&showversions=false
"""

# Set environment flags for dataquality import
import json
import os
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import requests

import dataquality
from dataquality.utils import tqdm
from dataquality.utils.auth import headers

dataquality.configure()

DATASET = "newsgroups"
TRAIN_DATASET_NAME = f"{DATASET}_train.csv"
TEST_DATASET_NAME = f"{DATASET}_test.csv"
DATASET_FOLDER_PATH = Path("galileo-ml-train") / "datasets" / "original" / DATASET
TASK_TYPE = "text_classification"

NUM_EPOCHS = 1
BATCH_SIZE = 32
EMB_DIM = 768


def download_dataset_from_aws(dataset_folder_path: Union[Path, str]) -> None:
    cmd = f"aws s3 cp --recursive s3://{dataset_folder_path} ."
    os.system(cmd)


def load_dataset_split(dataset: str, split: str) -> pd.DataFrame:
    dataset = pd.read_csv(dataset + f"_{split}.csv")
    print(dataset.info(memory_usage="deep"))
    return dataset


def generate_random_embeddings(batch_size: int, emb_dims: int) -> np.ndarray:
    return np.random.rand(batch_size, emb_dims)


def generate_random_probabilities(batch_size: int, num_classes: int) -> np.ndarray:
    probs = np.random.rand(batch_size, num_classes)
    return probs / probs.sum(axis=-1).reshape(-1, 1)  # Normalize to sum to 1


def log_data():
    download_dataset_from_aws(DATASET_FOLDER_PATH)
    train_dataset = load_dataset_split(DATASET, "train")
    test_dataset = load_dataset_split(DATASET, "test")

    dataquality.login()
    dataquality.init(
        project_name="test_IT",
        run_name=f"{DATASET}_{datetime.today()}",
        task_type=TASK_TYPE,
    )

    t_start = time.time()
    dataquality.log_input_data(
        text=train_dataset["text"],
        labels=train_dataset["label"],
        ids=train_dataset["id"],
        split="train",
    )
    dataquality.log_input_data(
        text=test_dataset["text"],
        labels=test_dataset["label"],
        ids=test_dataset["id"],
        split="test",
    )
    dataquality.set_labels_for_run(train_dataset["label"].unique())
    print(f"Input logging took {time.time() - t_start} seconds")

    t_start = time.time()
    num_classes = train_dataset["label"].nunique()
    # Simulates model training loop
    for epoch_idx in range(NUM_EPOCHS):
        print(f"Epoch {epoch_idx}")
        # Train
        print("Training")
        for i in tqdm(range(0, len(train_dataset), BATCH_SIZE)):
            batch = train_dataset[i : i + BATCH_SIZE]

            embedding = generate_random_embeddings(len(batch), EMB_DIM)
            probs = generate_random_probabilities(len(batch), num_classes)

            dataquality.log_model_outputs(
                emb=embedding,
                probs=probs,
                split="train",
                epoch=epoch_idx,
                ids=batch["id"],
            )
        # Test
        print("Testing")
        for i in tqdm(range(0, len(test_dataset), BATCH_SIZE)):
            batch = test_dataset[i : i + BATCH_SIZE]

            embedding = generate_random_embeddings(len(batch), EMB_DIM)
            probs = generate_random_probabilities(len(batch), num_classes)

            dataquality.log_model_outputs(
                emb=embedding,
                probs=probs,
                split="test",
                epoch=epoch_idx,
                ids=batch["id"],
            )
    time_spent = time.time() - t_start
    print(f"Took {time_spent} seconds")
    return time_spent


if __name__ == "__main__":
    time_spent_logging = log_data()
    dataquality.finish()
    print("Sleeping while job processes")
    for i in tqdm(range(100)):
        time.sleep(1)
    req = dict(
        project_id=str(dataquality.config.current_project_id),
        run_id=str(dataquality.config.current_run_id),
        split="training",
        meta_cols=["galileo_text_length", "galileo_language_id"],
    )

    response = requests.post(
        f"{dataquality.config.api_url}/proc/export",
        data=json.dumps(req),
        headers=headers(dataquality.config.token),
    )
    x = str(response.content, "utf-8")
    data = StringIO(x)
    pdf = pd.read_csv(data)
    assert "galileo_language_id" in pdf.columns
    assert "galileo_text_length" in pdf.columns
    assert len(pdf) == len(load_dataset_split(DATASET, "train"))
    assert pd.notna(pdf).all().values[0]
    assert pd.notnull(pdf).all().values[0]
    print("Done!")
