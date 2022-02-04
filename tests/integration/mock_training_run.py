"""Use dataquality client as if we were training a model without training a model
Usage: `python model_training_run.py`
To change datasets change DATASET flag to something in this s3 bucket:
https://s3.console.aws.amazon.com/s3/buckets/galileo-ml-train?region=us-west-
1&prefix=datasets/original/&showversions=false
"""

# Set environment flags for dataquality import
import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

os.environ["GALILEO_API_URL"] = "https://api.dev.rungalileo.io"
os.environ["GALILEO_MINIO_URL"] = "data.dev.rungalileo.io"
os.environ["GALILEO_MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["GALILEO_MINIO_SECRET_KEY"] = "minioadmin"
os.environ["GALILEO_AUTH_METHOD"] = "email"
os.environ["GALILEO_USERNAME"] = "adminy_guy@rungalileo.io"
os.environ["GALILEO_PASSWORD"] = "Admin123@"

import dataquality  # noqa
from dataquality.utils import tqdm  # noqa

DATASET = "amazon_polarity"
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
    print(dataset.info(memory_usage="deep"))  # type: ignore
    return dataset


def generate_random_embeddings(batch_size: int, emb_dims: int) -> np.ndarray:
    return np.random.rand(batch_size, emb_dims)


def generate_random_probabilities(batch_size: int, num_classes: int) -> np.ndarray:
    probs = np.random.rand(batch_size, num_classes)
    return probs / probs.sum(axis=-1).reshape(-1, 1)  # Normalize to sum to 1


if __name__ == "__main__":
    download_dataset_from_aws(DATASET_FOLDER_PATH)
    train_dataset = load_dataset_split(DATASET, "train")
    test_dataset = load_dataset_split(DATASET, "test")

    dataquality.login()
    dataquality.init(
        project_name="test_large_dataset", run_name=DATASET, task_type=TASK_TYPE
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
            batch = train_dataset[i : i + BATCH_SIZE]

            embedding = generate_random_embeddings(len(batch), EMB_DIM)
            probs = generate_random_probabilities(len(batch), num_classes)

            dataquality.log_model_outputs(
                emb=embedding,
                probs=probs,
                split="test",
                epoch=epoch_idx,
                ids=batch["id"],
            )
    print(f"Took {time.time() - t_start} seconds")
