import os
from random import random

import pandas as pd
import vaex
from sklearn.datasets import fetch_20newsgroups

import dataquality
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.schemas.split import Split
from tests.conftest import LOCATION, SPLITS, SUBDIRS, TEST_PATH

NUM_RECORDS = 50
NUM_LOGS = 10


def validate_uploaded_data(expected_num_records: int) -> None:
    """
    Checks for testing
    """
    for split in SPLITS:
        # Output data
        split_output_data = {}
        for subdir in SUBDIRS:
            file_path = f"{TEST_PATH}/{split}/{subdir}/{subdir}.arrow"
            # Ensure files were cleaned up
            data = vaex.open(file_path).to_pandas_df()
            assert not data.isnull().any().any()
            split_output_data[subdir] = data

        data = split_output_data["data"]
        emb = split_output_data["emb"]
        prob = split_output_data["prob"]

        assert len(data) == len(emb) == len(prob) == expected_num_records
        assert (
            sorted(data["id"].unique())
            == sorted(emb["id"].unique())
            == sorted(prob["id"].unique())
        )


def validate_cleanup_data():
    """
    Checks for testing
    """
    for split in SPLITS:
        # Ensure files were cleaned up
        assert not os.path.isdir(f"{LOCATION}/{split}")


def _log_data(num_records=NUM_RECORDS, num_logs=NUM_LOGS, unique_ids=True) -> None:
    """
    Logs some mock data to disk
    """

    # Log train/test data
    for split in [Split.test, Split.training]:
        newsgroups_train = fetch_20newsgroups(
            subset="train" if split == Split.training else split.value,
            remove=("headers", "footers", "quotes"),
        )
        assert num_records * num_logs <= len(
            newsgroups_train.data
        ), f"num_records*num_logs must be less than {len(newsgroups_train.data)} "
        dataset = pd.DataFrame()
        dataset["text"] = newsgroups_train.data
        dataset["label"] = newsgroups_train.target
        dataset = dataset[: num_records * num_logs]
        gconfig = GalileoDataConfig(
            text=dataset["text"], labels=dataset["label"], split=split.value
        )
        dataquality.log_batch_input_data(gconfig)

    for split in [Split.training, Split.test]:
        for ln in range(num_logs):
            emb = [[random() for _ in range(20)] for _ in range(num_records)]
            probs = [[random() for _ in range(4)] for _ in range(num_records)]
            epoch = 0

            # Need unique ids
            if unique_ids:
                r = range(ln * num_records, (ln + 1) * num_records)
                ids = list(r)
            else:
                ids = list(range(num_records))

            model_config = GalileoModelConfig(
                emb=emb, probs=probs, split=split.value, epoch=epoch, ids=ids
            )
            dataquality.log_model_outputs(model_config)
