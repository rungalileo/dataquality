import os
from glob import glob
from random import random

import numpy as np
import pandas as pd
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
        # Output file names
        output_files = {"data": [], "emb": [], "prob": []}
        # Output data
        output_results = {"data": [], "emb": [], "prob": []}
        for subdir in SUBDIRS:
            for file in sorted(glob(f"{TEST_PATH}/{split}/{subdir}/*")):
                fname = file.split("/")[-1]
                # Ensure file was cleaned up
                assert not os.path.isfile(f"{LOCATION}/{fname}")
                output_files[subdir].append(fname.split(".")[0])
                if "arrow" in fname:
                    data = pd.read_feather(file)
                    assert not data.isnull().any().any()
                else:
                    data = np.load(file, allow_pickle=True)
                    assert not np.isnan(data).any()
                output_results[subdir].append(data)
        # The files should have identical names (file ext removed)
        assert output_files["data"] == output_files["emb"] == output_files["prob"]
        data = pd.concat(output_results["data"])
        emb = np.concatenate(output_results["emb"])
        prob = np.concatenate(output_results["prob"])
        assert len(data) == len(emb) == len(prob) == expected_num_records


def _log_data(num_records=NUM_RECORDS, num_logs=NUM_LOGS) -> None:
    """
    Logs some mock data to disk
    """

    # Log train/test data
    for split in [Split.test, Split.training]:
        newsgroups_train = fetch_20newsgroups(
            subset="train" if split == Split.training else split.value,
            remove=("headers", "footers", "quotes"),
        )
        dataset = pd.DataFrame()
        dataset["text"] = newsgroups_train.data
        dataset["label"] = newsgroups_train.target
        dataset = dataset[:num_records]
        gconfig = GalileoDataConfig(
            text=dataset["text"], labels=dataset["label"], split=split.value
        )
        dataquality.log_batch_input_data(gconfig)

    for split in [Split.training, Split.test]:
        for _ in range(num_logs):
            emb = [[random() for _ in range(20)] for _ in range(num_records)]
            probs = [[random() for _ in range(4)] for _ in range(num_records)]
            epoch = 0
            ids = list(range(num_records))

            model_config = GalileoModelConfig(
                emb=emb, probs=probs, split=split.value, epoch=epoch, ids=ids
            )
            dataquality.log_model_outputs(model_config)
