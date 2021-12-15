import os
from random import random
from typing import List, Optional

import numpy as np
import pandas as pd
import vaex
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm

import dataquality
from dataquality.schemas.split import Split
from tests.conftest import LOCATION, SPLITS, SUBDIRS, TEST_PATH

NUM_RECORDS = 50
NUM_LOGS = 30
MULTI_LABEL_NUM_TASKS = 10


def validate_uploaded_data(
    expected_num_records: int = None,
    meta_cols: Optional[List] = None,
    multi_label=False,
) -> None:
    """
    Checks for testing
    """
    expected_num_records = expected_num_records or NUM_RECORDS * NUM_LOGS
    meta_cols = meta_cols or {}
    for split in SPLITS:
        # Output data
        split_output_data = {}
        for subdir in SUBDIRS:
            file_path = f"{TEST_PATH}/{split}/{subdir}/{subdir}.hdf5"
            # Ensure files were cleaned up
            data = vaex.open(file_path)
            prob_cols = data.get_column_names(regex="prob*")
            for c in data.get_column_names():
                if c in prob_cols + ["emb"]:
                    assert not np.isnan(data[c].values).any()
                else:
                    vals = data[c].values
                    assert all([i is not None and i != "nan" for i in vals])
            split_output_data[subdir] = data

        data = split_output_data["data"]
        emb = split_output_data["emb"]
        prob = split_output_data["prob"]

        for c in meta_cols:
            assert c in data.get_column_names()
        assert list(emb.get_column_names()) == ["id", "emb"]

        assert "data_schema_version" in data.columns
        assert len(data) == len(emb) == len(prob) == expected_num_records
        assert (
            sorted(data["id"].unique())
            == sorted(emb["id"].unique())
            == sorted(prob["id"].unique())
        )
        if multi_label:
            for c in prob.get_column_names():
                assert c == "id" or c.startswith("prob_") or c.startswith("gold_")
                assert c != "prob" and c != "gold"
            assert (
                len(prob.get_column_names(regex="prob*"))
                == len(prob.get_column_names(regex="gold*"))
                == len(data.get_column_names(regex="pred*"))
                == MULTI_LABEL_NUM_TASKS
            )


def validate_cleanup_data():
    """
    Checks for testing
    """
    for split in SPLITS:
        # Ensure files were cleaned up
        assert not os.path.isdir(f"{LOCATION}/{split}")


def _log_data(
    num_records=NUM_RECORDS,
    num_logs=NUM_LOGS,
    unique_ids=True,
    num_emb=20,
    meta=None,
    multi_label=False,
) -> None:
    """
    Logs some mock data to disk
    """
    meta = meta or {}
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

        if multi_label:
            labels = []
            for i in range(len(dataset)):
                task_labels = np.random.randint(10, size=MULTI_LABEL_NUM_TASKS)
                labels.append(task_labels)
            dataquality.set_tasks_for_run(
                [str(i) for i in range(MULTI_LABEL_NUM_TASKS)]
            )
        else:
            labels = dataset["label"]
            dataquality.set_labels_for_run([str(i) for i in range(len(set(labels)))])
        dataquality.log_input_data(
            text=dataset["text"], labels=labels, split=split.value, meta=meta
        )

    num_labels_in_task = np.random.randint(low=1, high=10, size=MULTI_LABEL_NUM_TASKS)
    if multi_label:
        run_labels = [[str(i) for i in range(tl)] for tl in num_labels_in_task]
        dataquality.set_labels_for_run(run_labels)
    for split in [Split.training, Split.test]:
        for ln in tqdm(range(num_logs)):
            emb = [[random() for _ in range(num_emb)] for _ in range(num_records)]
            if multi_label:
                probs = []
                for i in range(num_records):
                    probs_per_task = []
                    for num_labels in num_labels_in_task:
                        probs_per_task.append(np.random.rand(num_labels))
                    probs.append(probs_per_task)
            else:
                probs = np.random.rand(num_records, len(set(labels)))
            epoch = 0

            # Need unique ids
            if unique_ids:
                r = range(ln * num_records, (ln + 1) * num_records)
                ids = list(r)
            else:
                ids = list(range(num_records))

            dataquality.log_model_outputs(
                emb=emb, probs=probs, split=split.value, epoch=epoch, ids=ids
            )
