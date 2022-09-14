import os
from functools import partial
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
MULTI_LABEL_NUM_TASKS = 4


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
            # epoch = 0 for general testing
            try:
                file_path = f"{TEST_PATH}/{split}/0/{subdir}/{subdir}.hdf5"
                data = vaex.open(file_path)
            except FileNotFoundError:
                # Handle autolog test
                file_path = f"{TEST_PATH}/{split}/1/{subdir}/{subdir}.hdf5"
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
        # Since we log logits, the probs should all sum to 1
        for col in prob.get_column_names(regex="prob*"):
            probs = prob[col].values
            assert np.allclose(1, np.sum(probs, axis=-1))
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


def _log_text_classification_data(
    num_records=NUM_RECORDS,
    num_logs=NUM_LOGS,
    unique_ids=True,
    num_embs=20,
    meta=None,
    multi_label=False,
) -> None:
    """
    Logs mock data for text classification/multi-label to disk
    """
    meta = meta or {}
    # Log train/test data
    num_labels_in_task = np.random.randint(low=1, high=10, size=MULTI_LABEL_NUM_TASKS)
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
                task_labels = [np.random.randint(i) for i in num_labels_in_task]
                labels.append(task_labels)
            dataquality.set_tasks_for_run(
                [str(i) for i in range(MULTI_LABEL_NUM_TASKS)], binary=False
            )
        else:
            labels = dataset["label"]
            dataquality.set_labels_for_run([str(i) for i in range(len(set(labels)))])
        dataquality.set_split(split)
        func = partial(
            dataquality.log_data_samples,
            texts=dataset["text"],
            ids=list(range(len(dataset))),
            meta=meta,
        )
        if multi_label:
            func(task_labels=labels)
        else:
            func(labels=labels)

    if multi_label:
        run_labels = [[str(i) for i in range(tl)] for tl in num_labels_in_task]
        dataquality.set_labels_for_run(run_labels)
    for split in [Split.training, Split.test]:
        for ln in tqdm(range(num_logs)):
            embs = [[random() for _ in range(num_embs)] for _ in range(num_records)]
            if multi_label:
                logits = []
                for i in range(num_records):
                    logits_per_task = []
                    for num_labels in num_labels_in_task:
                        logits_per_task.append(np.random.rand(num_labels))
                    logits.append(logits_per_task)
            else:
                logits = np.random.rand(num_records, len(set(labels)))
            epoch = 0

            # Need unique ids
            if unique_ids:
                r = range(ln * num_records, (ln + 1) * num_records)
                ids = list(r)
            else:
                ids = list(range(num_records))

            dataquality.set_epoch(epoch)
            dataquality.set_split(split)
            dataquality.log_model_outputs(embs=embs, logits=logits, ids=ids)
