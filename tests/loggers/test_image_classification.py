import numpy as np
import vaex

import dataquality
import dataquality as dq
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION


def test_duplicate_ids_augmented(set_test_config, cleanup_after_use) -> None:
    # This test is to ensure that duplicate ids caused by augmentation are not logged
    set_test_config(task_type="image_classification")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = ["A", "C", "B", "A", "C"]
    ids = [0, 1, 2, 3, 4]

    dq.set_labels_for_run(["A", "B", "C"])
    for split in ["training", "validation", "test"]:
        dq.log_data_samples(texts=text_inputs, labels=gold, split=split, ids=ids)
        dq.set_split(split)
        dq.set_epoch(0)
        dq.log_model_outputs(
            embs=np.random.rand(3, 10),
            logits=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ids=[0, 0, 1],
        )
        dq.log_model_outputs(
            embs=np.random.rand(3, 10),
            logits=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ids=[0, 0, 1],
        )
        dq.set_epoch(1)
        dq.log_model_outputs(
            embs=np.random.rand(3, 10),
            logits=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            ids=[0, 0, 1],
        )
    ThreadPoolManager.wait_for_threads()
    validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
    validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")
