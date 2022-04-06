import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

import dataquality


def test_please_work(cleanup_after_use, set_test_config):
    set_test_config(task_type="text_classification")

    dataquality.log_input_data(
        text=["some text"], labels=["a_label"], split="training", ids=[0]
    )
    dataquality.set_labels_for_run(["a_label", "not_a_label"])
    dataquality.set_split("training")
    dataquality.set_epoch(0)
    dataquality.log_model_outputs(embs=[[0.3, 0.4]], logits=[[0.4, 0.2]], ids=[0])
    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()
    print("pausing")


def test_explainatron(cleanup_after_use, set_test_config):
    set_test_config(task_type="text_classification")

    newsgroups = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    dataset = pd.DataFrame()
    dataset["text"] = newsgroups.data
    label_ind = newsgroups.target_names
    dataset["label"] = [label_ind[i] for i in newsgroups.target]
    dataset = dataset[:100]
    dataquality.log_input_data(
        text=dataset["text"],
        labels=dataset["label"],
        split="training",
        ids=list(range(len(dataset["text"]))),
    )

    dataquality.set_labels_for_run(list(set(dataset["label"])))

    num_rows = len(dataset)
    dataquality.set_split("training")

    for epoch in range(5):
        dataquality.set_epoch(epoch)
        embs = np.random.rand(num_rows, 40)
        logits = np.random.rand(num_rows, len(set(dataset["label"])))
        ids = list(range(num_rows))
        dataquality.log_model_outputs(
            embs=embs,
            logits=logits,
            ids=ids,
            meta={"explainatron_meta": ["please_work"] * num_rows},
        )

    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()
    print("pausing, maybe should do some testing here")
    # dataquality.finish()
