import numpy as np
import pytest

import dataquality as dq
from dataquality.exceptions import GalileoException


def test_duplicate_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_multi_label")
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = [["A", "C", "B"]] * 5
    ids = [0, 1, 2, 3, 4]

    dq.log_input_data(text=text_inputs, labels=gold, split="validation", ids=ids)
    dq.log_input_data(text=text_inputs, labels=gold, split="training", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_input_data(text=text_inputs, labels=gold, split="validation", ids=ids)

    dq.log_input_data(text=text_inputs, labels=gold, split="test", ids=ids)

    with pytest.raises(GalileoException):
        dq.log_input_data(text=text_inputs, labels=gold, split="test", ids=ids)


def test_duplicate_output_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_multi_label")
    num_samples = 5
    num_tasks = 4
    classes_per_task = 3
    text_inputs = [
        "what movies star bruce willis",
        "show me films with drew barrymore from the 1980s",
        "what movies starred both al pacino and robert deniro",
        "find me all of the movies that starred harold ramis and bill murray",
        "find me a movie with a quote about baseball in it",
    ]
    gold = [["A", "C", "B", "D"]] * 5
    ids = list(range(5))

    dq.log_input_data(text=text_inputs, labels=gold, split="validation", ids=ids)
    dq.log_input_data(text=text_inputs, labels=gold, split="training", ids=ids)

    emb = np.random.rand(num_samples, 100)
    logits = [[np.random.rand(classes_per_task)] * num_tasks] * num_samples
    ids = list(range(5))
    dq.log_model_outputs(emb=emb, logits=logits, ids=ids, split="training", epoch=0)
    dq.log_model_outputs(emb=emb, logits=logits, ids=ids, split="training", epoch=0)

    with pytest.raises(GalileoException) as e:
        dq.get_data_logger().upload()

    assert str(e.value).startswith("It seems as though you do not have unique ids")
