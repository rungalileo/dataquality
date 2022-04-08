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
