from typing import Callable
from unittest import mock

import datasets
import numpy as np
import pandas as pd
import pytest
import vaex

import dataquality as dq
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.seq2seq import Seq2SeqDataLogger
from dataquality.loggers.model_logger.seq2seq import Seq2SeqModelLogger
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import TestSessionVariables, tokenizer


@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        vaex.from_dict(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        datasets.Dataset.from_dict(
            dict(
                summary=["summary 1", "summary 2", "summary 3"],
                title=["title_1", "title_2", "title_3"],
                my_id=[1, 2, 3],
            )
        ),
    ],
)
def test_log_dataset(
    dataset: DataSet,
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")
    logger = Seq2SeqDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dq.set_tokenizer(tokenizer)
        dq.log_dataset(
            dataset, text="summary", label="title", id="my_id", split="train"
        )

        assert logger.texts == ["summary 1", "summary 2", "summary 3"]
        assert logger.labels == ["title_1", "title_2", "title_3"]
        assert logger.ids == [1, 2, 3]
        assert logger.split == Split.training

    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/data_0.arrow")
    expected_cols = [
        "id",
        "split",
        "text",
        "label",
        "token_label_positions",
        "token_label_offsets",
    ]
    assert sorted(df.get_column_names()) == sorted(expected_cols)


def test_log_dataset_no_tokenizer(set_test_config: Callable) -> None:
    set_test_config(task_type="seq2seq")
    df = pd.DataFrame(
        {
            "summary": ["summary 1", "summary 2", "summary 3"],
            "title": ["title_1", "title_2", "title_3"],
            "my_id": [1, 2, 3],
        }
    )
    logger = Seq2SeqDataLogger()
    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        with pytest.raises(AssertionError) as e:
            dq.log_dataset(df, text="summary", label="title", id="my_id", split="train")
    assert str(e.value) == (
        "You must set your tokenizer before logging. Use `dq.set_tokenizer`"
    )


def test_log_model_outputs(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")

    tokenized_labels = [
        np.arange(10).tolist(),
        np.arange(18).tolist(),
        np.arange(20).tolist(),
        np.arange(4).tolist(),
    ]

    config = mock.MagicMock()
    config.id_to_tokens = {}
    config.id_to_tokens["training"] = dict(zip(list(range(4)), tokenized_labels))
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.padding_side = "right"
    config.tokenizer = mock_tokenizer

    batch_size = 4
    seq_len = 20
    vocab_size = 100

    logits = np.random.rand(batch_size, seq_len, vocab_size)
    # Set the locations of the "padded" tokens to -100 for the logits
    # TODO how can we make sure we are removing the padding tokens correctly??
    for idx, token_labels in enumerate(tokenized_labels):
        logits[idx, len(token_labels) :] = -100

    log_data = dict(
        ids=list(range(batch_size)),
        logits=logits,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    # Because we have our own instance of the logger, we just replace this function
    # in place so we don't have to deal with the softmax
    # TODO: Figure out how to handle the softmax!
    # logger.convert_logits_to_probs = lambda logits: logits
    with mock.patch("dataquality.core.log.get_model_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_model_outputs(**log_data)
    ThreadPoolManager.wait_for_threads()
    logger.check_for_logging_failures()
    output_data = vaex.open(f"{test_session_vars.LOCATION}/training/0/*.arrow")
    expected_cols = [
        "id",
        "token_deps",
        "token_gold_logprobs",
        "token_top_logprobs",
        "perplexity",
        "split",
        "epoch",
        "data_error_potential",
    ]
    assert sorted(output_data.get_column_names()) == sorted(expected_cols)
    assert len(output_data) == 4

    token_gold_logprobs = output_data["token_gold_logprobs"].tolist()
    token_top_logprobs = output_data["token_top_logprobs"].tolist()
    token_deps = output_data["token_deps"].tolist()

    for token_labels, token_dep, token_gold_logprob, token_top_logprob in zip(
        tokenized_labels, token_deps, token_gold_logprobs, token_top_logprobs
    ):
        assert (
            len(token_labels)
            == len(token_dep)
            == len(token_gold_logprob)
            == len(token_top_logprob)
        )
        assert -100 not in token_gold_logprob
        for dep in token_dep:
            assert 0 <= dep <= 1
