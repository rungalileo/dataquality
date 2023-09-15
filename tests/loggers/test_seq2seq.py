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
from dataquality.schemas.seq2seq import TOP_K
from dataquality.schemas.split import Split
from dataquality.utils.seq2seq import get_top_logprob_indices
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
    config.tokenizer.decode = lambda x: "Fake"

    batch_size = 4
    seq_len = 20
    vocab_size = 100

    logits = np.random.rand(batch_size, seq_len, vocab_size)
    # Set the locations of the "padded" tokens to 0 for the logits
    for idx, token_labels in enumerate(tokenized_labels):
        logits[idx, len(token_labels) :] = 0

    log_data = dict(
        ids=list(range(batch_size)),
        logits=logits,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    with mock.patch("dataquality.core.log.get_model_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_model_outputs(**log_data)
    ThreadPoolManager.wait_for_threads()
    logger.check_for_logging_failures()
    output_data = vaex.open(f"{test_session_vars.LOCATION}/training/0/*.arrow")
    expected_cols = [
        "id",
        "token_logprobs",
        "top_logprobs",
        "perplexity",
        "split",
        "epoch",
    ]
    assert sorted(output_data.get_column_names()) == sorted(expected_cols)
    assert len(output_data) == 4

    token_logprobs = output_data["token_logprobs"].tolist()
    top_logprobs = output_data["top_logprobs"].tolist()
    perplexities = output_data["perplexity"].tolist()

    for token_labels, sample_token_logprobs, sample_top_logprobs, perplexity in zip(
        tokenized_labels, token_logprobs, top_logprobs, perplexities
    ):
        assert (
            len(token_labels) == len(sample_token_logprobs) == len(sample_top_logprobs)
        )
        # Check all logprobs are < 0
        for token_logprob in sample_token_logprobs:
            assert token_logprob < 0

        # Check that we ignore <pad> token by checking that for each
        # token in sample_top_logprobs the top logprobs are not all equal.
        # Additionally check the general structure of top_logprobs
        for token_top_logprobs in sample_top_logprobs:
            assert len(token_top_logprobs) == TOP_K

            logprobs = [candidate[1] for candidate in token_top_logprobs]
            assert not np.allclose(logprobs[0], logprobs)

            assert np.alltrue(
                [candidate[0] == "Fake" for candidate in token_top_logprobs]
            )

        assert perplexity > 0


def test_model_logger_remove_padding() -> None:
    """Test _remove_padding and _retrieve_sample_labels

    Ensure that _remove_padding removes the correct tokens for each
    sample based on the `sample_labels` and the tokenzier padding direction.
    """
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
    # First test removing from right padding
    mock_tokenizer.padding_side = "right"
    config.tokenizer = mock_tokenizer

    batch_size = 4
    max_seq_len = 20
    vocab_size = 100

    logprobs = np.random.rand(batch_size, max_seq_len, vocab_size)
    # Set pad tokens on the right with -1
    for idx, token_labels in enumerate(tokenized_labels):
        logprobs[idx, len(token_labels) :] = -1
    # Create the top indices just using logits
    top_indices = logprobs[:, :, 5]

    # Note we don't differentiate between logits and logprobs for this test
    log_data = dict(
        ids=list(range(batch_size)),
        logits=logprobs,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    for sample_id, (sample_logprobs, sample_top_indices) in enumerate(
        zip(logprobs, top_indices)
    ):
        sample_labels = logger._retrieve_sample_labels(sample_id)
        # Test the retrieve samples method
        assert np.allclose(sample_labels, tokenized_labels[sample_id])

        no_pad_logprobs, no_pad_top_indices = logger._remove_padding(
            sample_labels, sample_logprobs, sample_top_indices
        )
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0

    # Test padding on the 'left'
    logger.logger_config.tokenizer.padding_side = "left"
    logprobs = np.random.rand(batch_size, max_seq_len, vocab_size)
    # Set pad tokens on the left with -1
    for idx, token_labels in enumerate(tokenized_labels):
        logprobs[idx, : -len(token_labels)] = -1
    # Create the top indices just using logits
    top_indices = logprobs[:, :, 5]

    for sample_id, (sample_logprobs, sample_top_indices) in enumerate(
        zip(logprobs, top_indices)
    ):
        sample_labels = logger._retrieve_sample_labels(sample_id)
        no_pad_logprobs, no_pad_top_indices = logger._remove_padding(
            sample_labels, sample_logprobs, sample_top_indices
        )
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0
