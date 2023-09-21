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
from dataquality.schemas.seq2seq import (
    TOP_K,
    AlignedTokenData,
    LogprobData,
    ModelGeneration,
)
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.seq2seq.generation import add_generated_output_to_df
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


@mock.patch("dataquality.utils.seq2seq.align_tokens_to_character_spans")
@mock.patch("dataquality.utils.seq2seq.generate_sample_output")
def test_add_generated_output_to_df(
    mock_generate_sample_output: mock.Mock,
    mock_align_tokens_to_character_spans: mock.Mock,
) -> None:
    """Test the complex vaex batched processing function for generation

    Tings to Mock
        - generate_sample_output: This can be fairly simple. The
        one thing that we want to vary would be the length of things returned
        - tokenizer: decode can be quite simple + encode
        - model + generation_config: just to have as input params
        - align_tokens_to_character_spans: This is already tested so let's just
        mock the output!

    What to test: Main thing to test is that the vaex format is all as expected.
        - Test that each column has the correctly curated result
        - Test that we don't have the extra column caused by vaex's flatten
    """
    # Mock the tokenizer
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.decode.return_value = "Fake output"
    mock_tokenizer.return_value = {"offset_mapping": []}

    # Mock the model
    mock_model = mock.MagicMock()  # Don't actually need any functions for this

    # Mock generation_config
    mock_generation_config = mock.MagicMock()

    # Mock the generation process
    def mock_generate_output() -> ModelGeneration:
        """Create simple dummy ModelGeneration"""
        # Generate fake outputs
        num_fake_tokens = 4  # np.random.randint(3, 10)
        gen_ids = np.arange(num_fake_tokens)
        gen_token_logprobs = np.zeros(num_fake_tokens) - 1
        gen_top_logprobs = [[("A", -1), ("B", -2)] for _ in range(num_fake_tokens)]
        gen_logprob_data = LogprobData(
            token_logprobs=gen_token_logprobs, top_logprobs=gen_top_logprobs
        )
        return ModelGeneration(gen_ids, gen_logprob_data)

    mock_generate_sample_output.return_value = mock_generate_output()

    # Mock aligned output for a single sample
    fake_token_label_offsets = [[(0, 1), (1, 20), (20, 21), (21, 22), (22, 23)]]
    fake_token_label_positions = [[{0, 1}, {1}, {0}, {2, 3}, {3}]]
    mock_align_tokens_to_character_spans.return_value = AlignedTokenData(
        token_label_offsets=fake_token_label_offsets,
        token_label_positions=fake_token_label_positions,
    )

    # Create fake df with vaex
    df = vaex.from_dict({"text": ["Fake Input"] * 100})

    df = add_generated_output_to_df(
        df, mock_model, mock_tokenizer, mock_generation_config
    )

    # Make sure everything is in check!
    assert len(df) == 100
    assert df[C.generated_output.value].tolist() == ["Fake output"] * 100
    # Convert to correct type
    fake_token_label_positions = [[list(val) for val in fake_token_label_positions[0]]]
    assert (
        df[C.generated_token_label_positions.value].tolist()
        == fake_token_label_positions * 100
    )
    fake_token_label_offsets = [[list(val) for val in fake_token_label_offsets[0]]]
    assert (
        df[C.generated_token_label_offsets.value].tolist()
        == fake_token_label_offsets * 100
    )
    generated_token_logprobs = df[C.generated_token_logprobs.value].tolist()
    generated_top_logprobs = df[C.generated_top_logprobs.value].tolist()
    for logprobs, top_logprobs in zip(generated_token_logprobs, generated_top_logprobs):
        num_tokens = len(logprobs)
        assert num_tokens == len(top_logprobs)
        assert np.array_equal(logprobs, np.zeros(num_tokens) - 1)
        assert top_logprobs == [[("A", -1), ("B", -2)] for _ in range(num_tokens)]

    # Make sure that we have removed the column left after flattening
    assert f"____{C.generation_data.value}" not in df.column_names
