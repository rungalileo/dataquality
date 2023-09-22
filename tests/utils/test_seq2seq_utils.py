from dataclasses import dataclass
from typing import List, Set, Tuple
from unittest import mock

import numpy as np
import pytest
import torch

from dataquality.exceptions import GalileoException
from dataquality.loggers.model_logger.seq2seq import Seq2SeqModelLogger
from dataquality.schemas.seq2seq import TOP_K, LogprobData
from dataquality.utils.seq2seq import (
    remove_padding,
)
from dataquality.utils.seq2seq.generation import (
    generate_sample_output,
)
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
    process_sample_logprobs,
)
from dataquality.utils.seq2seq.offsets import (
    rollup_offset_mapping,
)


@mock.patch("dataquality.utils.seq2seq.generation.process_sample_logprobs")
@mock.patch("dataquality.utils.seq2seq.generation.get_top_logprob_indices")
def test_generate_sample_output(
    mock_get_top_logprob_indices: mock.Mock, mock_process_sample_logprobs: mock.Mock
) -> None:
    """Test the logic for generating over a single sample.

    Things to mock:
        - Anything model related
            - Assume .generate() works
            - Assume getting logits works
        - Mock the tokenizer
            - Mock the tokenize function
            - Mock the decode function
        - Mock get_top_logprob_indices - check inputs
        - Mock process_sample_logprobs since we test this seperately

    Things to test:
        - Check that the fake pad token is removed
        - Check that we have logprobs by checking input to process_sample_logprobs
        - Check that logprobs is correct shape to process_sample_logprobs
        - Check that gen_ids is correct shape to process_sample_logprobs
        - Check that we have the correct attributes in ModelGeneration
    """
    # Mock the tokenizer
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 0]])}

    # Mock the model
    mock_model = mock.MagicMock()
    # Add a fake <pad> token to the generated ids
    mock_model.generate.return_value = torch.tensor([[1, 10, 20, 30]])

    # Mock the model forward function to return random logits
    # for a single batch element
    @dataclass
    class FakeOutput:
        logits: torch.tensor

    # Mock model output and model device
    mock_model.return_value = FakeOutput(torch.rand((1, 3, 20)))
    mock_model.device = torch.device("cpu")

    # Mock generation_config
    mock_generation_config = mock.MagicMock()

    # Mock util helper function. Note we don't mock the return value of
    # mock_get_top_logprob_indices since we only care about its input
    fake_token_logprobs = np.array([-0.5, -0.25, -0.11])
    fake_top_logprob_data = [
        [("A", -0.1), ("B", -1)],
        [("A", -0.1), ("B", -1)],
        [("A", -0.1), ("B", -1)],
    ]
    mock_process_sample_logprobs.return_value = LogprobData(
        token_logprobs=fake_token_logprobs,
        top_logprobs=fake_top_logprob_data,
    )

    with mock.patch("torch.no_grad"):
        model_generation = generate_sample_output(
            "test str", mock_model, mock_tokenizer, mock_generation_config
        )

    # Check logprobs
    logprobs = mock_get_top_logprob_indices.call_args.args[0]
    assert logprobs.shape == (3, 20)
    # Check that we infact have logprobs
    assert np.allclose(1.0, np.sum(np.exp(logprobs), axis=-1))

    # Check ModelGeneration
    # Check gen_ids - Make sure the <pad> token is removed!
    assert np.array_equal(model_generation.generated_ids, np.array([10, 20, 30]))
    assert np.array_equal(
        model_generation.generated_logprob_data.token_logprobs, fake_token_logprobs
    )
    assert model_generation.generated_logprob_data.top_logprobs == fake_top_logprob_data


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

        no_pad_logprobs = remove_padding(sample_labels, "right", sample_logprobs)
        no_pad_top_indices = remove_padding(sample_labels, "right", sample_top_indices)
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0

    # Test padding on the 'left'
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
        no_pad_logprobs = remove_padding(sample_labels, "left", sample_logprobs)
        no_pad_top_indices = remove_padding(sample_labels, "left", sample_top_indices)
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0


def test_process_sample_logprobs():
    """Test process_sample_logprobs

    Ensure that the extracted label logprobs are correct
    and that the top_logprobs data is as expected.
    """
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.decode.return_value = "Fake"

    seq_len = 10
    vocab_size = 100

    fake_logprobs = np.random.rand(seq_len, vocab_size)
    fake_labels = np.arange(seq_len)
    fake_top_indices = np.tile(np.arange(TOP_K), (seq_len, 1))

    logprob_data = process_sample_logprobs(
        fake_logprobs, fake_labels, fake_top_indices, mock_tokenizer
    )

    # Check that the token_logprobs are correct
    token_logprobs = logprob_data.token_logprobs
    for i in range(len(token_logprobs)):
        assert token_logprobs[i] == fake_logprobs[i, fake_labels[i]]

    # Check that the top_logprobs are correct
    top_loprobs = logprob_data.top_logprobs
    assert len(top_loprobs) == seq_len
    assert len(top_loprobs[0]) == TOP_K
    for i, token_top_logprobs in enumerate(top_loprobs):
        pred_top_logprobs = [token[1] for token in token_top_logprobs]
        assert np.allclose(pred_top_logprobs, fake_logprobs[i, :TOP_K])


def test_process_sample_logprobs_incorrect_shape():
    """Test process_sample_logprobs with incorrect label shape"""
    mock_tokenizer = mock.MagicMock()
    seq_len = 10
    vocab_size = 100
    fake_logprobs = np.zeros((seq_len, vocab_size))
    fake_top_indices = np.zeros((seq_len, 5))

    # We expect labels to have shape (seq_len,) when passing
    # to process_sample_logprobs
    incorrect_labels = np.zeros((seq_len, 1))

    with pytest.raises(GalileoException) as excinfo:
        _, _ = process_sample_logprobs(
            fake_logprobs, incorrect_labels, fake_top_indices, mock_tokenizer
        )

    assert (
        "Invalid shape (10, 1), process_sample_logprobs"
        " expects sample_labels to be a 1D array" == str(excinfo.value)
    )


def test_get_top_logprob_indices() -> None:
    """
    Test getting the top 5 logprobs with two different tensor shapes!
        - [seq_len, vc]
        - [bs, seq_len, vc]

    Use arange so that we can expect the exact result!
    """
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    # Test logprobs shape - [seq_len, vocab_size]
    logprobs = np.random.rand(seq_len, vocab_size)
    copy_logprobs = logprobs.copy()
    top_logprob_indices = get_top_logprob_indices(logprobs)
    # Make sure we don't modify the logprobs
    assert np.allclose(logprobs, copy_logprobs)

    # Manually argsort - i.e. the slower way to do this!
    manual_top_logprob_indices = np.argsort(logprobs, axis=-1)[:, -TOP_K:]
    # Note top_logprob_indices is not guaranteed to be sorted
    for token, gt_token in zip(top_logprob_indices, manual_top_logprob_indices):
        token = set(list(token))
        gt_token = set(list(gt_token))
        assert token == gt_token

    # Test logprobs shape - [batch_size, seq_len, vocab_size]
    # Use a simple constructed case where each token has the same "logprobs"
    logprobs = np.tile(np.arange(vocab_size), (batch_size, seq_len, 1))
    top_logprob_indices = get_top_logprob_indices(logprobs)

    assert top_logprob_indices.shape == (batch_size, seq_len, TOP_K)
    # Manually construct desired output based on how partition works
    gt_top_logprob_indices = np.tile(
        np.array([98, 99, 97, 96, 95]), (batch_size, seq_len, 1)
    )
    assert np.allclose(top_logprob_indices, gt_top_logprob_indices)


@pytest.mark.parametrize(
    "offsets,span_offsets,span_positions",
    [
        [
            [(0, 1), (0, 20), (21, 22), (21, 23), (0, 0)],
            [(0, 1), (1, 20), (20, 21), (21, 22), (22, 23)],
            [{0, 1}, {1}, set(), {2, 3}, {3}],
        ],
        [
            [
                (0, 1),
                (0, 1),
                (1, 2),
                (1, 2),
                (2, 3),
                (2, 3),
                (3, 4),
                (3, 4),
                (4, 5),
                (4, 5),
                (5, 6),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (8, 9),
                (9, 10),
                (9, 10),
                (10, 11),
                (10, 11),
                (11, 12),
                (11, 12),
                (11, 12),
                (12, 13),
                (13, 14),
                (13, 14),
                (14, 15),
                (14, 15),
                (15, 16),
                (15, 16),
                (16, 17),
                (16, 17),
                (16, 17),
                (17, 18),
                (17, 18),
                (18, 19),
                (18, 19),
                (19, 20),
                (19, 20),
                (20, 22),
                (21, 22),
                (21, 22),
                (22, 23),
                (22, 23),
            ],
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15),
                (15, 16),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
            ],
            [
                {0, 1},
                {2, 3},
                {4, 5},
                {6, 7},
                {8, 9},
                {10, 11},
                {12},
                {13},
                {14, 15},
                {16, 17},
                {18, 19},
                {20, 21, 22},
                {23},
                {24, 25},
                {26, 27},
                {28, 29},
                {30, 31, 32},
                {33, 34},
                {35, 36},
                {37, 38},
                {39},
                {39, 40, 41},
                {42, 43},
            ],
        ],
        [
            [(3, 6), (6, 7), (7, 9), (9, 18), (18, 25)],
            [(0, 3), (3, 6), (6, 7), (7, 9), (9, 18), (18, 25)],
            [set(), {0}, {1}, {2}, {3}, {4}],
        ],
    ],
)
def test_rollup_spans(
    offsets: List[Tuple[int, int]],
    span_offsets: List[Tuple[int, int]],
    span_positions: List[Set[int]],
) -> None:
    assert rollup_offset_mapping(offsets) == (span_offsets, span_positions)
