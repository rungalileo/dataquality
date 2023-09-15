from typing import List, Set, Tuple
from unittest import mock

import numpy as np
import pytest

from dataquality.loggers.model_logger.seq2seq import Seq2SeqModelLogger
from dataquality.schemas.seq2seq import TOP_K
from dataquality.utils.seq2seq import (get_top_logprob_indices,
                                       rollup_offset_mapping)


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
