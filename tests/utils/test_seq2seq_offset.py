from typing import Callable, Generator, List, Set, Tuple
from unittest.mock import Mock

import pytest
import vaex
from datasets import Dataset
from transformers import GenerationConfig, T5ForConditionalGeneration

import dataquality as dq
from dataquality.integrations.seq2seq.hf import watch
from dataquality.loggers.data_logger.seq2seq import Seq2SeqDataLogger
from dataquality.schemas.seq2seq import Seq2SeqInputCols as C
from dataquality.schemas.task_type import TaskType
from dataquality.utils.seq2seq.offsets import (
    get_cutoff_from_truncated_tokenization,
    get_position_of_last_offset_target,
    rollup_offset_mapping,
)
from tests.conftest import tokenizer_T5


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


def test_get_position_of_last_offset_input(
    set_test_config: Callable, cleanup_after_use: Generator
):
    """
    Test that get_position_of_last_offset_input returns the correct cut-off point for
    the input text string.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_model = Mock(spec=T5ForConditionalGeneration)
    mock_model.device = "cpu"
    mock_generation_config = Mock(spec=GenerationConfig)
    watch(mock_model, tokenizer_T5, mock_generation_config, max_input_tokens=4)

    input_1, input_2 = "dog dog dog done - tricked you", "bird"
    ds = Dataset.from_dict(
        {
            "id": [0, 1],
            "input": [input_1, input_2],
            "target": ["a b c d e f g h i j", "1"],
        }
    )
    dq.log_dataset(ds, text="input", label="target", split="train")

    data_logger = Seq2SeqDataLogger()
    in_frame_split = vaex.open(
        f"{data_logger.input_data_path}/training/*.{data_logger.INPUT_DATA_FILE_EXT}"
    )
    input_offsets = get_cutoff_from_truncated_tokenization(
        in_frame_split, C.text, tokenizer_T5, data_logger.logger_config.max_input_tokens
    ).tolist()

    assert len(input_offsets) == 2
    # The EOS token is always the last token, which we don't count for the cutoff point
    # The 4 tokens are 3 tokens 'dog' + EOS
    assert input_1[: input_offsets[0]] == "dog dog dog"
    # We only have 1 token 'bird' (got to the end of the string)
    assert input_2[: input_offsets[1]] == "bird"


def test_get_position_of_last_offset_target(
    set_test_config: Callable, cleanup_after_use: Generator
):
    """
    Test that get_position_of_last_offset_target returns the correct cut-off point for
    the target text string.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_model = Mock(spec=T5ForConditionalGeneration)
    mock_model.device = "cpu"
    mock_generation_config = Mock(spec=GenerationConfig)
    watch(mock_model, tokenizer_T5, mock_generation_config, max_target_tokens=6)

    target_1, target_2 = "cat cat cat cat cat done", "cat"
    ds = Dataset.from_dict(
        {
            "id": [0, 1],
            "input": ["a b c d e f g h i j", "1"],
            "target": [target_1, target_2],
        }
    )
    dq.log_dataset(ds, text="input", label="target", split="train")

    data_logger = Seq2SeqDataLogger()
    in_frame_split = vaex.open(
        f"{data_logger.input_data_path}/training/*.{data_logger.INPUT_DATA_FILE_EXT}"
    )
    target_offsets = get_position_of_last_offset_target(
        in_frame_split, C.token_label_offsets
    ).tolist()

    assert len(target_offsets) == 2
    # The EOS token is always the last token, which we don't count for the cutoff point
    # The 6 tokens are 5 tokens 'cat' + EOS
    assert target_1[: target_offsets[0]] == "cat cat cat cat cat"
    # We only have 1 token 'cat' (got to the end of the string)
    assert target_2[: target_offsets[1]] == "cat"
