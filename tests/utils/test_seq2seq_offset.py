from typing import List, Set, Tuple

import pyarrow as pa
import pytest
import vaex

from dataquality.utils.seq2seq.offsets import (
    rollup_offset_mapping,
)


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


def test_get_position_of_last_offset_target():
    """
    Test that get_position_of_last_offset_target returns the correct cut-off point for
    the target text string.
    """
    vaex.from_arrays(token_label_offsets=pa.array([1, 2, 3, 3]))

    assert False


def test_get_position_of_last_offset_input():
    """
    Test that get_position_of_last_offset_input returns the correct cut-off point for
    the input text string.
    """
    assert False
