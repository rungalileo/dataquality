from typing import Dict, List, Set, Tuple

import pytest

from dataquality.utils.seq2seq import _associate_tokens_with_characters, _rollup_spans


@pytest.mark.parametrize(
    "chars,offsets",
    [
        [
            {
                0: {0},
                1: {0},
                2: {},
                3: {1},
                4: {1},
                5: {1},
                6: {},
                7: {2, 3},
                8: {2, 3},
                9: {},
            },
            [
                {"offsets": (0, 2), "token_positions": {0}},
                {"offsets": (2, 3), "token_positions": {}},
                {"offsets": (3, 6), "token_positions": {1}},
                {"offsets": (6, 7), "token_positions": {}},
                {"offsets": (7, 9), "token_positions": {2, 3}},
                {"offsets": (9, 10), "token_positions": {}},
            ],
        ],
        [
            {
                0: {0},
                1: {0},
                2: {},
                3: {1},
                4: {1},
                5: {1},
                6: {},
                7: {2, 3},
                8: {2, 3},
            },
            [
                {"offsets": (0, 2), "token_positions": {0}},
                {"offsets": (2, 3), "token_positions": {}},
                {"offsets": (3, 6), "token_positions": {1}},
                {"offsets": (6, 7), "token_positions": {}},
                {"offsets": (7, 9), "token_positions": {2, 3}},
            ],
        ],
        [
            {0: {0}, 1: {0}, 2: {}, 3: {1}, 4: {1}, 5: {1}, 6: {}, 7: {2, 3}},
            [
                {"offsets": (0, 2), "token_positions": {0}},
                {"offsets": (2, 3), "token_positions": {}},
                {"offsets": (3, 6), "token_positions": {1}},
                {"offsets": (6, 7), "token_positions": {}},
                {"offsets": (7, 8), "token_positions": {2, 3}},
            ],
        ],
    ],
)
def test_rollup_spans(chars: Dict[int, Set[int]], offsets: List[Dict]) -> None:
    assert _rollup_spans(chars) == offsets


@pytest.mark.parametrize(
    "input_string,offset_mapping,char_tokens",
    [
        [
            "hi cat t",
            [(0, 2), (3, 6), (7, 8), (7, 8), (0, 0)],
            {
                0: {0},
                1: {0},
                2: set(),
                3: {1},
                4: {1},
                5: {1},
                6: set(),
                7: {2, 3},
            },
        ],
    ],
)
def test_associate_tokens_with_characters(
    input_string: str,
    offset_mapping: List[Tuple[int, int]],
    char_tokens: Dict[int, Set[int]],
) -> None:
    assert (
        _associate_tokens_with_characters(input_string, offset_mapping) == char_tokens
    )
