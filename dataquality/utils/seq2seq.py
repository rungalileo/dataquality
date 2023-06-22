from typing import Dict, List, Set, Tuple


def _associate_tokens_with_characters(
    input_string: str, offset_mapping: List[Tuple[int, int]]
) -> Dict[int, Set[int]]:
    """For each character in the string, figure out which token positions(s)
    it's associated with.

    The offset_mapping provided by the tokenizer can ignore things like spaces,
    which cause the offsets not to align with the true characters in the input-string.

    This aligns each character in the input_string to its associated token (if it maps
    to one at all)

    ex:
        input_string = "hi cat t"
        offset_mapping = [(0, 2), (3, 6), (7, 8), (7, 8), (0, 0)]
        (0,0) is the end of sequence token, which is ignored (because it's range(0, 0))

        _associate_tokens_with_characters(input_string, offset_mapping) ->
        char_index_to_token_indices = {
            0: {0},
            1: {0},
            2: {},
            3: {1},
            4: {1},
            5: {1},
            6: {},
            7: {2, 3},
        }
    """

    # offset_mapping is something you get from calling
    # `tokenizer.encode_plus(..., return_offsets_mapping=True)`
    # with an HF "Fast" tokenizer.
    char_index_to_token_indices: Dict[int, Set[int]] = {
        i: set() for i in range(len(input_string))
    }

    for token_index, offsets in enumerate(offset_mapping):
        # Map along the offset range (which is a tuple[int, int])
        for i in range(*offsets):
            char_index_to_token_indices[i].add(token_index)

    return char_index_to_token_indices


def _rollup_spans(char_index_to_token_indices: Dict[int, Set[int]]) -> List[Dict]:
    """When adjacent string positions are associated with the same tokens,
    combine them together into a span.

    char_index_to_token_indices may have multiple indices associated with the same
    tokens

    ex:
        {0: {0},
         1: {0},
         2: {},
         3: {1},
         4: {1},
         5: {1},
         6: {},
         7: {2, 3},
         8: {2, 3},
         9: {}}
        would have all indices with the same token positions combined
        [{'offsets': (0, 2), 'token_positions': {0}},
         {'offsets': (2, 3), 'token_positions': {}},
         {'offsets': (3, 6), 'token_positions': {1}},
         {'offsets': (6, 7), 'token_positions': {}},
         {'offsets': (7, 9), 'token_positions': {2, 3}},
         {'offsets': (9, 10), 'token_positions': {}}]
    """

    spans: List[Dict] = []
    left_pos = 0
    right_pos = 0
    cur_token_set = None
    for right_pos in range(len(char_index_to_token_indices)):
        position_token_set = char_index_to_token_indices[right_pos]

        if cur_token_set is None:
            cur_token_set = position_token_set
        if cur_token_set != position_token_set:
            spans.append(
                {
                    "offsets": (left_pos, right_pos),
                    "token_positions": cur_token_set,
                }
            )
            left_pos = right_pos
            cur_token_set = position_token_set
    # Include final span
    spans.append(
        {
            "offsets": (left_pos, right_pos + 1),
            "token_positions": cur_token_set,
        }
    )
    return spans


def align_tokens_with_inputs(
    input_strings: List[str],
    tokenized_inputs: List[List[int]],
    offset_mappings: List[List[Tuple[int, int]]],
) -> Tuple[List[List[int]], List[List[Tuple[int, int]]]]:
    """For each input string and provided tokens, construct the token positions aligned
    to the text, and their character offsets

    The token offsets from the tokenizer will ignore things like spaces, which will mess
    up the token-character alignment if you try to index into the original
    (non-tokenized) string. This aligns that with the spaces and returns the
    space-aligned character offsets for each token
    """
    input_token_positions: List[List[int]] = []
    input_token_offsets: List[List[Tuple[int, int]]] = []
    for input_string, tokens, offset_mapping in zip(
        input_strings, tokenized_inputs, offset_mappings
    ):
        char_index_to_token_indices = _associate_tokens_with_characters(
            input_string, offset_mapping
        )
        spans = _rollup_spans(char_index_to_token_indices)
        span_positions: List[int] = []
        span_offsets: List[Tuple[int, int]] = []
        for span in spans:
            span_positions.append(span["token_positions"])
            span_offsets.append((span["offsets"][0], span["offsets"][1]))
        input_token_positions.append(span_positions)
        input_token_offsets.append(span_offsets)

    return input_token_positions, input_token_offsets
