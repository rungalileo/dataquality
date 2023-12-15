from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pyarrow as pa
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast
from vaex import DataFrame, register_function

from dataquality.schemas.seq2seq import AlignedTokenData
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC


def _handle_overlapping_offsets(
    span_offsets: List[Tuple[int, int]],
    span_mapping: Dict[Tuple[int, int], Set[int]],
    cur_offset: Tuple[int, int],
    cur_tok_pos: int,
) -> None:
    """Handle the overlap of the last span's end to the current start

    (ie [(0, 1), (0, 20)])
    Traverse backwards until you find a previous offset that is less
    than the current start and add the current token pos to the previous spans
    """
    cur_start, cur_end = cur_offset
    prev_tok_pos = cur_tok_pos
    # Start 1 prior to the current span, so -1
    for prev_span_idx in range(len(span_offsets) - 1, -1, -1):
        prev_tok_pos -= 1
        prev_start, prev_end = span_offsets[prev_span_idx]

        # Base case - we hit a previous span that isn't overlapping. Break out
        if prev_end <= cur_start:
            break

        # First, add the current token position to the previous span, since it
        # overlaps
        span_mapping[(prev_start, prev_end)].add(cur_tok_pos)
        # If the previous span's start is less than the current start,
        # create the span between them. ex: [(*0*, 2), (*1*, 3)] -> (0, 1)
        # And add only the previous span's token position
        if prev_start < cur_start:
            new_offset = (prev_start, cur_start)
            span_offsets.append(new_offset)
            span_mapping[new_offset].add(prev_tok_pos)

        # If the previous span's end is greater than the current start,
        # then create that span in between them
        # ex: [(0, *2*), (*1*, 3)] -> (1, 2)
        # BUT if the current start is the same as the prior span's start,
        # that span already exists ie [(0, 2), (0, 3)] -> (0, 2), so ignore
        if prev_end > cur_start and prev_start != cur_start:
            new_offset = (cur_start, prev_end)
            span_offsets.append(new_offset)
            # We add both token positions because this range falls between both
            span_mapping[new_offset].update({prev_tok_pos, cur_tok_pos})

        # If the previous span's end is _less_ then the current end
        # (it could be equal OR less), we need to create a span in between them,
        # and assign the position of the current idx (it's end-exclusive)
        # ex: [(0, 2), (1, 3)] -> (2, 3)
        if prev_end < cur_end:
            new_offset = (prev_end, cur_end)
            span_offsets.append(new_offset)
            span_mapping[new_offset].add(cur_tok_pos)

        # In some edge cases, the previous span's end is less than current end,
        # and the previous span's start is _greater_ than the current spans
        # start. This means the current span is _cutting_ into the previous span
        # ex: [(0, 2), (1, 3)] -> would create
        # [(0, 2), (0, 1), (1, 2), (2, 3)]. We need to remove the previous
        # span (0, 2) because it's no longer valid. We cannot return overlapping
        # spans, and that span is now encapsulated by the newly created ones
        if prev_start < cur_start < prev_end:
            del span_offsets[prev_span_idx]
            span_mapping.pop((prev_start, prev_end))


def _add_spans_offsets(
    span_offsets: List[Tuple[int, int]],
    span_mapping: Dict[Tuple[int, int], Set[int]],
    cur_offset: Tuple[int, int],
    cur_tok_pos: int,
) -> None:
    cur_start, cur_end = cur_offset
    # First element
    if not cur_tok_pos:
        span_offsets.append(cur_offset)
        span_mapping[cur_offset].add(cur_tok_pos)
        return

    # Check for the eos token. Ignore it
    if cur_start == cur_end:
        return

    # Compare the current spans offsets to the previous
    last_start, last_end = span_offsets[-1]
    # Base case, last span's end is current span start
    if last_end == cur_start:
        span_offsets.append(cur_offset)
        span_mapping[cur_offset].add(cur_tok_pos)
    # Gap in tokens (like a space). fill the gap with no tokens
    elif last_end < cur_start:
        # Fill the empty space with nothing
        span_offsets.append((last_end, cur_start))
        span_mapping[(last_end, cur_start)] = set()
        # Then add the current span
        span_offsets.append(cur_offset)
        span_mapping[cur_offset].add(cur_tok_pos)

    # Final case: overlap of last end to the current start (ie [(0, 1), (0, 20)])
    # Traverse backwards until you find a previous offset that is less
    # than the current start and add the current token pos to the previous spans
    else:
        _handle_overlapping_offsets(span_offsets, span_mapping, cur_offset, cur_tok_pos)


def rollup_offset_mapping(
    offset_mapping: List[Tuple[int, int]]
) -> Tuple[List[Tuple[int, int]], List[Set[int]]]:
    """For a single sample's tokenizer offsets, extract the character level offsets
    for each token.

    Character level offsets align each token with it's character index in a sample.
    They follow rules:
        1. There must not be a gap between offsets. The end of one must be the beginning
            of the next
        2. The span offsets must be strictly increasing

    Each offset has 0 or more token positions associated. These positions indicate which
    tokens exist in the range indicated by the offset.
    ex: {'offsets': (0, 3), 'token_positions': {0, 1}}} means that tokens 0 and 1
    from the tokenizer are encapsulated by the character range (0, 3]

    We take overlapping ranges and gaps between ranges, and fill them in contiguously

    ex:
        offset_mapping = [(0, 1), (0, 20), (22, 23), (0, 0)]

        is rolled into
            [
                {'offsets': (0, 1), 'token_positions': {0, 1}},
                {'offsets': (1, 20), 'token_positions': {1}},
                {'offsets': (20, 22), 'token_positions': {}},
                {'offsets': (22, 23), 'token_positions': {2}},
            ]

        and returned as
        [(0, 1), (1, 20), (20, 22), (22, 23)], [{0, 1}, {1}, {}, {2}]
    """
    # In rare cases, we can have a completely empty offsets_mapping
    # when we have a tokenized empty string with no special chars.
    if len(offset_mapping) == 0:
        return [], []

    # The ordered span offsets for backtracking in case of an overlap
    # Like [(0, 1), (1, 3), (3, 4)]
    # We guarantee that the spans here will be contiguous, non-repeating and
    # non-overlapping
    span_offsets: List[Tuple[int, int]] = []
    # The mapping of a given offset range to their corresponding token positions
    # Each key will be in-line with the `span_offsets` list
    # ex: {(0, 1): {0}, {1, 3}: {}, {3, 4}: {2, 3, 4}}
    span_mapping: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    # If there are uncaptured chars in the beginning of the string (like spaces),
    # record those
    first_offset_loc = offset_mapping[0][0]
    if first_offset_loc != 0:
        span_mapping[(0, first_offset_loc)] = set()

    for cur_tok_pos, cur_offset in enumerate(offset_mapping):
        _add_spans_offsets(span_offsets, span_mapping, cur_offset, cur_tok_pos)

    return list(span_mapping.keys()), list(span_mapping.values())


def align_tokens_to_character_spans(
    samples_offsets: List[List[Tuple[int, int]]], disable_tqdm: bool = False
) -> AlignedTokenData:
    """Iterates through each samples offsets and creates character-aligned spans

    Parameters:
    -----------
    disable_tqdm: bool
        Flag for disabling tqdm. Used generally when we are calling
        align_tokens_to_character_spans over small (e.g. 1 sample) batches
    """
    all_offsets = []
    all_token_positions = []
    for offset_mapping in tqdm(
        samples_offsets,
        leave=False,
        desc="Aligning characters with tokens",
        disable=disable_tqdm,
    ):
        offsets, token_positions = rollup_offset_mapping(offset_mapping)
        all_offsets.append(offsets)
        all_token_positions.append(token_positions)

    return AlignedTokenData(
        token_label_offsets=all_offsets, token_label_positions=all_token_positions
    )


def add_input_cutoff_to_df(
    df: DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    text_col: str,
    max_tokens: Optional[int] = None,
) -> DataFrame:
    """
    Find the cutoff point in string coresponding to the last token.

    We tokenize the text and truncate after max_tokens tokens, i.e., we only keep the
    first max_tokens tokens. To find the position in the text corresponding to the
    last token we use the offset_mapping returned by the tokenizer.
    """
    df_copy = df.copy()

    @register_function()
    def _get_position_of_last_offset_input(texts: pa.array) -> np.ndarray:
        """Tokenize the texts and find the position of the last offset."""
        offset_mapping = tokenizer(
            texts.to_pylist(),
            truncation=True,
            max_length=max_tokens,
            return_offsets_mapping=True,
        )["offset_mapping"]
        # At least for the T5 tokenizer, the offset_mapping contains an extra offset
        # (0,0) that is added at the end of the list (even when input = "").
        # We skip it and take the previous to last element with offsets[-2].
        input_cut_off = np.array(
            [offsets[-2][-1] if len(offsets) >= 2 else 0 for offsets in offset_mapping],
            dtype="int32",
        )
        return input_cut_off

    df_copy[S2SIC.input_cutoff.value] = df_copy[
        text_col
    ]._get_position_of_last_offset_input()
    # We materialize to run all data through tokenizer once and avoid running it
    # multiple times when exporting the df
    df_copy = df_copy.materialize(S2SIC.input_cutoff.value)
    return df_copy


def add_target_cutoff_to_df(df: DataFrame, target_offsets_col: str) -> DataFrame:
    """
    Look at the last offset of the tokenized target to find the position of the last
    character of the target string that was used by the model.
    Note that typically the model does not use the entire target during teacher forcing
    and there is a cut-off point (for example 128 tokens, or 512 tokens, etc).
    """
    df_copy = df.copy()

    @register_function()
    def _get_position_of_last_offset_target(offsets: pa.array) -> np.ndarray:
        return np.array(
            [
                offsets_row[-1][-1].as_py() if len(offsets_row) > 0 else 0
                for offsets_row in offsets
            ],
            dtype="int32",
        )

    df_copy[S2SIC.target_cutoff.value] = df_copy[
        target_offsets_col
    ]._get_position_of_last_offset_target()
    return df_copy


def align_response_tokens_to_character_spans(
    tokenizer: PreTrainedTokenizerFast,
    tokenized_response: List[int],
    max_input_tokens: Optional[int],
) -> Tuple[AlignedTokenData, str]:
    """Decodes then re-tokenizes the isolated response to get the character alignments

    TODO This can prob be done with just tokenizing the "target" in isolation!!
        Specifically, we tokenize the Targets, then we figure out the index
        of the last token from the tokenized_response and find where that is
        in the offset map and slice the offset map accordingly.
        This may also avoid strange space issues with tokenizers hanlding words
        at the start of a document.

    Return:
        -------
        aligned_token_data: AlignedTokenData
            Aligned token data for a single Response - batch dim = 1.
        decoded_response: str
            The string representation of the Response, used as the
            Target string in the console. Note: we do not remove
            special characters, so these will appear in the console!
    """
    decoded_response = tokenizer.decode(tokenized_response)
    re_tokenized_response = tokenizer(
        [decoded_response],
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        # I believe that this should be handled! We can prob set to None
        max_length=max_input_tokens,
    )
    return (
        align_tokens_to_character_spans(
            re_tokenized_response["offset_mapping"], disable_tqdm=True
        ),
        decoded_response,
    )
