from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

from scipy.special import log_softmax

import numpy as np
import pyarrow as pa
import torch
import vaex
from tqdm.auto import tqdm
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.schemas.seq2seq import AlignedTokenData, TOP_LOGPROBS_SCHEMA
from dataquality.schemas.seq2seq import Seq2SeqInputCols as IC
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C


def get_top_logprob_indices(
    logprobs:np.ndarray, k: int = 5
):
    """Extract per-token top-k logprobs

    logprobs can either be at the sample level or batch level.

    In both situations, we compute the top logprobs along the final (-1)
    vocab dimension. We use `np.argpartition` to remove the overhead of
    sorting along the vocab dimension - O(nlog(n)) -> O(n).
    For reference see: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html

    TODO this can be so so much faster with torch on gpu!

    Parameters:
        logprobs: per-token logprobs for a sample / batch
            sample level: shape = [seq_len, vocab_size]
            batch level: shape = [batch_size, seq_len, vocab_size]

    Return:
        top_logprob_indices: indices of the top-k per-token logprobs
            shape = [..., 5] - where we preserve all but the last dimension
    """
    # Multiply by -1 to reverse the order
    logprobs *= -1
    partitioned_logprob_indices = np.argpartition(logprobs, 5, axis=-1)
    top_logprob_indices = partitioned_logprob_indices[..., :5]
    logprobs *= -1

    return top_logprob_indices


def extract_top_logprobs(
        sample_logprobs: np.ndarray, top_indices: np.ndarray, tokenizer: PreTrainedTokenizerFast
) -> List[List[Tuple[str, float]]]:
    """Extract per token top_logprobs for a single sample

    For each token, we extract the top-k predicted tokens
    and corresponding logprobs. Then we convert predicted token_ids
    into strings using the tokenizer.

    Example top_logprobs data format for an example sequence:
    [
        [("the", -0.2), ("a", -0.6), ...],
        [("cat", -0.05), ("dog", -0.1), ...],
        ...
    ]

    Breaking down this format:
        - The sample is represented as a List of Lists per token.
        - For each token, we store a fixed length (k) list of tuples for each of
        the top-k predicted tokens - (token_string, logprob).

    Parameters:
        sample_logprobs: shape = [seq_len, Vocab size]
        top_indices: shape = [seq_len, k]
        tokenizer: model tokenizer, used for decoding

    Return:
        top_logprobs:
            len(top_logprobs) == sample_logprobs.shape[0] == # tokens sample
            len(top_logprobs[i]) == k
    """
    # Extract top_k logprob indices - shape = [seq_len, k]
    sample_top_logprobs = np.take_along_axis(sample_logprobs, top_indices, axis=-1)

    # Generate top_k (string, logprob) token by token
    top_logprobs: List[List[Tuple[str, float]]] = []
    for token_top_ids, token_top_logprobs in zip(top_indices, sample_top_logprobs):
        # List of Tuple[str, int] --> (token string, logprob)
        token_top_logprobs_mapping = []
        # Loop over the top_k predictions for the given token position
        for pred_token_id, logprob in zip(token_top_ids, token_top_logprobs):
            # TODO: See how slow tokenizer decode is and if we just want to index into the vocab directly
            str_token = tokenizer.decode(pred_token_id)
            token_top_logprobs_mapping.append((str_token, logprob))

        top_logprobs.append(token_top_logprobs_mapping)

    return top_logprobs


def process_sample_logprobs(
    sample_logprobs: np.ndarray,
    sample_labels: np.ndarray,
    sample_top_indices: np.ndarray,
    tokenizer: PreTrainedTokenizerFast
) -> Tuple[np.ndarray, List[List[Tuple[str, float]]]]:
    """Extract label_logprobs and top_logprobs

    Whether the labels are GT target labels or generated labels, the
    process is identical. Extract the per token probability assigned to the
    token label and the top-k logprobs.

    Preconditions:
        - We assume that all inputs have been stripped of any padding tokens!

    TODO Add inputs!

    Returns:
        gold_logprobs: GT label logprobs
            gold_logprobs.shape = [seq_len]
        top_logprobs: List of top-k predictions + logprobs
            type = List[Dict[str, float]]
            len(top_logprobs) = seq_len
    """
    # Ensure shape = [len(labels), 1]
    # Allows labels to have shape [labels], [1, labels]
    # TODO ENSURE THAT LABELS IS THE RIGHT SHAPE COMING IN
    assert len(sample_labels.shape) == 1
    sample_labels = sample_labels[..., None]

    # Extract token_logprobs - shape [len(labels)]
    label_logprobs = np.take_along_axis(sample_logprobs, sample_labels, axis=-1).squeeze()

    # Compute top_logprobs
    top_logprobs = extract_top_logprobs(
        sample_logprobs,
        sample_top_indices,
        tokenizer
    )

    return label_logprobs, top_logprobs


def generate_sample_output(
    input_str: str,
    model: PreTrainedModel,
    device: torch.device,
    generation_config: GenerationConfig,
    tokenizer: PreTrainedTokenizerFast
) -> Tuple[np.ndarray, np.array, List[List[Tuple[str, float]]]]:
    # Shape - [1, seq_len]
    # TODO - we can run into trouble if they tokenize in a different way
    input_ids = tokenizer(
        input_str,
        truncation=True,
        return_tensors="pt"
    )["input_ids"].to(device)

    with torch.no_grad():
        gen_ids = model.to(device).generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

    """Does a number of things that we lay out with comments"""
    # 1) Strip the beginning <pad> token and keep as Tensor
    gen_ids = gen_ids[..., 1:]

    # 2) Get the model logits over the generated output
    with torch.no_grad():
        model_outputs = model(input_ids=input_ids, labels=gen_ids)
        logits = model_outputs.logits.cpu().numpy()

    # 3) Compute log probs
    # TODO This is much faster on the gpu
    # First take log_softmax
    logprobs = log_softmax(logits, axis=-1)

    # 4) Do some reshaping and tensor massaging
    logprobs = logprobs.squeeze()
    gen_ids = gen_ids.squeeze().cpu().numpy()

    # 5) Compute the top_logprob_indices
    top_logprobs_indices = get_top_logprob_indices(logprobs)

    # 5) Get the gen_logprobs and top_logprobs
    gen_token_logprobs, gen_top_logprobs = process_sample_logprobs(
        logprobs,
        sample_labels=gen_ids,
        sample_top_indices=top_logprobs_indices,
        tokenizer=tokenizer
    )

    return gen_ids, gen_token_logprobs, gen_top_logprobs


def add_generated_output_to_df(
    df: vaex.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    device: torch.device,
    generation_config: GenerationConfig,
) -> vaex.DataFrame:
    """This is hacky for now, but we will highlight each step of the way

    During generation we want to generate the following columns with
    given row types. We also note down which columns are pa.arrays:
        - generate_output: str
        - generated_token_label_positions: List[List[int]] --> pa
        - generated_token_label_offsets: List[Tuple[int, int]] --> pa
        - generated_token_logprobs: np.array() --> pa
        - generated_top_logprobs: List[List[Tuple[str, int]]] --> pa
    """
    # See above for element types of each list

    @vaex.register_function()
    def generate_batch_outputs(
        texts: pa.array
    ) -> pa.StructArray:
        """Generate over a batch of text input samples

        ...
        """
        generated_outputs = []
        generated_token_label_positions = []
        generated_token_label_offsets = []
        generated_token_logprobs = []
        generated_top_logprobs = []

        for sample in texts:
            # Generate and extract model outputs
            (
                generated_ids,
                sample_generated_token_logprobs,
                sample_generated_top_logprobs
            ) = generate_sample_output(
                    input_str=str(sample),
                    model=model,
                    device=device,
                    generation_config=generation_config,
                    tokenizer=tokenizer
            )

            # Make sure to strip special tokens - e.g. <eos>
            generated_outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
            # Re-tokenize the data to get the token position offsets
            encoded_data = tokenizer(
                [generated_outputs[-1]], return_offsets_mapping=True
            )
            aligned_data = align_tokens_to_character_spans(encoded_data["offset_mapping"])
            # aligned_data assumes batches, so for single samples it returns
            # batched list of length == 1.
            generated_token_label_positions.append(aligned_data.token_label_positions[0])
            generated_token_label_offsets.append(aligned_data.token_label_offsets[0])

            generated_token_logprobs.append(sample_generated_token_logprobs)
            generated_top_logprobs.append(sample_generated_top_logprobs)

        # Ensure correct pyarrow format for top_logprobs
        generated_top_logprobs = pa.array(generated_top_logprobs, type=TOP_LOGPROBS_SCHEMA)
        return pa.StructArray.from_arrays(
            arrays=[generated_outputs, generated_token_label_positions, generated_token_label_offsets, generated_token_logprobs, generated_top_logprobs],
            names=[C.generated_output.value, C.generated_token_label_positions.value, C.generated_token_label_offsets.value, C.generated_token_logprobs.value, C.generated_top_logprobs.value]
        )

    df["Combined_Output"] = df.func.generate_batch_outputs(df[IC.text])

    # extract out each independent column
    df = df.struct.flatten("Combined_Output")
    # struct.flatten creates a column per key (created above),
    # where the column name will be `Combined_Output_<key>` so we rename them
    for col in [C.generated_output.value, C.generated_token_label_positions.value, C.generated_token_label_offsets.value, C.generated_token_logprobs.value, C.generated_top_logprobs.value]:
        df.rename(f"Combined_Output_{col}", col)


    # Check that this works
    df = df.drop("____Combined_Output")
    return df


# def add_generated_output_to_df(
#     df: vaex.DataFrame,
#     tokenizer: PreTrainedTokenizerFast,
#     model: PreTrainedModel,
#     device: torch.device,
#     generation_config: GenerationConfig,
# ) -> vaex.DataFrame:
#     """This is hacky for now, but we will highlight each step of the way
#
#     During generation we want to generate the following columns with
#     given row types. We also note down which columns are pa.arrays:
#         - generate_output: str
#         - generated_token_label_positions: List[List[int]] --> pa
#         - generated_token_label_offsets: List[Tuple[int, int]] --> pa
#         - generated_token_logprobs: np.array() --> pa
#         - generated_top_logprobs: List[List[Tuple[str, int]]] --> pa
#     """
#     # See above for element types of each list
#     generated_outputs = []
#     generated_token_label_positions = []
#     generated_token_label_offsets = []
#     generated_token_logprobs = []
#     generated_top_logprobs = []
#
#     # For generation we iterate sample by sample over df['input']
#     print("here")
#     inputs = df[IC.text.value].values
#     for sample in inputs:
#         # Generate and extract model outputs
#         (
#             generated_ids,
#             sample_generated_token_logprobs,
#             sample_generated_top_logprobs
#         ) = generate_sample_output(
#                 input_str=str(sample),
#                 model=model,
#                 device=device,
#                 generation_config=generation_config,
#                 tokenizer=tokenizer
#         )
#
#         # Make sure to strip special tokens - e.g. <eos>
#         generated_outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
#         # Re-tokenize the data to get the token position offsets
#         encoded_data = tokenizer(
#             [generated_outputs[-1]], return_offsets_mapping=True
#         )
#         aligned_data = align_tokens_to_character_spans(encoded_data["offset_mapping"])
#         # aligned_data assumes batches, so for single samples it returns
#         # batched list of length == 1.
#         generated_token_label_positions.append(aligned_data.token_label_positions[0])
#         generated_token_label_offsets.append(aligned_data.token_label_offsets[0])
#
#         generated_token_logprobs.append(sample_generated_token_logprobs)
#         generated_top_logprobs.append(sample_generated_top_logprobs)
#
#     # Assign the vaex columns manually for now!
#     df[C.generated_output.value] = np.array(generated_outputs)
#     df[C.generated_token_label_positions.value] = pa.array(generated_token_label_positions)
#     df[C.generated_token_label_offsets.value] = pa.array(generated_token_label_offsets)
#     df[C.generated_token_logprobs.value] = pa.array(generated_token_logprobs)
#     # Use schema for top_logprobs
#     top_logprobs_schema = pa.list_(pa.map_(pa.string(), pa.float32()))
#     df[C.generated_top_logprobs.value] = pa.array(generated_top_logprobs, type=top_logprobs_schema)
#
#     return df

# def add_generated_output_to_df(
#     df: vaex.DataFrame,
#     tokenizer: PreTrainedTokenizerFast,
#     model: PreTrainedModel,
#     device: torch.device,
#     generation_config: GenerationConfig,
# ) -> vaex.DataFrame:
#     @vaex.register_function()
#     def generate_output(text: pa.array) -> np.ndarray:
#         results = []
#         for sample in text:
#             # Shape - [1, seq_len]
#             input_ids = tokenizer(str(sample), return_tensors="pt")["input_ids"].to(
#                 device
#             )
#             generation_respone = model.to(device).generate(
#                 input_ids=input_ids,
#                 generation_config=generation_config,
#                 return_dict_in_generate=True,  # Required to get token probs
#                 output_scores=True,  # Required to get token probs
#             )
#
#             # Remove the <pad> token to seed generation
#             generated_tokens = generation_respone.sequences[0, 1:]
#             # TODO: Skip logits for now, will update later
#             # generated_logits = torch.stack(generation_respone.scores)[:, 0, :]
#
#             # Note that the model may also end with the <eos> token. We should
#             # check for this to get the correct logits!
#             if generated_tokens[-1] == tokenizer.eos_token_id:
#                 # Cut off the generated <eos> token if need
#                 # And remove the probability associated with its generation
#                 generated_tokens = generated_tokens[:-1]
#                 # generated_logits = generation_respone.scores[:-1]
#
#             results.append(tokenizer.decode(generated_tokens))
#         return np.array(results)#, np.array(token_logprobs), np.array(top_logprobs)
#
#     df[C.generated_output.value] = df.func.generate_output(df[C.text.value])
#     return df


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
    samples_offsets: List[List[Tuple[int, int]]]
) -> AlignedTokenData:
    """Iterates through each samples offsets and creates character-aligned spans"""
    all_offsets = []
    all_token_positions = []
    for offset_mapping in tqdm(
        samples_offsets, leave=False, desc="Aligning characters with tokens"
    ):
        offsets, token_positions = rollup_offset_mapping(offset_mapping)
        all_offsets.append(offsets)
        all_token_positions.append(token_positions)
    return AlignedTokenData(
        token_label_offsets=all_offsets, token_label_positions=all_token_positions
    )


def get_token_dep_from_labels(
    probs: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts DEP per token prediction using the labels as indexing tools

    First, extract the probabilities of the GT token label

    Probs is a numpy array of shape [batch_size, max_token_len, vocab_size] where
    for each sample (text input) in the batch, every token of that sample has a
    probability vector of size vocab_size (which can be 30k+).

    Labels is of shape [batch_size, max_token_length], where for each sample, it
    indicates the index into the vocab that the token should be (the token label).

    We use advanced indexing to extract out only the probabilities for the token
    label for each sample, for each batch.

    Then, we get the second highest probabilities per token via similar indexing.

    Finally, compute dep and return.

    Returns: (token_dep, gold_probs)

    NOTE: This function is not actively being used as we don't require the user
    to pass in labels. However, if we want to support that flow (which would make
    processing faster and more memory efficient), we can leverage these here.
    """
    batch_size, max_sequence_length, vocab_size = probs.shape
    clean_labels = labels.copy()
    # The labels are set to -100 for ignored tokens. Since the shape is of
    # `max_token_length`, many tokens in a particular sample may be ignored if they
    # don't exist. Similarly, in the case of a decoder-only model, the inputs will
    # be a part of the sample, so the labels are set to -100 so they are ignored
    clean_labels[clean_labels == -100] = 0

    # Create an array of indices for advanced indexing
    batch_indices = np.arange(batch_size)[:, np.newaxis]
    sequence_indices = np.arange(max_sequence_length)[np.newaxis, :]

    # Use advanced indexing to extract the logits for the label tokens
    gold_probs = probs[batch_indices, sequence_indices, clean_labels]

    # Now we set the location of the gold_probs to 0 so we can easily get the
    # second highest, _non_gold_ probs
    probs_no_gold = probs.copy()
    probs_no_gold[batch_indices, sequence_indices, labels] = 0
    # The probability of the second highest for each token in the sample
    second_probs = probs_no_gold.max(axis=-1)
    token_dep = (1 - (gold_probs - second_probs)) / 2
    return token_dep, gold_probs


def unpad_dep_probs_from_labels(
    token_dep: np.ndarray, token_gold_probs: np.ndarray, labels: np.ndarray
) -> Tuple[pa.array, pa.array]:
    """Unpads the incoming numpy array by looking for padded/ignored indices

    Ignored/padded indices are indicated by a -100 in the labels array.

    token_dep, token_gold_probs, and labels are of shape
    [batch_size, max_token_length], but for each sample in the batch, the tokens
    for that sample that are ignored are -100 in the labels matrix.
    So we use that to get only the ones we care about.

    We return a pyarrow array because each batch will have a different shape, which
    can't be represented in numpy

    NOTE: This function is not actively being used as we don't require the user
    to pass in labels. However, if we want to support that flow (which would make
    processing faster and more memory efficient), we can leverage these here.
    """
    batch_deps = []
    batch_gold_probs = []
    for batch_token_dep, batch_token_probs, batch_labels in zip(
        token_dep, token_gold_probs, labels
    ):
        batch_deps.append(batch_token_dep[batch_labels != -100])
        batch_gold_probs.append(batch_token_probs[batch_labels != -100])

    dep = pa.array(batch_deps)
    gold_probs = pa.array(batch_gold_probs)
    return dep, gold_probs
