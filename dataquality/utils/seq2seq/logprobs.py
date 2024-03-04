from typing import List, Tuple

import numpy as np
from transformers import PreTrainedTokenizerFast

from dataquality.exceptions import GalileoException
from dataquality.schemas.seq2seq import TOP_K, LogprobData


def get_top_logprob_indices(logprobs: np.ndarray) -> np.ndarray:
    """Extract per-token top-k logprobs

    logprobs can either be at the sample level or batch level.

    In both situations, we compute the top logprobs along the final (-1)
    vocab dimension. We use `np.argpartition` to remove the overhead of
    sorting along the vocab dimension - O(nlog(n)) -> O(n).
    For reference see: https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html

    Post-conditions:
    ----------------
        - logprobs is left unchanged

    TODO this can be so so much faster with torch on gpu!

    Parameters:
    -----------
    logprobs: np.ndarray of shape [(optional)batch_size, seq_len, vocab_size]
        per-token logprobs for a sample or a batch

    Return:
    -------
    top_logprob_indices: np.ndarray of shape - [..., TOP_K]
        Indices of the top-k per-token logprobs. Note we preserve
        all but the last dimension (i.e. not -1) to seemlesly handle
        samples or batches.
    """
    # Multiply by -1 to reverse the order
    logprobs *= -1
    partitioned_logprob_indices = np.argpartition(logprobs, TOP_K, axis=-1)
    top_logprob_indices = partitioned_logprob_indices[..., :TOP_K]
    logprobs *= -1

    return top_logprob_indices


def extract_top_logprobs(
    sample_logprobs: np.ndarray,
    top_indices: np.ndarray,
    tokenizer: PreTrainedTokenizerFast,
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
    -----------
    sample_logprobs: np.ndarray of shape - [seq_len, Vocab size]
    top_indices: np.ndarray of shape - [seq_len, k]
    tokenizer: PreTrainedTokenizerFast

    Return:
    -------
    top_logprobs: List[List[Tuple[str, float]]]
        len(top_logprobs) == sample_logprobs.shape[0] == num_tokens
        len(top_logprobs[i]) == TOP_K
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
            str_token = tokenizer.decode(pred_token_id)
            token_top_logprobs_mapping.append((str_token, logprob))

        top_logprobs.append(token_top_logprobs_mapping)

    return top_logprobs


def process_sample_logprobs(
    sample_logprobs: np.ndarray,
    sample_labels: np.ndarray,
    sample_top_indices: np.ndarray,
    tokenizer: PreTrainedTokenizerFast,
) -> LogprobData:
    """Extract label_logprobs and top_logprobs

    Whether the labels are GT target labels or generated labels, the
    process is identical. Extract the per token probability assigned to the
    token label and the top-k logprobs.

    Preconditions:
        - We assume that all inputs have been stripped of any padding tokens!

    Parameters:
    -----------
    sample_logprobs: np.ndarray of shape - [seq_len, vocab_size]
        Per-token logprobs for a single sample
    sample_labels: np.ndarray of shape - [seq_len]
        Per-token lables for the sample. As a pre-condition
        we assume that this is a 1D tensor with length seq_len.
        This is important for extracting logprobs
    sample_top_indices: np.ndarray of shape - [seq_len, TOP_K]
        Top_K logprob indices for each token. Note that these
        are not in order.
    tokenizer: PreTrainedTokenizerFast
        Tokenizer used by the model

    Returns:
    --------
    logprob_data: LogprobData
        token_logprobs and top_logprobs for the sample
    """
    # Ensure final shape - [len(labels), 1]
    if sample_labels.ndim != 1:
        raise GalileoException(
            f"Invalid shape {sample_labels.shape}, process_sample_logprobs"
            f" expects sample_labels to be a 1D array"
        )
    sample_labels = sample_labels[..., None]

    # Extract token_logprobs return shape [len(labels), 1]
    # Squeeze final dimension to get - shape [len(labels)]
    token_logprobs = np.take_along_axis(
        sample_logprobs, sample_labels, axis=-1
    ).squeeze(-1)

    # Compute top_logprobs
    top_logprobs = extract_top_logprobs(sample_logprobs, sample_top_indices, tokenizer)
    return LogprobData(token_logprobs, top_logprobs)
