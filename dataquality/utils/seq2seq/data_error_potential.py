from typing import Tuple

import numpy as np
import pyarrow as pa


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
