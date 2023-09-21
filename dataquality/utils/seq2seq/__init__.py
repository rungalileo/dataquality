import numpy as np


def remove_padding(
    labels: np.ndarray, padding_side: str, padded_token_seq: np.ndarray
) -> np.ndarray:
    """Remove padding tokens from a single token sequence

    To remove padding tokens we use the tokenized labels and slice
    tokens depending on the padding side of the tokenizer.

    We assume padded_token_seq is a sequence tokens with shape
    [max_seq_len, ...], where  len(labels) = num_tokens <= max_seq_len
    and `...` indicates 0+ extra dimensions.

    Parameters:
    -----------
    labels: np.ndarray of shape - [num_tokens]
        Token label ids for the sample. Used to get length of
        non-padding logits.
    padding_side: str
        Comes from the tokenizer used for the model, determines
        which side padding is applied.
    padded_token_seq: np.ndarray of shape - [max_seq_len, ...]
        Padded token sequence. The first dimension must be the token
        dimension and be >= num_tokens. The following dimensions are
        unrestricted.

    Returns:
    -------
    non_padded_token_seq: np.ndarray of shape - [num_tokens, ...]
        Sequence with padded tokens removed, leaving other dimensions
        un-altered.
    """
    # Remove padding based on the padding_side of the tokenizer
    num_tokens = len(labels)
    if padding_side == "right":
        return padded_token_seq[:num_tokens]

    return padded_token_seq[-num_tokens:]
