import numpy as np


def remove_padding(
    padded_token_seq: np.ndarray,
    num_tokens: int,
    padding_side: str,
) -> np.ndarray:
    """Remove padding tokens from a single token sequence

    To remove padding tokens we use the tokenized labels and slice
    tokens depending on the padding side of the tokenizer.

    We assume padded_token_seq is a sequence tokens with shape
    [max_seq_len, ...], where  len(labels) = num_tokens <= max_seq_len
    and `...` indicates 0+ extra dimensions.

    Parameters:
    -----------
    padded_token_seq: np.ndarray of shape - [max_seq_len, ...]
        Padded token sequence. The first dimension must be the token
        dimension and be >= num_tokens. The following dimensions are
        unrestricted.
    num_tokens: int
        Length of the non-padded logits.
    padding_side: str
        Comes from the tokenizer used for the model, determines
        which side padding is applied.

    Returns:
    -------
    non_padded_token_seq: np.ndarray of shape - [num_tokens, ...]
        Sequence with padded tokens removed, leaving other dimensions
        un-altered.
    """
    # Remove padding based on the padding_side of the tokenizer
    if padding_side == "right":
        return padded_token_seq[:num_tokens]

    return padded_token_seq[-num_tokens:]
