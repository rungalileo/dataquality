from typing import Optional, Tuple

import numpy as np
import torch
from torch.nn import NLLLoss


def compute_confidence(probs: np.ndarray) -> np.ndarray:
    """Compute the confidence(s)

    probs - [n_tokens, C]
    gold_labels - [n_tokens]

    Return:
    -------
        - confidences [n_tokens]: confidence per token
    """
    return np.max(probs, axis=-1)


# TODO: is this string or int?
def compute_loss(probs: np.ndarray, gold_labels: np.ndarray) -> np.ndarray:
    """Compute the NLLLoss for each token in probs

    probs - [n_tokens, C]
    gold_labels - [n_tokens]

    Assumes for now that probs is a matrix of probability vectors per token,
    NOT the logits. Thus we have to we have to take the log since NLLLoss
    expects log probs.

    For the NLLLoss we set reduction = "None" because we want
    a loss per token

    Return:
    -------
        - loss [n_tokens]: The loss per NER token

    !NOTE!: if we pass in logits we can use the more numerically stable
    CrossEntropyLoss and avoid taking the log
    """
    loss = NLLLoss(reduction="none")
    log_probs = np.log(probs)
    temp = loss(torch.tensor(log_probs), torch.tensor(gold_labels)).numpy()
    import pdb; pdb.set_trace()
    return temp


def select_span_token_for_prob(
    probs: np.ndarray, method: str, gold_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[float]]:
    """Select the representative token for a span's prob vector

    Based on the method provided, compute that metric for
    each of the NER tokens and then select the "representative"
    token.

    Return:
    -------
        - prob_token: Probability vector for selected token - shape[n_classes]
        - gold_label: The gold label for the selected token
    """
    import pdb; pdb.set_trace()
    gold_label = None
    if method == "confidence":
        confidences = compute_confidence(probs)
        selected = np.argmin(confidences)
    elif method == "loss":
        assert gold_labels, "You must include gold_labels to select span probs for prob_loss."  # Required for linting
        losses = compute_loss(probs, gold_labels)
        selected = np.argmax(losses)
        gold_label = gold_labels[selected]
    else:
        raise ValueError(f"Cannot select span token for method {method}")

    # Return the gold_label as well as a single element in an array
    return probs[selected, :], gold_label
