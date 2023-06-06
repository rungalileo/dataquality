from typing import Optional, Tuple

import numpy as np

from dataquality.schemas.ner import NERProbMethod


def compute_confidence(probs: np.ndarray) -> np.ndarray:
    """Compute the confidences for a prob array

    Where confidence is the max prob for a given token

    probs - [n_tokens, n_classes]

    Return:
    -------
      - confidences [n_tokens]: confidence per token
    """
    return np.max(probs, axis=-1)


def compute_nll_loss(probs: np.ndarray, gold_labels: np.ndarray) -> np.ndarray:
    """Compute the NLLLoss for each token in probs

    Assumes for now that probs is a matrix of probability vectors per token,
    NOT the logits. Thus we have to take the log since Negative Log-Likelihood
    Loss expects log probs.

    probs - [n_tokens, n_classes]
    gold_labels - [n_tokens], each element is index of gold label for that token

    Return:
    -------
        - loss [n_tokens]: The loss per NER token
    """
    log_probs = np.log(probs)
    loss = -log_probs[np.arange(log_probs.shape[0]), gold_labels]
    return loss


def select_span_token_for_prob(
    probs: np.ndarray, method: NERProbMethod, gold_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[int]]:
    """Select the representative token for a span's prob vector

    Based on the method provided, compute that metric for
    each of the NER tokens and then select the "representative"
    token.

    probs - [n_tokens, n_classes]
    method - NERProbMethod (confidence or loss)
    gold_labels - [n_tokens], each element is index of gold label for that token

    Return:
    -------
        - prob_token: Probability vector for selected token - shape[n_classes]
        - gold_label: The gold label index for the selected token (if method is loss)
    """
    gold_label = None
    if method == NERProbMethod.confidence:
        confidences = compute_confidence(probs)
        selected = np.argmin(confidences)
    elif method == NERProbMethod.loss:
        assert (  # Required for linting
            gold_labels is not None and gold_labels.size > 0
        ), "You must include gold_labels to select span probs for loss_prob."
        losses = compute_nll_loss(probs, gold_labels)
        selected = np.argmax(losses)
        gold_label = gold_labels[selected]
    else:
        raise ValueError(f"Cannot select span token for method {method}")

    return probs[selected, :], gold_label
