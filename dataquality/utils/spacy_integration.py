from typing import Any, List

import numpy as np
from scipy.special import softmax
from spacy.tokens import Doc
from spacy.training import offsets_to_biluo_tags

from dataquality.exceptions import GalileoException


def validate_obj(an_object: Any, check_type: Any, has_attr: str) -> None:
    if not isinstance(an_object, check_type):
        raise GalileoException(
            f"Expected a {check_type}. Received {str(type(an_object))}"
        )

    if not hasattr(an_object, has_attr):
        raise GalileoException(f"Your {check_type} must have a {has_attr} attribute")


def _convert_spacy_ner_logits_to_probs(logits: np.ndarray, pred: int) -> np.ndarray:
    """Converts ParserStepModel per token logits to probabilities.

    Not all logits outputted by the spacy model are valid probabilities, for this reason
    spacy will ignore potential actions even if they might've had the largest prob mass.
    To account for this, we first sort the logits for each token and then zero out
    all logits larger than the predicted logit (as these must've been ignored by spacy
    or else they would've become the prediction). Finally we take the softmax to convert
    them to probabilities.

    :param logits: ParserStepModel logits for a single token, minus the -U tag logit
    shape of [num_classes]
    :param pred: the idx of the spacy's valid prediction from the logits
    :return: np array of probabilities. shape of [num_classes]
    """
    assert len(logits.shape) == 1

    # Sort in descending order
    argsorted_sample_logits = np.flip(np.argsort(logits))

    # Get all logit indices where pred_logit > logit
    # These are 'valid' because spacy ignored all logits > pred_logit
    # as it they were determined to not be possible given the current state.
    valid_logit_indices = argsorted_sample_logits[
        np.where(argsorted_sample_logits == pred)[0][0] :
    ]

    valid_probs = softmax(logits[valid_logit_indices])

    # non valid_logit_indices should be set to 0
    probs = np.zeros(logits.shape)
    probs[valid_logit_indices] = valid_probs

    return probs


def _convert_spacy_ents_for_doc_to_predictions(
    docs: List[Doc], labels: List[str]
) -> List[List[int]]:
    """Converts spacy's representation of ner spans to their per token predictions.

    Uses some spacy utility code to convert from start/end/label representation to the
    BILUO per token corresponding tagging scheme.

    """
    prediction_indices = []
    for doc in docs:
        pred_output = offsets_to_biluo_tags(
            doc, [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        )
        pred_output_ind = [labels.index(tok_pred) for tok_pred in pred_output]
        prediction_indices.append(pred_output_ind)
    return prediction_indices
