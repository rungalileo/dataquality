import numpy as np

from dataquality.utils.spacy_integration import convert_spacy_ner_logits_to_valid_logits


def test_convert_spacy_ner_logits_to_valid_logits():
    arr_1 = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    pred_1 = 2

    assert np.allclose(
        convert_spacy_ner_logits_to_valid_logits(arr_1, pred_1),
        [-0.6, -0.6, 0.7, 0.6, 0.5, 0.4],
    )

    arr_2 = np.array([3.4, -0.9, 0.4, 2.8])
    pred_2 = 0

    assert np.allclose(
        convert_spacy_ner_logits_to_valid_logits(arr_2, pred_2), [3.4, -0.9, 0.4, 2.8]
    )

    arr_3 = np.array([3.4, -0.9, 0.4, 2.8])
    pred_3 = 1

    assert np.allclose(
        convert_spacy_ner_logits_to_valid_logits(arr_3, pred_3),
        [-1.9, -0.9, -1.9, -1.9],
    )

    print("pausing")
