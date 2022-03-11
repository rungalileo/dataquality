from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import spacy
import vaex
from spacy.pipeline.ner import EntityRecognizer
from spacy.training import Example

import dataquality
from dataquality.core.integrations.spacy import (
    GalileoEntityRecognizer,
    log_input_examples,
    unwatch,
    watch,
)
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.schemas.task_type import TaskType
from tests.conftest import LOCATION
from tests.utils.spacy_integration import load_ner_data_from_local, train_model


training_data = [
    (
        "what is SEMRUSH PRO? Can you run complex queries ? Can you identify "
        "active usage ?",
        {
            "entities": [
                (21, 32, "Questions About the Product"),
                (51, 67, "Questions About the Product"),
            ]
        },
    ),
    ("Thank you for your subscription renewal", {"entities": [(19, 39, "Renew")]}),
    (
        "you can upgrade your account for an old price,while you can upgrade your "
        "account for $399.95/month",
        {"entities": [(8, 28, "Potential Upsell"), (60, 80, "Potential Upsell")]},
    ),
    (
        "I like EMSI ordered the pro package",
        {"entities": [(12, 23, "Product Usage")]},
    ),
    (
        "Here you go, your account is created",
        {
            "entities": [
                (0, 11, "Action item accomplished"),
                (29, 36, "Action item accomplished"),
            ]
        },
    ),
]

test_data = [
    ("Thank you for your subscription renewal", {"entities": [(32, 39, "Renew")]}),
]


def test_log_input_examples(set_test_config):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)

    watch(nlp)
    log_input_examples(training_examples, "training")

    # assert that we added ids to the examples for later joining with model outputs
    assert all(
        [
            examples.predicted.user_data["id"] == i
            for i, examples in enumerate(training_examples)
        ]
    )

    logged_data = vaex.open(f"{LOCATION}/input_data.arrow")

    assert logged_data["id"].tolist() == [0, 1, 2, 3, 4]
    assert logged_data["split"].tolist() == ["training"] * len(training_examples)
    assert all(
        [
            text == training_data[i][0]
            for i, text in enumerate(logged_data["text"].tolist())
        ]
    )

    # Checks for logged gold spans matching converts from token to char idxs
    logged_token_indices = logged_data["text_token_indices"].tolist()
    for i, (split_plus_id, ents) in enumerate(
        dataquality.get_model_logger().logger_config.gold_spans.items()
    ):
        assert len(logged_token_indices[i]) % 2 == 0
        ents_as_char_idxs = []
        for ent in ents:
            start_char_idx = logged_token_indices[i][2 * ent[0]]
            end_char_idx = logged_token_indices[i][2 * ent[1] - 1]
            ents_as_char_idxs.append((start_char_idx, end_char_idx, ent[2]))

        assert training_data[i][1]["entities"] == ents_as_char_idxs


def test_watch(set_test_config):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)
    watch(nlp)

    assert text_ner_logger_config.user_data["nlp"] == nlp
    assert dataquality.get_data_logger().logger_config.labels == [
        "B-Questions About the Product",
        "B-Potential Upsell",
        "B-Action item accomplished",
        "B-Renew",
        "B-Product Usage",
        "I-Questions About the Product",
        "I-Potential Upsell",
        "I-Action item accomplished",
        "I-Renew",
        "I-Product Usage",
        "L-Questions About the Product",
        "L-Potential Upsell",
        "L-Action item accomplished",
        "L-Renew",
        "L-Product Usage",
        "U-Questions About the Product",
        "U-Potential Upsell",
        "U-Action item accomplished",
        "U-Renew",
        "U-Product Usage",
        "O",
    ]
    assert dataquality.get_data_logger().logger_config.tagging_schema == "BILOU"

    assert isinstance(nlp.get_pipe("ner"), EntityRecognizer)
    assert isinstance(nlp.get_pipe("ner"), GalileoEntityRecognizer)


@pytest.mark.skip(
    reason="Implementation hinges on more info from spacy or a bug fix, " "see unwatch"
)
def test_unwatch(set_test_config):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    original_ner = nlp.add_pipe("ner")

    training_data = [
        (
            "what is SEMRUSH PRO? Can you run complex queries ? Can you identify "
            "active usage ?",
            {
                "entities": [
                    (21, 32, "Questions About the Product"),
                    (51, 67, "Questions About the Product"),
                ]
            },
        ),
        ("Thank you for your subscription renewal", {"entities": [(19, 39, "Renew")]}),
        (
            "you can upgrade your account for an old price,while you can upgrade "
            "your account for $399.95/month",
            {"entities": [(8, 28, "Potential Upsell"), (60, 80, "Potential Upsell")]},
        ),
        (
            "I like EMSI ordered the pro package",
            {"entities": [(12, 23, "Product Usage")]},
        ),
        (
            "Here you go, your account is created",
            {
                "entities": [
                    (0, 11, "Action item accomplished"),
                    (29, 36, "Action item accomplished"),
                ]
            },
        ),
    ]

    training_examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)
    watch(nlp)
    unwatch(nlp)

    assert isinstance(nlp.get_pipe("ner"), EntityRecognizer)
    assert not isinstance(nlp.get_pipe("ner"), GalileoEntityRecognizer)

    assert nlp.get_pipe("ner").moves == original_ner.moves
    assert nlp.get_pipe("ner").moves == original_ner.model


def test_embeddings_get_updated(cleanup_after_use, set_test_config):
    """This test both checks our spacy wrapper end to end and that embs update.

    If embeddings stop updating that means the spacy architecture somehow changed
    and would make our user's embeddings seem meaningless
    """
    set_test_config(task_type=TaskType.text_ner)
    train_model(training_data=training_data, test_data=training_data, num_epochs=2)

    _, embs, _ = load_ner_data_from_local("training")
    embs = embs["emb"].to_numpy()

    dataquality.get_data_logger()._cleanup()

    train_model(training_data=training_data, test_data=training_data, num_epochs=1)

    _, embs_2, _ = load_ner_data_from_local("training")
    embs_2 = embs_2["emb"].to_numpy()

    assert embs.shape == embs_2.shape
    assert not np.allclose(embs, embs_2)


def test_spacy_ner(cleanup_after_use, set_test_config) -> None:
    """An end to end test of functionality"""
    set_test_config(task_type=TaskType.text_ner)
    num_epochs = 2
    training_losses = train_model(training_data, test_data, num_epochs=num_epochs)

    # loss values gotten from running the script with Galileo Logging turned off
    # assert np.allclose(training_losses, [25.50000334, 14.20009732])

    data, embs, probs = load_ner_data_from_local("training", epoch=num_epochs - 1)

    assert len(data) == 5
    assert all(data["id"] == range(len(data)))
    gt_data = pd.DataFrame(
        data={
            "id": range(len(data)),
            "split": ["training"] * len(data),
            "text": [data_sample[0] for data_sample in training_data],
            "text_token_indices": [np.array([ 0,  4,  5,  7,  8, 15, 16, 19, 19, 20, 21, 24, 25, 28, 29, 32, 33,
       40, 41, 48, 49, 50, 51, 54, 55, 58, 59, 67, 68, 74, 75, 80, 81, 82]), np.array([ 0,  5,  6,  9, 10, 13, 14, 18, 19, 31, 32, 39]), np.array([ 0,  3,  4,  7,  8, 15, 16, 20, 21, 28, 29, 32, 33, 35, 36, 39, 40,
       45, 45, 46, 46, 51, 52, 55, 56, 59, 60, 67, 68, 72, 73, 80, 81, 84,
       85, 86, 86, 92, 92, 93, 93, 98]), np.array([ 0,  1,  2,  6,  7, 11, 12, 19, 20, 23, 24, 27, 28, 35]), np.array([ 0,  4,  5,  8,  9, 11, 11, 12, 13, 17, 18, 25, 26, 28, 29, 36])],
            "data_schema_version": [1] * len(data),
        }
    )
    assert data.equals(gt_data), f"Received the following data df {data}"

    # TODO: Need a less flakey test here
    assert embs["id"].tolist() == list(range(len(embs)))
    embs = embs["emb"].to_numpy().astype(np.float16)
    gt_embs = np.array([[ 5.1172e+00,  2.3711e+00,  1.3086e+00,  9.2969e-01,  5.8691e-01,
        -4.8779e-01,  1.2783e+00,  1.0127e+00,  3.4805e+00,  1.6016e+00,
         2.2637e+00,  1.5723e+00,  2.4414e-01, -2.5859e+00,  2.1055e+00,
         5.4023e+00, -1.2422e+00, -1.5547e+00,  3.2949e+00,  8.2324e-01,
        -3.1270e+00,  1.0459e+00,  3.3945e+00, -9.0820e-02,  2.2344e+00,
         3.0977e+00,  3.6934e+00,  1.8291e+00, -2.5977e-01,  2.8672e+00,
         5.7373e-01, -3.1519e-01, -5.5127e-01,  5.8594e-01,  1.4707e+00,
        -2.6001e-01, -3.5596e-01,  3.6172e+00,  9.2432e-01,  1.3877e+00,
         3.5156e+00,  1.8525e+00,  2.3867e+00,  2.6289e+00,  3.7617e+00,
         2.4199e+00,  2.0645e+00,  3.5527e+00,  2.8027e+00,  3.6406e+00,
        -4.0015e-01,  6.2646e-01,  8.3496e-01,  1.7227e+00,  4.6055e+00,
         1.0605e+00,  5.2031e+00,  1.7881e+00,  7.9590e-01, -2.1033e-01,
        -6.2598e-01,  1.3135e+00,  2.0586e+00, -5.6055e-01],
       [ 9.7705e-01,  3.4941e+00,  9.8145e-01,  7.1484e-01, -3.0029e-02,
         3.1230e+00,  1.0254e+00,  6.5723e-01,  1.6758e+00, -7.9150e-01,
         6.1572e-01,  2.6641e+00,  3.4619e-01, -1.3193e+00,  5.7129e-02,
         5.0391e-01, -1.2734e+00, -1.2581e-02,  1.5703e+00,  9.5752e-01,
        -8.1885e-01, -2.9248e-01,  2.5273e+00,  6.5332e-01,  1.0049e+00,
         1.0176e+00,  1.6182e+00,  2.2383e+00, -5.9912e-01,  4.4609e+00,
         2.7383e+00,  9.4775e-01, -2.0645e+00,  2.7051e+00,  1.6885e+00,
         1.7676e+00,  1.1699e+00,  1.1592e+00,  7.8223e-01,  2.4492e+00,
        -3.7524e-01,  2.7324e+00,  2.0488e+00,  2.0469e+00,  1.8787e-01,
         8.8281e-01, -1.8823e-01,  5.3672e+00,  2.4160e+00, -1.7566e-01,
         2.3657e-01,  3.9038e-01,  2.8008e+00, -1.3506e+00,  2.5391e+00,
         6.9336e-01,  2.3767e-01,  4.0039e+00, -3.0737e-01, -6.8066e-01,
        -1.1738e+00,  8.0371e-01,  1.9463e+00,  3.9941e-01],
       [ 3.5117e+00,  2.9004e+00,  0.0000e+00,  8.6963e-01,  1.4893e+00,
         2.1738e+00, -1.5469e+00,  9.6875e-01,  0.0000e+00,  7.3340e-01,
        -1.5293e+00,  2.8301e+00, -6.3770e-01,  0.0000e+00,  3.2031e+00,
         0.0000e+00, -1.7520e+00,  0.0000e+00,  3.9805e+00,  3.5859e+00,
        -2.0935e-01,  2.0898e+00,  4.8853e-01,  4.4570e+00,  1.2021e+00,
         1.5215e+00,  3.9609e+00,  3.1323e-01, -3.5898e+00,  5.1445e+00,
         3.1562e+00, -4.5859e+00, -2.1309e+00,  2.8301e+00, -1.5439e+00,
         0.0000e+00,  1.7354e+00,  3.3936e-01, -6.6113e-01,  1.4150e+00,
        -2.7949e+00,  4.7227e+00,  1.1826e+00, -1.0518e+00, -7.0850e-01,
         8.6816e-01, -1.3350e+00,  4.6797e+00,  3.2285e+00,  1.8457e+00,
         1.5186e+00,  2.8691e+00,  2.8652e+00,  0.0000e+00,  3.5312e+00,
         0.0000e+00, -2.5293e-01, -1.4023e+00,  2.6777e+00,  2.4570e+00,
         1.7314e+00,  0.0000e+00,  2.8301e+00, -4.5142e-01],
       [ 1.7881e+00,  2.9082e+00,  1.1357e+00,  3.4453e+00,  2.5449e+00,
         2.5605e+00,  1.0840e-01,  1.0332e+00,  5.9180e-01,  1.1562e+00,
         1.2793e+00,  1.4565e-02,  1.0625e+00,  1.9912e+00, -2.6562e-01,
         1.0039e+00, -2.5293e+00, -1.6670e+00,  2.4238e+00,  1.4521e+00,
        -7.5098e-01,  1.4775e+00,  2.5488e+00,  2.2422e+00,  1.1201e+00,
         1.6670e+00,  3.0215e+00,  9.3994e-01, -9.0234e-01,  5.8008e+00,
         2.1602e+00,  8.4814e-01, -7.0752e-01,  3.1714e-01,  2.0781e+00,
         2.4023e+00,  1.0361e+00,  5.6396e-01,  1.6182e+00,  1.2051e+00,
        -8.0176e-01,  5.1172e+00,  3.0371e+00,  2.0371e+00, -4.1235e-01,
         1.6035e+00,  5.1445e+00,  2.5488e+00,  2.5098e+00,  1.5586e+00,
        -1.6510e-02,  9.1113e-01,  2.6660e+00,  1.8965e+00,  1.1904e+00,
         2.0039e+00,  1.1650e+00,  2.5684e+00, -5.6061e-02,  1.0674e+00,
        -2.0898e+00,  4.0508e+00,  1.5547e+00,  3.1519e-01],
       [ 2.2637e+00,  2.7773e+00,  1.9443e+00,  2.0371e+00,  9.2957e-02,
         3.1875e+00, -4.7241e-01, -7.7246e-01,  5.1484e+00, -1.3350e+00,
        -3.4448e-01, -3.2861e-01,  3.7910e+00,  1.5176e+00,  7.0264e-01,
         1.6284e-01, -1.0791e+00,  3.4106e-01,  1.0195e+00,  1.4941e+00,
        -5.3613e-01, -6.0254e-01,  4.5117e+00,  1.2588e+00,  3.0371e+00,
        -4.6802e-01,  1.4961e+00,  2.1543e+00, -1.2773e+00,  6.8164e+00,
         1.4385e+00,  3.0020e+00,  5.9229e-01,  1.3438e+00,  2.1699e+00,
         1.9639e+00, -1.0020e+00,  1.9375e+00,  5.6299e-01,  1.9717e+00,
         2.1426e+00,  3.4473e+00,  2.7852e+00,  4.6445e+00, -3.2031e+00,
         1.0437e-01,  4.4297e+00,  6.3750e+00,  6.4062e-01,  1.4648e+00,
        -2.7695e+00,  9.1064e-01,  3.1699e+00,  1.0605e+00,  2.2129e+00,
         4.2969e+00,  4.7900e-01,  4.7632e-01, -1.9604e-01,  6.1426e-01,
        -1.3506e+00, -2.6245e-02,  2.5273e+00, -1.1121e-01],
       [ 1.4404e+00,  1.6855e+00,  1.8721e+00,  2.4597e-01,  1.4893e+00,
         2.2148e+00,  7.9102e-01,  3.5371e+00,  5.3906e-01, -1.0898e+00,
         1.3193e+00,  2.0752e-01,  5.4297e-01,  1.5264e+00,  1.9346e+00,
        -1.2683e-01, -2.6699e+00,  2.7773e+00,  2.3145e+00,  4.0503e-01,
        -4.4653e-01,  2.3499e-01,  4.8789e+00,  1.6777e+00,  5.6592e-01,
         1.6055e+00,  1.2041e+00,  3.2051e+00, -1.2366e-01,  4.2305e+00,
         2.1465e+00,  2.8262e+00,  1.2231e-01,  1.0908e+00,  1.8672e+00,
         3.7549e-01,  1.4248e+00,  4.2148e+00,  4.9829e-01,  0.0000e+00,
        -1.0684e+00,  2.1992e+00,  1.7656e+00,  1.7207e+00,  2.7441e+00,
        -2.0469e+00, -1.3945e+00,  7.1133e+00,  4.8398e+00,  1.0215e+00,
        -6.3428e-01,  1.8027e+00,  6.9678e-01, -1.0771e+00, -9.9414e-01,
         3.6426e-01, -1.5850e+00,  1.2568e+00, -3.1421e-01,  1.2529e+00,
         5.3940e-03,  1.9365e+00,  3.3652e+00, -4.8169e-01],
       [ 3.1680e+00,  4.2852e+00,  1.6895e+00,  1.7676e+00,  1.0830e+00,
         2.3926e+00,  1.6885e+00,  4.1675e-01,  4.0000e+00, -4.8169e-01,
         6.9971e-01,  8.1445e-01,  2.8594e+00,  6.6211e-01,  3.0684e+00,
         2.2422e+00, -1.5557e+00, -1.0059e+00,  1.6533e+00, -1.2469e-01,
        -2.3022e-01,  1.2275e+00,  2.5762e+00, -8.0383e-02,  2.7832e+00,
         1.5020e+00,  2.2305e+00,  2.7852e+00, -2.6688e-02,  4.3164e+00,
         1.9316e+00,  2.6440e-01, -5.9131e-01,  3.8892e-01,  1.8955e+00,
         1.9883e+00,  1.5508e+00,  1.7051e+00, -2.7026e-01,  2.3477e+00,
        -2.1152e+00,  3.5898e+00,  1.5449e+00,  1.3525e+00,  5.7080e-01,
         8.0713e-01,  1.7363e+00,  3.9902e+00,  1.7031e+00,  2.8760e-01,
         1.2080e+00,  2.1680e+00,  1.8594e+00, -2.8906e-01,  2.0469e+00,
         2.6367e+00,  1.3350e+00,  1.9473e+00,  4.5215e-01,  2.1816e+00,
        -2.0059e+00,  1.2627e+00,  2.9102e+00,  3.2251e-01],
       [ 3.0332e+00,  2.0430e+00,  1.1201e+00,  1.2646e+00,  3.6836e+00,
        -2.3340e-01,  7.4170e-01,  1.8887e+00,  1.2061e+00, -1.2959e+00,
         1.7139e+00,  4.1602e-01,  1.3730e+00,  2.1992e+00,  1.0469e+00,
         1.5957e+00, -3.8672e+00, -3.7354e-01,  2.3340e+00,  1.0830e+00,
         1.8525e+00,  1.4004e+00,  3.6211e+00,  1.7471e+00,  5.3271e-01,
         7.2119e-01,  1.6709e+00,  1.1143e+00, -5.7080e-01,  5.0938e+00,
         1.0254e+00,  2.1113e+00, -7.4219e-01,  1.0215e+00,  5.1953e+00,
         1.3525e+00,  1.6240e+00,  1.3232e+00,  9.9902e-01,  3.4082e+00,
        -1.6953e+00,  4.2695e+00,  2.9883e+00,  2.2852e+00,  1.5322e+00,
         6.4209e-01,  3.3965e+00,  3.8574e+00,  4.1992e+00,  3.4155e-01,
         9.3359e-01, -2.3474e-01,  1.8047e+00,  7.7783e-01,  1.1006e+00,
         5.4297e-01,  1.3513e-01,  1.0527e+00,  3.3325e-01,  1.1934e+00,
        -1.8145e+00, -3.1372e-02,  2.3945e+00, -8.0518e-01]], dtype=np.float16)
    assert all([span_emb in gt_embs for span_emb in embs])

    # arrange the probs array to account for misordering of logged samples
    probs = probs.sort_values(by=["sample_id","span_start"]).drop("id", axis=1).round(4)
    assert len(probs) == 8
    gt_probs = pd.DataFrame(
        data={
            "sample_id": [0, 0, 1, 2, 2, 3, 4, 4],
            "split": ["training"] * len(probs),
            "epoch": [num_epochs - 1] * len(probs),
            "is_gold": [True] * len(probs),
            "is_pred": [False] * len(probs), # why are there no preds?
            "span_start": [5, 11, 4, 2, 13, 3, 0, 7],
            "span_end": [8, 14, 6, 5, 16, 5, 3, 8],
            "gold": ['Questions About the Product', 'Questions About the Product', 'Renew', 'Potential Upsell', 'Potential Upsell', 'Product Usage', 'Action item accomplished', 'Action item accomplished'],
            "pred": [""] * len(probs),
            "data_error_potential": [0.509, 0.5088, 0.5083, 0.508, 0.5085, 0.5125, 0.5119, 0.5092],
            "galileo_error_type": ["missed_label"] * len(probs),
        }
    )
    assert probs.equals(gt_probs), f"Received the following probs df {probs}"


@pytest.mark.skip(reason="SpacyPatchState no longer exists")
def test_galileo_transition_based_parser_forward(set_test_config):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")

    text_samples = ["Some text", "Some other text", "Some more text"]

    # mock_transition_based_parser_model = Mock()
    docs = []
    for i, text in enumerate(text_samples):
        doc = nlp(text)
        doc.user_data["id"] = i
        docs.append(doc)

    fake_embeddings = np.random.rand(sum([len(doc) for doc in docs]), 64)

    def mock_transition_based_parser_forward(model, X, is_train):
        mock_parser_step_model = Mock()
        mock_parser_step_model._func = lambda x: print("parser_step_model forward fn")
        mock_parser_step_model.tokvecs = fake_embeddings
        return mock_parser_step_model, lambda x: print(
            "transition_based_parser backprop fn"
        )

    # TODO: Need to replace this
    # SpacyPatchState.orig_transition_based_parser_forward = (
    #     mock_transition_based_parser_forward
    # )

    dataquality.set_epoch(0)
    dataquality.set_split("training")

    # TODO: Need to replace with a different call for this test to work
    # galileo_transition_based_parser_forward(
    #     mock_transition_based_parser_model, docs, is_train=True
    # )

    # TODO: Need to fix this as well
    # assert SpacyPatchState.model_logger.ids == [0, 1, 2]
    # assert SpacyPatchState.model_logger.epoch == 0
    # assert SpacyPatchState.model_logger.split == "training"
    # assert SpacyPatchState.model_logger.probs == [[], [], []]
    # assert len(SpacyPatchState.model_logger.emb) == 3
    # assert all(
    #     [
    #         embedding.shape == (len(docs[i]), 64)
    #         for i, embedding in enumerate(SpacyPatchState.model_logger.emb)
    #     ]
    # )

    assert text_ner_logger_config.user_data["_spacy_state_for_pred"] == [
        None,
        None,
        None,
    ]


@pytest.mark.skip(reason="Still need to implement the mock ParserStepModel")
def test_galileo_parser_step_forward():
    pass


mocks_training_data = [
    (
        "what is SEMRUSH PRO? Can you run complex queries ? Can you identify "
        "active usage ?",
        {
            "entities": [
                (21, 32, "Questions About the Product"),
                (51, 67, "Questions About the Product"),
            ]
        },
    ),
    ("Thank you for your subscription renewal", {"entities": [(19, 39, "Renew")]}),
    (
        "you can upgrade your account for an old price,while you can upgrade your "
        "account for $399.95/month",
        {"entities": [(8, 28, "Potential Upsell"), (60, 80, "Potential Upsell")]},
    ),
]
