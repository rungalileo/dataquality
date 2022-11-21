from itertools import chain
from typing import Callable, Generator, List
from unittest import mock

import datasets
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import vaex

import dataquality
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.text_ner import TextNERDataLogger
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import TEST_PATH
from tests.test_utils.ner_constants import (
    GOLD_SPANS,
    LABELS,
    NER_INPUT_DATA,
    NER_INPUT_ITER,
    NER_INPUT_TUPLES,
    TEXT_INPUTS,
    TEXT_TOKENS,
)

model_logger = TextNERModelLogger()


def test_gold_span_extraction() -> None:
    logger = TextNERDataLogger()
    gold_spans = [
        {"start": 0, "end": 4, "label": "YEAR"},
        {"start": 17, "end": 29, "label": "ACTOR"},
    ]
    token_indices = [(0, 4), (5, 11), (12, 16), (17, 22), (17, 22), (23, 29), (23, 29)]
    new_spans = logger._extract_gold_spans(gold_spans, token_indices)
    good_new_spans = [
        {"start": 0, "end": 1, "label": "YEAR"},
        {"start": 3, "end": 7, "label": "ACTOR"},
    ]
    assert new_spans == good_new_spans


def test_construct_gold_sequence() -> None:
    len_sequence = 15
    case_1_seq_bio = [
        "O",
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "O",
        "B-MISC",
        "I-MISC",
        "I-MISC",
        "O",
        "B-PER",
        "B-ORG",
        "B-MISC",
        "O",
        "O",
    ]
    case_1_seq_bioes = [
        "O",
        "O",
        "B-PER",
        "E-PER",
        "S-ORG",
        "O",
        "B-MISC",
        "I-MISC",
        "E-MISC",
        "O",
        "S-PER",
        "S-ORG",
        "S-MISC",
        "O",
        "O",
    ]
    case_1_seq_bilou = [
        "O",
        "O",
        "B-PER",
        "L-PER",
        "U-ORG",
        "O",
        "B-MISC",
        "I-MISC",
        "L-MISC",
        "O",
        "U-PER",
        "U-ORG",
        "U-MISC",
        "O",
        "O",
    ]
    case_1_spans = [
        {"start": 2, "end": 4, "label": "PER"},
        {"start": 4, "end": 5, "label": "ORG"},
        {"start": 6, "end": 9, "label": "MISC"},
        {"start": 6, "end": 9, "label": "MISC"},
        {"start": 10, "end": 11, "label": "PER"},
        {"start": 11, "end": 12, "label": "ORG"},
        {"start": 12, "end": 13, "label": "MISC"},
    ]
    model_logger.logger_config.tagging_schema = "BIO"
    assert case_1_seq_bio == model_logger._construct_gold_sequence(
        len_sequence, case_1_spans
    )
    model_logger.logger_config.tagging_schema = "BIOES"
    assert case_1_seq_bioes == model_logger._construct_gold_sequence(
        len_sequence, case_1_spans
    )
    model_logger.logger_config.tagging_schema = "BILOU"
    assert case_1_seq_bilou == model_logger._construct_gold_sequence(
        len_sequence, case_1_spans
    )


def test_pred_span_extraction_bio() -> None:
    case_1_seq = ["O", "O", "B-PER", "I-ORG", "B-ORG", "O", "O"]
    case_1_spans = [
        {"start": 2, "end": 3, "label": "PER"},
        {"start": 4, "end": 5, "label": "ORG"},
    ]

    case_2_seq = ["O", "O", "B-PER", "I-PER", "I-ORG", "O", "O"]
    case_2_spans = [{"start": 2, "end": 4, "label": "PER"}]

    case_3_seq = ["O", "O", "B-PER", "I-ORG", "I-PER", "O", "O"]
    case_3_spans = [{"start": 2, "end": 3, "label": "PER"}]

    case_4_seq = ["O", "O", "B-PER", "B-ORG", "B-PER", "O", "B-PER"]
    case_4_spans = [
        {"start": 2, "end": 3, "label": "PER"},
        {"start": 3, "end": 4, "label": "ORG"},
        {"start": 4, "end": 5, "label": "PER"},
        {"start": 6, "end": 7, "label": "PER"},
    ]

    case_5_seq = ["O", "O", "B-PER", "I-PER", "I-PER", "O", "I-PER"]
    case_5_spans = [{"start": 2, "end": 5, "label": "PER"}]

    assert model_logger._extract_spans_bio(case_1_seq) == case_1_spans
    assert model_logger._extract_spans_bio(case_2_seq) == case_2_spans
    assert model_logger._extract_spans_bio(case_3_seq) == case_3_spans
    assert model_logger._extract_spans_bio(case_4_seq) == case_4_spans
    assert model_logger._extract_spans_bio(case_5_seq) == case_5_spans


def test_pred_span_extraction_bioes() -> None:
    model_logger.logger_config.tagging_schema = "BIOES"
    case_1_seq = ["O", "O", "B-PER", "E-PER", "S-ORG", "O", "O"]
    case_1_spans = [
        {"start": 2, "end": 4, "label": "PER"},
        {"start": 4, "end": 5, "label": "ORG"},
    ]

    case_2_seq = ["O", "O", "B-PER", "I-PER", "I-ORG", "B-PER", "O"]
    case_2_spans = []

    case_3_seq = ["O", "O", "S-PER", "S-ORG", "B-PER", "O", "O"]
    case_3_spans = [
        {"start": 2, "end": 3, "label": "PER"},
        {"start": 3, "end": 4, "label": "ORG"},
    ]

    case_4_seq = ["O", "O", "B-PER", "B-ORG", "B-PER", "O", "B-PER"]
    case_4_spans = []

    case_5_seq = ["O", "O", "B-PER", "I-PER", "E-PER", "O", "I-PER"]
    case_5_spans = [{"start": 2, "end": 5, "label": "PER"}]

    assert model_logger._extract_spans_token_level(case_1_seq) == case_1_spans
    assert model_logger._extract_spans_token_level(case_2_seq) == case_2_spans
    assert model_logger._extract_spans_token_level(case_3_seq) == case_3_spans
    assert model_logger._extract_spans_token_level(case_4_seq) == case_4_spans
    assert model_logger._extract_spans_token_level(case_5_seq) == case_5_spans


def test_pred_span_extraction_bilou() -> None:
    model_logger.logger_config.tagging_schema = "BILOU"
    case_1_seq = ["O", "O", "B-PER", "L-PER", "U-ORG", "O", "O"]
    case_1_spans = [
        {"start": 2, "end": 4, "label": "PER"},
        {"start": 4, "end": 5, "label": "ORG"},
    ]

    case_2_seq = ["O", "O", "B-PER", "I-PER", "I-ORG", "B-PER", "O"]
    case_2_spans = []

    case_3_seq = ["O", "O", "U-PER", "U-ORG", "B-PER", "O", "O"]
    case_3_spans = [
        {"start": 2, "end": 3, "label": "PER"},
        {"start": 3, "end": 4, "label": "ORG"},
    ]

    case_4_seq = ["O", "O", "B-PER", "B-ORG", "B-PER", "O", "B-PER"]
    case_4_spans = []

    case_5_seq = ["O", "O", "B-PER", "I-PER", "L-PER", "O", "I-PER"]
    case_5_spans = [{"start": 2, "end": 5, "label": "PER"}]

    assert model_logger._extract_spans_token_level(case_1_seq) == case_1_spans
    assert model_logger._extract_spans_token_level(case_2_seq) == case_2_spans
    assert model_logger._extract_spans_token_level(case_3_seq) == case_3_spans
    assert model_logger._extract_spans_token_level(case_4_seq) == case_4_spans
    assert model_logger._extract_spans_token_level(case_5_seq) == case_5_spans


def test_ner_logging_bad_inputs(
    set_test_config: Callable, cleanup_after_use: Generator
) -> None:
    set_test_config(task_type=TaskType.text_ner)
    dataquality.set_tagging_schema("BIO")

    labels = [
        "B-foo",
        "I-foo",
        "B-bar",
        "I-bar",
        "B-foo-bar",
        "I-foo-bar",
        "B-bar-foo",
        "I-bar-foo",
        "O",
    ]
    dataquality.set_labels_for_run(labels)

    text_inputs = [
        f"sample text for sentence {i}" for i in range(3)
    ]  # (tokens) sample te xt for sent en ce
    token_boundaries_all = [
        [(0, 6), (7, 11), (7, 11), (12, 15), (16, 24), (16, 24), (16, 24), (25, 26)]
        for _ in range(3)
    ]
    gold_spans = [
        [
            {"start": 7, "end": 11, "label": "foo"},
            {"start": 12, "end": 15, "label": "bar"},
            {"start": 16, "end": 27, "label": "bar-foo"},
        ],
        [{"start": 16, "end": 26, "label": "foo-bar"}],
        [],
    ]

    ids = [1, 2, 3]
    split = "training"

    # Handle spans that don't align with token boundaries
    with pytest.raises(AssertionError):
        dataquality.log_data_samples(
            texts=text_inputs,
            text_token_indices=token_boundaries_all,
            gold_spans=gold_spans,
            ids=ids,
            split=split,
        )

    # Handle labels that are in gold but missing in registered labels
    gold_spans = [
        [
            {"start": 7, "end": 11, "label": "foo"},
            {"start": 12, "end": 15, "label": "bar"},
            {"start": 16, "end": 26, "label": "bar-foo"},
        ],
        [{"start": 16, "end": 26, "label": "bad_label"}],
        [],
    ]
    # Handle spans that don't align with token boundaries
    with pytest.raises(AssertionError):
        dataquality.log_data_samples(
            texts=text_inputs,
            text_token_indices=token_boundaries_all,
            gold_spans=gold_spans,
            ids=ids,
            split=split,
        )


@pytest.mark.parametrize("as_tensor", [False, True])
def test_ner_logging(
    as_tensor: bool, cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """
    To validate:
    * pred_conf and pred_loss exist in prob df and have correct shape
    * pred_loss_label exists in prob df
    * assert correct start and end index for extracted spans
    * spans within gold cannot be nested
    * spans within pred cannot be nested
    * all spans should be either gold, pred, or both (never neither)
    * 3 rows
    """
    set_test_config(task_type=TaskType.text_ner)
    dataquality.set_tagging_schema("BIO")
    dataquality.set_split(Split.training)

    labels = [
        "B-foo",
        "I-foo",
        "B-bar",
        "I-bar",
        "B-foo-bar",
        "I-foo-bar",
        "B-bar-foo",
        "I-bar-foo",
        "O",
    ]
    dataquality.set_labels_for_run(labels)

    text_inputs = [
        f"sample text for sentence {i}" for i in range(3)
    ]  # (tokens) sample te xt for sent en ce
    token_boundaries_all = [
        [(0, 6), (7, 11), (7, 11), (12, 15), (16, 24), (16, 24), (16, 24), (25, 26)]
        for _ in range(3)
    ]
    gold_spans = [
        [
            {"start": 7, "end": 11, "label": "foo"},
            {"start": 12, "end": 15, "label": "bar"},
            {"start": 16, "end": 26, "label": "bar-foo"},
        ],
        [{"start": 16, "end": 26, "label": "foo-bar"}],
        [],
    ]

    ids = [0, 1, 2]
    split = "training"

    dataquality.log_data_samples(
        texts=text_inputs,
        text_token_indices=token_boundaries_all,
        gold_spans=gold_spans,
        ids=ids,
    )

    pred_prob = np.array(
        [
            [
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.68, 0.1, 0.1, 0.1, 0, 0, 0.02, 0, 0],
                [0, 0.9, 0, 0.1, 0, 0, 0, 0, 0],
                [0.0, 0.4, 0.6, 0, 0, 0, 0, 0, 0],
                [0.0, 0, 0.05, 0, 0, 0, 0, 0, 0.95],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0.9, 0, 0],
            ],
            [
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0, 0.05, 0, 0.95, 0, 0, 0, 0.0],
                [0.0, 0.05, 0.05, 0, 0, 0.9, 0, 0, 0.0],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0.9, 0, 0],
            ],
            [
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0.9, 0, 0],
                [0.2, 0.05, 0.05, 0, 0, 0, 0, 0.7, 0],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0.9, 0, 0, 0, 0, 0],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
                [0.0, 0.05, 0.05, 0, 0, 0, 0, 0, 0.9],
            ],
        ]
    )

    dataquality.set_epoch(0)
    embs = np.random.rand(3, 8, 5)
    ids = [0, 1, 2]
    if as_tensor:
        embs = tf.convert_to_tensor(embs)
        pred_prob = tf.convert_to_tensor(pred_prob)
        ids = tf.convert_to_tensor(np.array(ids))
    dataquality.log_model_outputs(embs=embs, probs=pred_prob, ids=ids)

    ThreadPoolManager.wait_for_threads()
    c = dataquality.get_data_logger()
    c.validate_labels()
    c.upload()

    pred_spans = [
        [
            {"span_start": 1, "span_end": 3, "pred": "foo"},
            {"span_start": 3, "span_end": 4, "pred": "bar"},
            {"span_start": 7, "span_end": 8, "pred": "bar-foo"},
        ],
        [
            {"span_start": 4, "span_end": 6, "pred": "foo-bar"},
            {"span_start": 7, "span_end": 8, "pred": "bar-foo"},
        ],
        [{"span_start": 1, "span_end": 3, "pred": "bar-foo"}],
    ]

    gold_spans_correct = [
        [
            {"span_start": 1, "span_end": 3, "gold": "foo"},
            {"span_start": 3, "span_end": 4, "gold": "bar"},
            {"span_start": 4, "span_end": 8, "gold": "bar-foo"},
        ],
        [{"span_start": 4, "span_end": 8, "gold": "foo-bar"}],
        [],
    ]

    prob_path = f"{TEST_PATH}/{split}/0/prob/prob.hdf5"
    prob_df = vaex.open(prob_path)
    assert list(prob_df.loss_prob_label.to_numpy()) == [0, 2, 6, 6, 5, 5, 6, 7]
    assert prob_df.loss_prob.shape == (8, 9)  # 8 spans, 9 labels
    assert prob_df.conf_prob.shape == (8, 9)  # 8 spans, 9 labels

    for i in range(3):
        sample_pred_spans = pred_spans[i]
        pred_df = prob_df[prob_df[f"(is_pred) & (sample_id=={i})"]]
        df_pred_spans = pred_df[["span_start", "span_end", "pred"]].to_records()
        assert sample_pred_spans == df_pred_spans

        sample_gold_spans = gold_spans_correct[i]
        gold_df = prob_df[prob_df[f"(is_gold) & (sample_id=={i})"]]
        df_gold_spans = gold_df[["span_start", "span_end", "gold"]].to_records()
        assert sample_gold_spans == df_gold_spans

    assert len(prob_df[prob_df["(~is_gold) & (~is_pred)"]]) == 0

    emb_path = f"{TEST_PATH}/{split}/0/emb/emb.hdf5"
    emb_df = vaex.open(emb_path)

    assert len(emb_df) == len(prob_df)

    sample_path = f"{TEST_PATH}/{split}/0/data/data.arrow"
    sample_df = vaex.open(sample_path)

    assert len(sample_df) == 3

    # Test with logits
    c._cleanup()
    dataquality.set_tagging_schema("BIO")
    dataquality.set_split(Split.training)
    dataquality.set_labels_for_run(labels)
    dataquality.log_data_samples(
        texts=text_inputs,
        text_token_indices=token_boundaries_all,
        gold_spans=gold_spans,
        ids=ids,
        split=split,
    )
    dataquality.log_model_outputs(
        embs=embs,
        logits=pred_prob,
        ids=ids,
        split="training",
        epoch=0,
    )
    ThreadPoolManager.wait_for_threads()
    c = dataquality.get_data_logger()
    c.validate_labels()
    c.upload()
    prob_path = f"{TEST_PATH}/{split}/0/prob/prob.hdf5"
    prob_df = vaex.open(prob_path)
    assert list(prob_df.loss_prob_label.to_numpy()) == [0, 2, 6, 6, 5, 5, 6, 7]
    assert prob_df.loss_prob.shape == (8, 9)  # 8 spans, 9 labels
    assert prob_df.conf_prob.shape == (8, 9)  # 8 spans, 9 labels


def test_ghost_spans() -> None:
    gold_spans = [
        {"start": 7, "end": 11, "label": "foo"},  # span shift
        {"start": 12, "end": 15, "label": "bar"},  # wrong tag
        {"start": 16, "end": 26, "label": "bar-foo"},  # missed span
        {"start": 28, "end": 31, "label": "bar-foo"},  # span shift
    ]
    pred_spans = [
        {"start": 7, "end": 12, "label": "foo"},
        {"start": 12, "end": 15, "label": "bar-bar"},
        {"start": 27, "end": 30, "label": "foo"},
        {"start": 31, "end": 38, "label": "foo-foo"},  # ghost span
    ]
    results = [False, False, False, True]
    logger = TextNERModelLogger()
    for res, pred_span in zip(results, pred_spans):
        assert logger._is_ghost_span(pred_span, gold_spans) == res


def test_duplicate_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_ner")

    ids = list(range(5))

    dataquality.set_labels_for_run(LABELS)
    dataquality.set_tagging_schema("BIO")

    dataquality.log_data_samples(
        texts=TEXT_INPUTS,
        text_token_indices=TEXT_TOKENS,
        ids=ids,
        gold_spans=GOLD_SPANS,
        split="validation",
    )

    dataquality.log_data_samples(
        texts=TEXT_INPUTS,
        text_token_indices=TEXT_TOKENS,
        ids=ids,
        gold_spans=GOLD_SPANS,
        split="training",
    )

    with pytest.raises(GalileoException):
        dataquality.log_data_samples(
            texts=TEXT_INPUTS,
            text_token_indices=TEXT_TOKENS,
            ids=ids,
            gold_spans=GOLD_SPANS,
            split="validation",
        )

    dataquality.log_data_samples(
        texts=TEXT_INPUTS,
        text_token_indices=TEXT_TOKENS,
        ids=ids,
        gold_spans=GOLD_SPANS,
        split="test",
    )

    with pytest.raises(GalileoException):
        dataquality.log_data_samples(
            texts=TEXT_INPUTS,
            text_token_indices=TEXT_TOKENS,
            ids=ids,
            gold_spans=GOLD_SPANS,
            split="training",
        )


def test_duplicate_output_rows(set_test_config, cleanup_after_use) -> None:
    set_test_config(task_type="text_ner")

    ids = list(range(5))

    dataquality.set_labels_for_run(LABELS)
    dataquality.set_tagging_schema("BIO")

    dataquality.log_data_samples(
        texts=TEXT_INPUTS,
        text_token_indices=TEXT_TOKENS,
        ids=ids,
        gold_spans=GOLD_SPANS,
        split="validation",
    )

    dataquality.log_data_samples(
        texts=TEXT_INPUTS,
        text_token_indices=TEXT_TOKENS,
        ids=ids,
        gold_spans=GOLD_SPANS,
        split="training",
    )

    embs = [np.random.rand(119, 768) for _ in range(5)]
    logits = [np.random.rand(119, 28) for _ in range(5)]

    dataquality.log_model_outputs(
        embs=embs, logits=logits, ids=ids, split="training", epoch=0
    )
    dataquality.log_model_outputs(
        embs=embs, logits=logits, ids=ids, split="training", epoch=0
    )

    with pytest.raises(GalileoException) as e:
        dataquality.get_data_logger().upload()

    assert str(e.value).startswith("It seems as though you have duplicate spans")


def test_log_data_sample(
    set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_ner")
    dataquality.set_labels_for_run(LABELS)
    dataquality.set_tagging_schema("BIO")
    logger = TextNERDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        inp = {
            "text": TEXT_INPUTS[0],
            "text_token_indices": TEXT_TOKENS[0],
            "gold_spans": GOLD_SPANS[0],
            "id": 0,
            "split": "training",
        }
        dataquality.log_data_sample(**inp)

        assert logger.texts == [inp["text"]]
        assert logger.gold_spans == [inp["gold_spans"]]
        assert logger.ids == [0]
        assert logger.split == Split.training
        assert not hasattr(logger, "text_token_indices")  # Deleted after log
        flattened_tokens = list(chain(*inp["text_token_indices"]))
        assert logger.text_token_indices_flat == [flattened_tokens]


@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame(NER_INPUT_DATA),
        vaex.from_dict(NER_INPUT_DATA),
        NER_INPUT_ITER,
        datasets.Dataset.from_dict(NER_INPUT_DATA),
    ],
)
def test_log_dataset(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_ner")
    dataquality.set_labels_for_run(LABELS)
    dataquality.set_tagging_schema("BIO")
    logger = TextNERDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        dataquality.log_dataset(
            dataset,
            text="my_text",
            gold_spans="my_spans",
            id="my_id",
            text_token_indices="text_tokens",
            split="train",
        )

        assert logger.texts == TEXT_INPUTS
        assert logger.gold_spans == GOLD_SPANS
        # We delete it on successful log
        assert not hasattr(logger, "text_token_indices")
        flattened_input = [list(chain(*ind)) for ind in TEXT_TOKENS]
        assert logger.text_token_indices_flat == flattened_input
        assert logger.ids == list(range(5))
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "dataset",
    [
        NER_INPUT_TUPLES,
    ],
)
def test_log_dataset_tuple(
    dataset: DataSet, set_test_config: Callable, cleanup_after_use: Callable
) -> None:
    set_test_config(task_type="text_ner")
    dataquality.set_labels_for_run(LABELS)
    dataquality.set_tagging_schema("BIO")
    logger = TextNERDataLogger()

    with mock.patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger

        dataquality.log_dataset(
            dataset, text=0, gold_spans=1, id=2, text_token_indices=3, split="train"
        )

        assert logger.texts == TEXT_INPUTS
        assert logger.gold_spans == GOLD_SPANS
        # We delete it on successful log
        assert not hasattr(logger, "text_token_indices")
        flattened_input = [list(chain(*ind)) for ind in TEXT_TOKENS]
        assert logger.text_token_indices_flat == flattened_input
        assert logger.ids == list(range(5))
        assert logger.split == Split.training


@pytest.mark.parametrize(
    "labels, expected_err_message",
    [
        (LABELS, None),
        (
            [""],
            "No valid labels found among ['', 'NOT_']. A valid label is one that "
            "starts with either B-, I-, L-, E-, S-, or U- according to your "
            "particular tagging scheme",
        ),
        (
            ["B-"],
            "The class names following the tag cannot be empty. For example 'B-' is "
            "not allowed, but 'B-some_class' is.",
        ),
        (
            ["B_starting_word"],
            "No valid labels found among ['B_starting_word', 'NOT_B_starting_word']. "
            "A valid label is one that starts with either B-, I-, L-, E-, S-, or U- "
            "according to your particular tagging scheme",
        ),
    ],
)
def test_is_valid_span_label(
    set_test_config: Callable, labels: List[str], expected_err_message: str
):
    set_test_config(task_type="text_ner")

    dataquality.set_labels_for_run(labels)
    if expected_err_message:
        with pytest.raises(AssertionError) as e:
            TextNERDataLogger._clean_labels()
        assert e.value.args[0] == expected_err_message
    else:
        TextNERDataLogger._clean_labels()


@mock.patch("dataquality.utils.vaex.create_data_embs")
def test_create_and_upload_data_embs_skipped_ner(
    mock_create_embs: mock.MagicMock,
    cleanup_after_use: Callable,
    set_test_config: Callable,
) -> None:
    set_test_config(task_type="text_ner")

    df = vaex.from_arrays(id=list(range(10)))
    df["text"] = "sentence number " + df["id"].astype(str)
    logger = TextNERDataLogger()
    with pytest.warns(GalileoWarning):
        logger.create_and_upload_data_embs(df, "training", 3)
    mock_create_embs.assert_not_called()
