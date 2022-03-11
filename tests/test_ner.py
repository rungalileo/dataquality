from typing import Callable

import numpy as np
import pytest
import vaex

import dataquality
from dataquality.loggers.data_logger.text_ner import TextNERDataLogger
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import TEST_PATH

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

    assert model_logger._extract_pred_spans_bio(case_1_seq) == case_1_spans
    assert model_logger._extract_pred_spans_bio(case_2_seq) == case_2_spans
    assert model_logger._extract_pred_spans_bio(case_3_seq) == case_3_spans
    assert model_logger._extract_pred_spans_bio(case_4_seq) == case_4_spans
    assert model_logger._extract_pred_spans_bio(case_5_seq) == case_5_spans


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

    assert model_logger._extract_pred_spans_bioes(case_1_seq) == case_1_spans
    assert model_logger._extract_pred_spans_bioes(case_2_seq) == case_2_spans
    assert model_logger._extract_pred_spans_bioes(case_3_seq) == case_3_spans
    assert model_logger._extract_pred_spans_bioes(case_4_seq) == case_4_spans
    assert model_logger._extract_pred_spans_bioes(case_5_seq) == case_5_spans


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

    assert model_logger._extract_pred_spans_bilou(case_1_seq) == case_1_spans
    assert model_logger._extract_pred_spans_bilou(case_2_seq) == case_2_spans
    assert model_logger._extract_pred_spans_bilou(case_3_seq) == case_3_spans
    assert model_logger._extract_pred_spans_bilou(case_4_seq) == case_4_spans
    assert model_logger._extract_pred_spans_bilou(case_5_seq) == case_5_spans


def test_calculate_dep_score_across_spans() -> None:
    dep_scores = [0.9, 0.2, 0.4, 0.3, 0.7, 0.9, 0.4, 0.3, 0.4, 0.3, 0.7, 0.9, 0.4, 0.3]
    spans = [
        {"start": 2, "end": 5, "label": "PER"},
        {"start": 6, "end": 10, "label": "MISC"},
        {"start": 12, "end": 13, "label": "ORG"},
    ]
    span_dep_score = [0.7, 0.4, 0.4]
    assert span_dep_score == model_logger._calculate_dep_score_across_spans(
        spans, dep_scores
    )


def test_calculate_dep_scores() -> None:
    model_logger.logger_config.tagging_schema = "BIO"
    pred_prob = np.array(
        [
            [0.9, 0.05, 0.05, 0, 0, 0, 0],
            [0.1, 0.7, 0.1, 0.1, 0, 0, 0],
            [0, 0, 0, 0.1, 0.9, 0, 0],
            [0.0, 0.4, 0.6, 0, 0, 0, 0],
            [0.2, 0.05, 0.05, 0, 0, 0.7, 0],
        ]
    )
    model_logger.logger_config.labels = [
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "O",
        "B-MISC",
        "I-MISC",
    ]
    sample_len = 5
    gold_spans = [
        {"start": 0, "end": 2, "label": "PER"},
        {"start": 4, "end": 5, "label": "MISC"},
    ]
    pred_spans = [
        {"start": 3, "end": 4, "label": "ORG"},
        {"start": 4, "end": 5, "label": "MISC"},
    ]
    gold_dep, pred_dep = model_logger._calculate_dep_scores(
        pred_prob, gold_spans, pred_spans, sample_len
    )
    assert gold_dep == [0.2, 0.25]
    assert pred_dep == [0.8, 0.25]


def test_ner_logging_bad_inputs(set_test_config: Callable, cleanup_after_use) -> None:
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
        dataquality.log_input_data(
            text=text_inputs,
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
        dataquality.log_input_data(
            text=text_inputs,
            text_token_indices=token_boundaries_all,
            gold_spans=gold_spans,
            ids=ids,
            split=split,
        )


def test_ner_logging(cleanup_after_use: Callable, set_test_config: Callable) -> None:
    """
    To validate:
    * dep scores are all 0 <= dep <= 1
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

    dataquality.log_input_data(
        text=text_inputs,
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
    dataquality.log_model_outputs(
        emb=np.random.rand(3, 8, 5),
        probs=pred_prob,
        ids=[0, 1, 2],
    )

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
    for i in prob_df.data_error_potential.to_numpy():
        assert 0 <= i <= 1

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
    dataquality.set_labels_for_run(labels)
    dataquality.log_input_data(
        text=text_inputs,
        text_token_indices=token_boundaries_all,
        gold_spans=gold_spans,
        ids=ids,
        split=split,
    )
    dataquality.log_model_outputs(
        emb=np.random.rand(3, 8, 5),
        logits=pred_prob,
        ids=[0, 1, 2],
        split="training",
        epoch=0,
    )
    ThreadPoolManager.wait_for_threads()
    c = dataquality.get_data_logger()
    c.validate_labels()
    c.upload()
    prob_path = f"{TEST_PATH}/{split}/0/prob/prob.hdf5"
    prob_df = vaex.open(prob_path)
    for i in prob_df.data_error_potential.to_numpy():
        assert 0 <= i <= 1


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
