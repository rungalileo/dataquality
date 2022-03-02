import numpy as np

from dataquality.loggers.data_logger.text_ner import TextNERDataLogger
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger

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
