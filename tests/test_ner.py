from dataquality.loggers.data_logger.text_ner import TextNERDataLogger
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger

model_logger = TextNERModelLogger()


def test_pred_span_extraction() -> None:
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


def test_gold_span_extraction() -> None:
    logger = TextNERDataLogger()
    gold_spans = [
        {"start": 0, "end": 4, "label": "REVIEW"},
        {"start": 12, "end": 23, "label": "YEAR"},
    ]
    token_indices = [(0, 4), (5, 11), (12, 14), (15, 18), (19, 23)]
    new_spans = logger._extract_gold_spans(gold_spans, token_indices)
    good_new_spans = [
        {"start": 0, "end": 1, "label": "REVIEW"},
        {"start": 2, "end": 5, "label": "YEAR"},
    ]
    assert new_spans == good_new_spans
