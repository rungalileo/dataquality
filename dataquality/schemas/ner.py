from enum import Enum, unique


@unique
class NERProbMethod(str, Enum):
    confidence = "confidence"
    loss = "loss"


@unique
class NERErrorType(str, Enum):
    wrong_tag = "wrong_tag"
    missed_label = "missed_label"
    span_shift = "span_shift"
    ghost_span = "ghost_span"
    none = "None"  # Indicating that there is no error for this span (correct pred)


@unique
class TaggingSchema(str, Enum):
    BIO = "BIO"
    BILOU = "BILOU"
    BIOES = "BIOES"
    # IOB2 = "IOB2"
    # IOB = "IOB"
    # BILOES = "BILOES"


class NERColumns(str, Enum):
    id = "id"
    sample_id = "sample_id"
    split = "split"  # type: ignore
    epoch = "epoch"
    is_gold = "is_gold"
    is_pred = "is_pred"
    span_start = "span_start"
    span_end = "span_end"
    gold = "gold"
    pred = "pred"
    conf_prob = "conf_prob"
    loss_prob = "loss_prob"
    loss_prob_label = "loss_prob_label"
    galileo_error_type = "galileo_error_type"
    emb = "emb"
    inference_name = "inference_name"
