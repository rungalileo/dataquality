import pickle
from typing import Dict, List, Tuple
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import spacy
import vaex
from spacy.language import Language
from spacy.pipeline.ner import EntityRecognizer
from spacy.tokens import Doc
from spacy.training import Example

import dataquality
from dataquality.exceptions import GalileoException
from dataquality.integrations.spacy import (
    GalileoEntityRecognizer,
    GalileoTransitionBasedParserModel,
    log_input_docs,
    log_input_examples,
    unwatch,
    watch,
)
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCATION
from tests.test_utils.spacy_integration import load_ner_data_from_local, train_model
from tests.test_utils.spacy_integration_constants import (
    LONG_SHORT_DATA,
    LONG_TRAIN_DATA,
    MISALIGNED_SPAN_DATA,
    NER_CLASS_LABELS,
    NER_TRAINING_DATA,
    TestSpacyExpectedResults,
)
from tests.test_utils.spacy_integration_constants_inference import (
    NER_INFERENCE_DATA,
    NER_INFERENCE_PRED_TOKEN_SPANS,
    TestSpacyInfExpectedResults,
)


def test_log_input_examples_without_watch():
    with pytest.raises(GalileoException) as e:
        log_input_examples(NER_TRAINING_DATA, split="training")
    assert (
        e.value.args[0]
        == "Galileo does not have any logged labels. Did you forget to call "
        "watch(nlp) before log_input_examples(...)?"
    )


def test_log_input_list_of_tuples(nlp_watch: Language) -> None:
    with pytest.raises(GalileoException) as e:
        log_input_examples(NER_TRAINING_DATA, "training")
    assert (
        e.value.args[0]
        == "Expected a <class 'spacy.training.example.Example'>. Received "
        "<class 'tuple'>"
    )


def test_log_input_examples(
    nlp_init: Language, training_examples: List[Example]
) -> None:
    watch(nlp_init)
    log_input_examples(training_examples, "training")

    # assert that we added ids to the examples for later joining with model outputs
    assert all(
        [
            examples.predicted.user_data["id"] == i
            for i, examples in enumerate(training_examples)
        ]
    )

    logged_data = vaex.open(f"{LOCATION}/input_data/training/data_0.arrow")

    assert logged_data["id"].tolist() == [0, 1, 2, 3, 4]
    assert logged_data["split"].tolist() == ["training"] * len(training_examples)
    assert all(
        [
            text == NER_TRAINING_DATA[i][0]
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

        assert NER_TRAINING_DATA[i][1]["entities"] == ents_as_char_idxs


def test_watch(nlp_init: Language) -> None:
    watch(nlp_init)

    assert text_ner_logger_config.helper_data["nlp"] == nlp_init
    assert dataquality.get_data_logger().logger_config.labels == NER_CLASS_LABELS
    assert dataquality.get_data_logger().logger_config.tagging_schema == "BILOU"

    assert isinstance(nlp_init.get_pipe("ner"), EntityRecognizer)
    assert isinstance(nlp_init.get_pipe("ner"), GalileoEntityRecognizer)


def test_unwatch(nlp: Language, training_examples: List[Example]) -> None:
    original_ner = nlp.get_pipe("ner")
    nlp.initialize(lambda: training_examples)

    # This should be possible here
    pickle.dumps(nlp)
    # and this
    pickle.loads(pickle.dumps(nlp.get_pipe("ner")))
    # and this
    pickle.loads(pickle.dumps(nlp))

    for _ in range(3):  # This should be possible multiple times
        watch(nlp)
        unwatch(nlp)

    unwatched_ner = nlp.get_pipe("ner")
    assert isinstance(unwatched_ner, EntityRecognizer)
    assert not isinstance(unwatched_ner, GalileoEntityRecognizer)
    assert not isinstance(unwatched_ner.model, GalileoTransitionBasedParserModel)
    assert unwatched_ner.model == original_ner.model
    assert unwatched_ner.moves == original_ner.moves

    # Should be able to now save + load the pipeline component
    pickle.loads(pickle.dumps(nlp.get_pipe("ner")))
    # and the language
    pickle.loads(pickle.dumps(nlp))


def test_embeddings_get_updated(
    nlp: Language,
    training_examples: List[Example],
) -> None:
    """This test both checks our spacy wrapper end to end and that embs update.

    If embeddings stop updating that means the spacy architecture somehow changed
    and would make our user's embeddings seem meaningless
    """
    train_model(nlp, training_examples, num_epochs=2)
    unwatch(nlp)

    _, embs, _ = load_ner_data_from_local("training", inf_name_or_epoch=1)
    embs = embs["emb"].to_numpy()

    dataquality.get_data_logger()._cleanup()

    train_model(nlp, training_examples, num_epochs=1)

    _, embs_2, _ = load_ner_data_from_local("training", inf_name_or_epoch=0)
    embs_2 = embs_2["emb"].to_numpy()

    assert embs.shape == embs_2.shape
    assert not np.allclose(embs, embs_2)


def test_spacy_ner(
    nlp: Language,
    training_examples: List[Example],
) -> None:
    """An end to end test of functionality"""
    spacy.util.fix_random_seed(0)
    num_epochs = 2
    training_losses = train_model(nlp, training_examples, num_epochs=num_epochs)

    training_losses = np.array(training_losses).astype(np.float32)
    res = np.array(
        [25.50000334, 14.20009732, 41.1032235, 13.86421746], dtype=np.float32
    )
    assert np.allclose(training_losses, res, atol=1e-01)

    data, embs, probs = load_ner_data_from_local(
        "training", inf_name_or_epoch=num_epochs - 1
    )

    assert len(data) == 5
    assert all(data["id"] == range(len(data)))
    assert data.equals(
        TestSpacyExpectedResults.gt_data
    ), f"Received the following data df {data}"

    assert embs["id"].tolist() == list(range(len(embs)))
    embs = embs["emb"].to_numpy().astype(np.float16)
    assert all([span_emb in TestSpacyExpectedResults.gt_embs for span_emb in embs])

    assert len(probs) == 8
    # arrange the probs array to account for misordering of logged samples
    probs = probs.sort(["sample_id", "span_start"])
    # Drop multi-dimensional columns
    probs = probs.drop(["id", "conf_prob", "loss_prob"]).to_pandas_df()
    probs.equals(TestSpacyExpectedResults.gt_probs)


@pytest.mark.parametrize(
    "samples",
    [LONG_SHORT_DATA, LONG_TRAIN_DATA, NER_TRAINING_DATA],
)
def test_long_sample(
    samples: List[Tuple[str, Dict]],
    nlp: Language,
):
    """Tests logging a long sample during training"""
    all_examples = [
        Example.from_dict(nlp.make_doc(text), entities) for text, entities in samples
    ]
    optimizer = nlp.initialize(lambda: all_examples)

    old_log = TextNERModelLogger.log

    def new_log(*args, **kwargs):
        logger: TextNERModelLogger = args[0]
        assert len(logger.ids) == len(samples)
        assert len(logger.logits) == len(samples)
        assert len(logger.embs) == len(samples)

    TextNERModelLogger.log = new_log

    watch(nlp)
    log_input_examples(all_examples, split="training")

    dataquality.set_split("training")
    for epoch in range(2):
        dataquality.set_epoch(epoch)
        losses = {}
        nlp.update(all_examples, drop=0.5, sgd=optimizer, losses=losses)

    TextNERModelLogger.log = old_log
    ThreadPoolManager.wait_for_threads()
    del nlp


def test_spacy_does_not_log_misaligned_entities(nlp: Language) -> None:
    def make_examples(data):
        examples = []
        for text, annotations in data:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        return examples

    nlp.initialize(lambda: make_examples(NER_TRAINING_DATA))

    training_examples = make_examples(MISALIGNED_SPAN_DATA)

    assert len(training_examples[0].reference.ents) == 0

    nlp.initialize(lambda: training_examples)

    # Galileo code
    watch(nlp)
    log_input_examples(training_examples, "training")

    logged_gold_spans = dataquality.get_data_logger().logger_config.gold_spans
    assert len(logged_gold_spans["training_0"]) == 0


@pytest.mark.parametrize(
    "training_data",
    [
        NER_TRAINING_DATA,
        [(text, {"entities": []}) for text, entities in NER_TRAINING_DATA],
        [
            (text, entities if i != 1 else {"entities": []})
            for i, (text, entities) in enumerate(NER_TRAINING_DATA)
        ],
    ],
)
def test_log_input_examples_have_no_gold_spans(nlp: Language, training_data: List):
    def make_examples(data):
        examples = []
        for text, annotations in data:
            doc = nlp.make_doc(text)
            examples.append(Example.from_dict(doc, annotations))
        return examples

    nlp.initialize(lambda: make_examples(NER_TRAINING_DATA))

    training_examples = make_examples(training_data)

    watch(nlp)
    log_input_examples(training_examples, "training")

    samples_logged_gold_spans = dataquality.get_data_logger().logger_config.gold_spans

    for i, (_, logged_gold_spans) in enumerate(samples_logged_gold_spans.items()):
        original_spans = training_examples[i].reference.ents

        assert len(logged_gold_spans) == len(original_spans)
        for logged_gold_span, original_span in zip(logged_gold_spans, original_spans):
            assert logged_gold_span[0] == original_span.start
            assert logged_gold_span[1] == original_span.end
            assert logged_gold_span[2] == original_span.label_


def test_watch_nlp_with_no_gold_labels(nlp: Language) -> None:
    nlp.initialize()
    with pytest.raises(GalileoException) as e:
        watch(nlp)
    assert e.value.args[0] == (
        "Your nlp seems to not have been initialized with any "
        "ground truth spans. Galileo needs all labels to have "
        "been added to the model before calling "
        "watch(nlp). Please run `nlp.initialize(lambda: your_examples)` over a "
        "list of examples that include all of the gold spans you plan to make "
        "predictions over."
    )


@mock.patch("dataquality.utils.spacy_integration.is_spacy_using_gpu", return_value=True)
def test_require_cpu(
    mock_spacy_gpu: mock.MagicMock,
    nlp: Language,
    training_examples: List[Example],
) -> None:
    with pytest.raises(GalileoException):
        train_model(nlp, training_examples)


def test_log_input_examples_inference_split() -> None:
    with pytest.raises(GalileoException) as e:
        log_input_examples([], "inference")

    assert (
        e.value.args[0] == "`log_input_examples` cannot be used to log inference data. "
        "Try using `log_input_docs` instead."
    )


def test_log_input_docs_without_watch() -> None:
    with pytest.raises(GalileoException) as e:
        log_input_docs([], "inf-name")

    assert (
        e.value.args[0]
        == "Galileo does not have any logged labels. Did you forget to call "
        "watch(nlp) before log_input_examples(...)?"
    )


def test_log_input_docs_list_of_strs(nlp_watch: Language) -> None:
    with pytest.raises(GalileoException) as e:
        log_input_docs(NER_INFERENCE_DATA, "inf-name")
    assert (
        e.value.args[0] == "Expected a <class 'spacy.tokens.doc.Doc'>. Received "
        "<class 'str'>"
    )


def test_log_input_docs(nlp_watch: Language, inference_docs: List[Doc]) -> None:
    log_input_docs(inference_docs, "inf-name")

    # assert that we added ids to the docs for later joining with model outputs
    assert all([doc.user_data["id"] == i for i, doc in enumerate(inference_docs)])

    logged_data = vaex.open(f"{LOCATION}/input_data/inference/data_0.arrow")

    assert logged_data["id"].tolist() == [0, 1, 2, 3, 4]
    assert logged_data["split"].tolist() == ["inference"] * len(inference_docs)
    assert logged_data["inference_name"].tolist() == ["inf-name"] * len(inference_docs)
    assert all(
        [
            text == NER_INFERENCE_DATA[i]
            for i, text in enumerate(logged_data["text"].tolist())
        ]
    )
    # Checks that logged data was tokenized correctly
    logged_token_indices = logged_data["text_token_indices"].tolist()
    expected_token_indices = [
        list(row) for row in TestSpacyInfExpectedResults.gt_data.text_token_indices
    ]
    assert logged_token_indices == expected_token_indices


@mock.patch.object(TextNERModelLogger, "_extract_pred_spans")
def test_spacy_inference_only(
    mock_extract_pred_spans: MagicMock, nlp_watch: Language, inference_docs: List[Doc]
) -> None:
    spacy.util.fix_random_seed(0)
    # We don't care what the nlp model actually predicts,
    # mock the response to ensure pred_spans exist
    mock_extract_pred_spans.side_effect = NER_INFERENCE_PRED_TOKEN_SPANS

    log_input_docs(inference_docs, "inf-name")
    dataquality.set_split("inference", "inf-name")
    for doc in inference_docs:
        nlp_watch(doc)

    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()

    data, embs, probs = load_ner_data_from_local(
        "inference", inf_name_or_epoch="inf-name"
    )

    assert len(data) == 5
    assert all(data["id"] == range(len(data)))
    assert data.equals(TestSpacyInfExpectedResults.gt_data)

    assert embs["id"].tolist() == list(range(7))
    embs = embs["emb"].to_numpy()
    # Since order might change due to multi-threading we verify each embedding
    # is in the expected list, but don't check the exact order
    assert all([span_emb in TestSpacyInfExpectedResults.gt_embs for span_emb in embs])

    # arrange the probs array to account for misordering of logged samples
    assert len(probs) == 7
    probs = probs.sort(["sample_id", "span_start"])
    conf_probs = probs["conf_prob"].to_numpy()
    assert np.isclose(conf_probs, TestSpacyInfExpectedResults.gt_conf_prob).all()

    # Drop conf_prob since pandas doesn't support multi-dimensional arrays
    pdf = probs.drop(["id", "conf_prob"]).to_pandas_df()
    assert pdf.equals(TestSpacyInfExpectedResults.gt_probs)


def test_spacy_training_then_inference(
    nlp: Language, training_examples: List[Example], inference_docs: List[Doc]
) -> None:
    """Test that we can log training data, then inference data

    We don't assert exact values here, just that the data is logged to the correct files
    and a few simple assertions about the DF lengths and split names.
    """
    spacy.util.fix_random_seed(0)

    # first log the training run
    num_epochs = 2
    train_model(nlp, training_examples, num_epochs=num_epochs)

    with mock.patch.object(
        TextNERModelLogger, "_extract_pred_spans"
    ) as mock_extract_pred_spans:
        # We don't care what the nlp model actually predicts,
        # mock the response to ensure pred_spans exist
        mock_extract_pred_spans.side_effect = NER_INFERENCE_PRED_TOKEN_SPANS

        log_input_docs(inference_docs, "inf-name")
        dataquality.set_split("inference", "inf-name")
        for doc in inference_docs:
            # NLP is watched in training pipeline so we don't need to watch here
            nlp(doc)

    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()

    train_data, train_embs, train_probs = load_ner_data_from_local(
        "training", inf_name_or_epoch=1
    )
    assert len(train_data) == 5
    assert len(train_embs) == len(train_probs) == 8
    assert "inference_name" not in train_data.columns
    assert train_data.split.tolist() == ["training"] * 5

    inf_data, inf_embs, inf_probs = load_ner_data_from_local(
        "inference", inf_name_or_epoch="inf-name"
    )
    assert len(inf_data) == 5
    assert inf_data.split.tolist() == ["inference"] * 5
    assert inf_data.inference_name.tolist() == ["inf-name"] * 5
    assert len(inf_embs) == len(inf_probs) == 7
