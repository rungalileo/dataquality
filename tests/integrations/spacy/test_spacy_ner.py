import pickle
from typing import Callable, Dict, List, Tuple
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
import spacy
import vaex
from spacy.pipeline.ner import EntityRecognizer
from spacy.training import Example

import dataquality
from dataquality.exceptions import GalileoException
from dataquality.integrations.spacy import (
    GalileoEntityRecognizer,
    GalileoTransitionBasedParserModel,
    log_input_examples,
    unwatch,
    watch,
)
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import LOCATION
from tests.test_utils.spacy_integration import load_ner_data_from_local, train_model
from tests.test_utils.spacy_integration_constants import (
    LONG_SHORT_DATA,
    LONG_TRAIN_DATA,
    MISALIGNED_SPAN_DATA,
    NER_CLASS_LABELS,
    NER_TEST_DATA,
    NER_TRAINING_DATA,
    TestSpacyNerConstants,
)


def test_log_input_examples_without_watch(set_test_config, cleanup_after_use):
    text_ner_logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    with pytest.raises(GalileoException) as e:
        log_input_examples(NER_TRAINING_DATA, split="training")
    assert (
        e.value.args[0]
        == "Galileo does not have any logged labels. Did you forget to call "
        "watch(nlp) before log_input_examples(...)?"
    )


def test_log_input_list_of_tuples(set_test_config, cleanup_after_use):
    text_ner_logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))
    nlp.initialize(lambda: training_examples)

    watch(nlp)

    with pytest.raises(GalileoException) as e:
        log_input_examples(NER_TRAINING_DATA, "training")
    assert (
        e.value.args[0]
        == "Expected a <class 'spacy.training.example.Example'>. Received "
        "<class 'tuple'>"
    )


def test_log_input_examples(set_test_config, cleanup_after_use):
    text_ner_logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
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


def test_watch(set_test_config, cleanup_after_use):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)
    watch(nlp)

    assert text_ner_logger_config.helper_data["nlp"] == nlp
    assert dataquality.get_data_logger().logger_config.labels == NER_CLASS_LABELS
    assert dataquality.get_data_logger().logger_config.tagging_schema == "BILOU"

    assert isinstance(nlp.get_pipe("ner"), EntityRecognizer)
    assert isinstance(nlp.get_pipe("ner"), GalileoEntityRecognizer)


def test_unwatch(set_test_config):
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    original_ner = nlp.add_pipe("ner")

    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

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


def test_embeddings_get_updated(cleanup_after_use, set_test_config):
    """This test both checks our spacy wrapper end to end and that embs update.

    If embeddings stop updating that means the spacy architecture somehow changed
    and would make our user's embeddings seem meaningless
    """
    set_test_config(task_type=TaskType.text_ner)
    train_model(
        training_data=NER_TRAINING_DATA, test_data=NER_TRAINING_DATA, num_epochs=2
    )

    _, embs, _ = load_ner_data_from_local("training", epoch=1)
    embs = embs["emb"].to_numpy()

    dataquality.get_data_logger()._cleanup()

    train_model(
        training_data=NER_TRAINING_DATA, test_data=NER_TRAINING_DATA, num_epochs=1
    )

    _, embs_2, _ = load_ner_data_from_local("training", epoch=0)
    embs_2 = embs_2["emb"].to_numpy()

    assert embs.shape == embs_2.shape
    assert not np.allclose(embs, embs_2)


def test_spacy_ner(cleanup_after_use, set_test_config) -> None:
    """An end to end test of functionality"""
    spacy.util.fix_random_seed()
    set_test_config(task_type=TaskType.text_ner)
    num_epochs = 2
    training_losses = train_model(
        NER_TRAINING_DATA, NER_TEST_DATA, num_epochs=num_epochs
    )

    training_losses = np.array(training_losses).astype(np.float32)
    res = np.array(
        [25.50000334, 14.20009732, 41.1032235, 13.86421746], dtype=np.float32
    )
    assert np.allclose(training_losses, res, atol=1e-01)

    data, embs, probs = load_ner_data_from_local("training", epoch=num_epochs - 1)

    assert len(data) == 5
    assert all(data["id"] == range(len(data)))
    assert data.equals(
        TestSpacyNerConstants.gt_data
    ), f"Received the following data df {data}"

    assert embs["id"].tolist() == list(range(len(embs)))
    embs = embs["emb"].to_numpy().astype(np.float16)
    assert all([span_emb in TestSpacyNerConstants.gt_embs for span_emb in embs])

    # arrange the probs array to account for misordering of logged samples
    probs = (
        probs.sort_values(by=["sample_id", "span_start"]).drop("id", axis=1).round(4)
    )
    assert len(probs) == 8

    gt_probs = TestSpacyNerConstants.gt_probs
    for c in probs.columns:
        if np.issubdtype(probs[c].dtype, np.number):
            assert np.allclose(probs[c].values, gt_probs[c].values, atol=1e-3)
        else:
            assert (probs[c].values == gt_probs[c].values).all()


@pytest.mark.parametrize(
    "samples",
    [LONG_SHORT_DATA, LONG_TRAIN_DATA, NER_TRAINING_DATA],
)
def test_long_sample(
    samples: List[Tuple[str, Dict]],
    cleanup_after_use: Callable,
    set_test_config: Callable,
):
    """Tests logging a long sample during training"""
    TextNERModelLogger.logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    nlp = spacy.blank("en")
    nlp.add_pipe("ner")
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


def test_inference_split_raises_warning(
    cleanup_after_use: Callable, set_test_config: Callable
) -> None:
    """Tests that inference mode raises a warning and continues without dq client"""
    TextNERModelLogger.logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    nlp = spacy.blank("en")
    nlp.add_pipe("ner")
    all_examples = [
        Example.from_dict(nlp.make_doc(text), entities)
        for text, entities in NER_TRAINING_DATA
    ]
    nlp.initialize(lambda: all_examples)
    watch(nlp)
    dataquality.set_split(split="inference", inference_name="some_name")

    with patch(
        "dataquality.loggers.model_logger.text_ner.TextNERModelLogger"
    ) as mocked_model_logger_log:
        with pytest.warns(UserWarning) as record:
            nlp("some text here")
            assert len(record) == 1
            assert (
                record[0].message.args[0]
                == "Inference logging with Galileo coming soon. For now skipping "
                "logging"
            )
        assert not mocked_model_logger_log.called


def test_spacy_does_not_log_misaligned_entities(cleanup_after_use, set_test_config):
    TextNERModelLogger.logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    nlp = spacy.blank("en")
    nlp.add_pipe("ner", last=True)

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
def test_log_input_examples_have_no_gold_spans(
    set_test_config, cleanup_after_use, training_data
):
    TextNERModelLogger.logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")

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


def test_watch_nlp_with_no_gold_labels(set_test_config, cleanup_after_use):
    TextNERModelLogger.logger_config.reset()
    set_test_config(task_type=TaskType.text_ner)

    nlp = spacy.blank("en")
    nlp.add_pipe("ner")
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
    set_test_config: Callable,
    cleanup_after_use: Callable,
) -> None:
    with pytest.raises(GalileoException):
        train_model(NER_TRAINING_DATA, NER_TEST_DATA)
