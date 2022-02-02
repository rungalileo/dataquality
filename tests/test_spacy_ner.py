import os
# TODO: Why should I not need these again?
from spacy.pipeline.ner import EntityRecognizer

os.environ["GALILEO_API_URL"] = "https://api.dev.rungalileo.io"
os.environ["GALILEO_MINIO_URL"] = "data.dev.rungalileo.io"
os.environ["GALILEO_MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["GALILEO_MINIO_SECRET_KEY"] = "minioadmin"
os.environ["GALILEO_AUTH_METHOD"] = "email"
os.environ["GALILEO_USERNAME"] = "adminy_guy@rungalileo.io"
os.environ["GALILEO_PASSWORD"] = "Admin123@"

import dataquality
from dataquality.core.integrations.spacy import watch, log_input_examples, \
    GalileoEntityRecognizer, unwatch
from dataquality.schemas.task_type import TaskType
import spacy
import random
from tqdm import tqdm
from spacy.training import Example
from spacy.util import minibatch
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from tests.conftest import LOCATION, SPLITS, SUBDIRS, TEST_PATH
from spacy.language import Language
from typing import List, Tuple, Dict
from thinc.model import Model
from spacy.tokens import Doc
import numpy as np
from tests.utils.data_utils import validate_uploaded_data
import vaex
from vaex.dataframe import DataFrameLocal
from typing import Callable
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger

spacy.util.fix_random_seed()

dataquality.config.task_type = TaskType.text_ner

def _train_model(training_data: List[Tuple[str, Dict]], test_data: List[Tuple[str, Dict]], nlp: Language):
    n_iter = 5
    minibatch_size = 3
    ner = nlp.get_pipe("ner")

    # Spacy pre-processing
    training_examples = []
    for text, annotations in training_data:
        # 1) Setting the correct num labels
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

        # 2) For us, generating the docs/examples so we can log the tokenized outputs
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    optimizer = nlp.initialize(lambda: training_examples)

    # Galileo code
    watch(nlp)
    log_input_examples(training_examples, "training")

    training_losses = []
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
        for itn in range(n_iter):
            batches = minibatch(training_examples, minibatch_size)

            # TODO: could happen more cleanly
            text_ner_logger_config.user_data["epoch"] = itn
            text_ner_logger_config.user_data["split"] = "training"

            for batch in tqdm(batches):
                training_loss = nlp.update(
                    batch,
                    drop=0.5,
                    sgd=optimizer)

                training_losses.append(training_loss["ner"])

    # TODO: need to support the following line
    # nlp('Thank you for your subscription renewal')

    # Evaluation Loop
    test_eval_scores = nlp.evaluate(training_examples, batch_size=minibatch_size)
    assert test_eval_scores["token_acc"] == 1.0

    return training_losses

def test_spacy(cleanup_after_use) -> None:
    nlp = spacy.blank('en')
    nlp.add_pipe('ner', last=True)

    training_data = [
        (
            'what is SEMRUSH PRO? Can you run complex queries ? Can you identify active usage ?',
            {
                'entities': [(21, 32, 'Questions About the Product'),
                             (51, 67, 'Questions About the Product')]
            }),

        ('Thank you for your subscription renewal', {
            'entities': [(19, 39, 'Renew')]
        }),
        (
            'you can upgrade your account for an old price,while you can upgrade your account for $399.95/month',
            {
                'entities': [(8, 28, 'Potential Upsell'), (60, 80, 'Potential Upsell')]
            }),
        ('I like EMSI ordered the pro package', {
            'entities': [(12, 23, 'Product Usage')]
        }),
        ('Here you go, your account is created', {
            'entities': [(0, 11, 'Action item accomplished'),
                         (29, 36, 'Action item accomplished')]
        })
    ]

    test_data = [
        ('Thank you for your subscription renewal', {
            'entities': [(32, 39, 'Renew')]
        }),
    ]

    training_losses = _train_model(training_data, test_data, nlp)

    # loss values gotten from running the script with Galileo Logging turned off
    assert np.allclose(training_losses, [25.500003337860107, 14.20009732246399, 41.103223502635956, 13.864217460155487, 39.7820560336113, 13.508131921291351, 37.757276713848114, 12.961382985115051, 35.641219317913055, 11.587133049964905])

    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()
    # validate_uploaded_data()
    # logger._cleanup()
    # validate_cleanup_data()

    data, emb, prob = load_ner_data_from_local("training")

    #TODO Some assertions

def load_ner_data_from_local(split: str) -> (
    DataFrameLocal, DataFrameLocal, DataFrameLocal):
        """Loads post-logging locally created files.

        Returns: data, emb, and prob vaex dataframes
        """
        split_output_data = {}
        for subdir in SUBDIRS:
            file_path = (
                f"{TEST_PATH}/{split}/{subdir}/{subdir}."
                f"{'arrow' if subdir == 'data' else 'hdf5'}"
            )
            # Ensure files were cleaned up
            data = vaex.open(file_path)
            prob_cols = data.get_column_names(regex="prob*")
            for c in data.get_column_names():
                if c in prob_cols + ["emb"]:
                    assert not np.isnan(data[c].values).any()
                else:
                    vals = data[c].values
                    assert all([i is not None and i != "nan" for i in vals])
            split_output_data[subdir] = data

        return (
            split_output_data["data"],
            split_output_data["emb"],
            split_output_data["prob"],
        )


from dataquality.core.integrations.spacy import SpacyPatchState
from unittest.mock import Mock
from dataquality.core.integrations.spacy import galileo_transition_based_parser_forward


def test_galileo_transition_based_parser_forward():
    nlp = spacy.blank('en')

    text_samples = ["Some text", "Some other text", "Some more text"]

    mock_transition_based_parser_model = Mock()
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
        return mock_parser_step_model, lambda x: print("transition_based_parser backprop fn")

    SpacyPatchState.orig_transition_based_parser_forward = mock_transition_based_parser_forward

    text_ner_logger_config.user_data["epoch"] = 0
    text_ner_logger_config.user_data["split"] = "training"

    galileo_transition_based_parser_forward(mock_transition_based_parser_model, docs, is_train=True)

    assert SpacyPatchState.model_logger.ids == [0, 1, 2]
    assert SpacyPatchState.model_logger.epoch == 0
    assert SpacyPatchState.model_logger.split == "training"
    assert SpacyPatchState.model_logger.probs == [[], [], []]
    assert len(SpacyPatchState.model_logger.emb) == 3
    assert all([embedding.shape == (len(docs[i]), 64) for i, embedding in enumerate(SpacyPatchState.model_logger.emb)])

    assert text_ner_logger_config.user_data["_spacy_state_for_pred"] == [None, None, None]


def test_galileo_parser_step_forward():
    # TODO: fill out cause this may be more tricky to test...
    pass


def test_log_input_examples():
    nlp = spacy.blank('en')
    nlp.add_pipe("ner")

    training_data = [
        (
            'what is SEMRUSH PRO? Can you run complex queries ? Can you identify active usage ?',
            {
                'entities': [(21, 32, 'Questions About the Product'),
                             (51, 67, 'Questions About the Product')]
            }),

        ('Thank you for your subscription renewal', {
            'entities': [(19, 39, 'Renew')]
        }),
        (
            'you can upgrade your account for an old price,while you can upgrade your account for $399.95/month',
            {
                'entities': [(8, 28, 'Potential Upsell'), (60, 80, 'Potential Upsell')]
            }),
        ('I like EMSI ordered the pro package', {
            'entities': [(12, 23, 'Product Usage')]
        }),
        ('Here you go, your account is created', {
            'entities': [(0, 11, 'Action item accomplished'),
                         (29, 36, 'Action item accomplished')]
        })
    ]

    training_examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)

    watch(nlp)
    log_input_examples(training_examples, "training")

    # assert that we added ids to the examples for later joining with model outputs
    assert all([examples.predicted.user_data["id"] == i for i, examples in enumerate(training_examples)])

    logged_data = vaex.open(f"{LOCATION}/input_data.arrow")

    assert logged_data["id"].tolist() == [0, 1, 2, 3, 4]
    assert logged_data["split"].tolist() == ["training"] * len(training_examples)
    assert all([text == training_data[i][0] for i, text in enumerate(logged_data["text"].tolist())])

    # Checks for logged gold spans matching converts from token to char idxs
    logged_token_indices = logged_data["text_token_indices"].tolist()
    for i, (split_plus_id, ents) in enumerate(dataquality.get_model_logger().logger_config.gold_spans.items()):
        assert len(logged_token_indices[i]) % 2 == 0
        ents_as_char_idxs = []
        for ent in ents:
            start_char_idx = logged_token_indices[i][2 * ent[0]]
            end_char_idx = logged_token_indices[i][2 * ent[1] - 1]
            ents_as_char_idxs.append((start_char_idx, end_char_idx, ent[2]))

        assert training_data[i][1]["entities"] == ents_as_char_idxs


def test_watch():
    nlp = spacy.blank('en')
    ner = nlp.add_pipe("ner")

    training_data = [
        (
            'what is SEMRUSH PRO? Can you run complex queries ? Can you identify active usage ?',
            {
                'entities': [(21, 32, 'Questions About the Product'),
                             (51, 67, 'Questions About the Product')]
            }),

        ('Thank you for your subscription renewal', {
            'entities': [(19, 39, 'Renew')]
        }),
        (
            'you can upgrade your account for an old price,while you can upgrade your account for $399.95/month',
            {
                'entities': [(8, 28, 'Potential Upsell'), (60, 80, 'Potential Upsell')]
            }),
        ('I like EMSI ordered the pro package', {
            'entities': [(12, 23, 'Product Usage')]
        }),
        ('Here you go, your account is created', {
            'entities': [(0, 11, 'Action item accomplished'),
                         (29, 36, 'Action item accomplished')]
        })
    ]

    training_examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    nlp.initialize(lambda: training_examples)
    watch(nlp)

    assert text_ner_logger_config.user_data["nlp"] == nlp
    assert dataquality.get_data_logger().logger_config.labels == [
        'B-Questions About the Product',
        'B-Potential Upsell',
        'B-Action item accomplished',
        'B-Renew',
        'B-Product Usage',
        'I-Questions About the Product',
        'I-Potential Upsell',
        'I-Action item accomplished',
        'I-Renew',
        'I-Product Usage',
        'L-Questions About the Product',
        'L-Potential Upsell',
        'L-Action item accomplished',
        'L-Renew',
        'L-Product Usage',
        'U-Questions About the Product',
        'U-Potential Upsell',
        'U-Action item accomplished',
        'U-Renew',
        'U-Product Usage',
        'O'
    ]
    assert dataquality.get_data_logger().logger_config.tagging_schema == "BILOU"

    assert isinstance(nlp.get_pipe("ner"), EntityRecognizer)
    assert isinstance(nlp.get_pipe("ner"), GalileoEntityRecognizer)


# TODO: After this bug is resolved unwatch will be possible: https://github.com/explosion/spaCy/issues/10429
# def test_unwatch():
#     nlp = spacy.blank('en')
#     original_ner = nlp.add_pipe("ner")
#
#     training_data = [
#         (
#             'what is SEMRUSH PRO? Can you run complex queries ? Can you identify active usage ?',
#             {
#                 'entities': [(21, 32, 'Questions About the Product'),
#                              (51, 67, 'Questions About the Product')]
#             }),
#
#         ('Thank you for your subscription renewal', {
#             'entities': [(19, 39, 'Renew')]
#         }),
#         (
#             'you can upgrade your account for an old price,while you can upgrade your account for $399.95/month',
#             {
#                 'entities': [(8, 28, 'Potential Upsell'), (60, 80, 'Potential Upsell')]
#             }),
#         ('I like EMSI ordered the pro package', {
#             'entities': [(12, 23, 'Product Usage')]
#         }),
#         ('Here you go, your account is created', {
#             'entities': [(0, 11, 'Action item accomplished'),
#                          (29, 36, 'Action item accomplished')]
#         })
#     ]
#
#     training_examples = []
#     for text, annotations in training_data:
#         doc = nlp.make_doc(text)
#         training_examples.append(Example.from_dict(doc, annotations))
#
#     nlp.initialize(lambda: training_examples)
#     watch(nlp)
#     unwatch(nlp)
#
#     assert isinstance(nlp.get_pipe("ner"), EntityRecognizer)
#     assert not isinstance(nlp.get_pipe("ner"), GalileoEntityRecognizer)
#
#     assert nlp.get_pipe("ner").moves == original_ner.moves
#     assert nlp.get_pipe("ner").moves == original_ner.model

# TODO: below is a WIP
# # class MockExpectedData
# #
# # class MockEntityRecognizer:
# #     def __init__(self):
# #         self.model = MockTransitionBasedParser()
# #
# # class MockTransitionBasedParser:
# #     def __init__(self):
# #         self._func = lambda *args: MockParserStepModel()
# #
#
# def _calculate_emb_from_text(text: str) -> List[float]:
#     pass
#     # TODO: need to fillout with some deterministic way to go from text to embs
#
# class MockParserStepModel(Model):
#     def __init__(self, docs: List[Doc]):
#         self._func = self.mock_parser_step_model_forward
#
#         tokvecs = []
#         for doc in docs:
#             tokvecs.extend(_calculate_emb_from_text(doc.text))
#         self.tokvecs = np.array(tokvecs)
#
#     def mock_parser_step_model_forward(self, X, *args, **kwargs):
#         """Returns a """


# class MockTransitionBasedParserModel(Model):
#     def __init__(self):
#         self._func = self.mock_transition_based_parser_model_forward
#
#     def mock_transition_based_parser_model_forward(self, X, *args, **kwargs):
#         """Mocks TransitionBasedParser Model's forward func
#
#         Returns a MockParserStepModel and a noop backprop_fn
#         """
#         # TODO: probably need to clean this up
#         return MockParserStepModel(X), lambda d_output: 0.0
#         # TODO: Does noop backprop_fn need to return some number?
#         # I would guess it returns d_input
#
#

