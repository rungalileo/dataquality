from typing import Callable, Generator, List, Tuple, Union

import numpy as np
from scipy.special import softmax
from spacy.language import Language
from spacy.ml.parser_model import ParserStepModel
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.pipeline.ner import EntityRecognizer
from spacy.tokens import Doc
from spacy.training import Example, offsets_to_biluo_tags
from spacy.util import minibatch
from thinc.api import set_dropout_rate
from thinc.model import Model
from wrapt import CallableObjectProxy

import dataquality
from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.ner import TaggingSchema


class SpacyPatchState:
    """A state class to keep track of state without polluting the namespace"""

    model_logger: TextNERModelLogger
    orig_parser_step_forward: Callable
    orig_transition_based_parser_forward: Callable


class GalileoEntityRecognizer(CallableObjectProxy):
    """An EntityRecognizer proxy using the wrapt library.

    We subclass the CallableObjectProxy class because we override the __call__
    method. The wrapped 'ner' object is accessible via "self.__wrapped__".
    """

    def __init__(self, ner: EntityRecognizer):
        super(GalileoEntityRecognizer, self).__init__(ner)

        # TODO: is this necessary if we do typing above
        if not isinstance(ner, EntityRecognizer):
            raise GalileoException(
                f"Expected an EntityRecognizer component. Received {str(type(ner))}"
            )

        if not hasattr(ner, "model"):
            raise GalileoException(
                "Your ner EntityRecognizer must have a Thinc model attribute under "
                "model"
            )

        patch_transition_based_parser_forward(ner.model)

    def greedy_parse(self, docs: List[Doc], drop: float = 0.0) -> List:
        """Python-land implementation of the greedy_parse method in the ner component

        Transcribes the greedy_parse method in Parser to python. This allows us to call
        the Thinc model's forward function, which we patch, rather than the ridiculous
        C-math copy of it.
        """
        self._ensure_labels_are_added(docs)
        set_dropout_rate(self.model, drop)
        batch = self.moves.init_batch(docs)
        step_model = self.model.predict(docs)

        states = list(batch)
        non_final_states = [state for state in states]
        while non_final_states:
            scores = step_model.predict(non_final_states)
            self.transition_states(non_final_states, scores)  # updates non_final_states
            non_final_states = [
                state for state in non_final_states if not state.is_final()
            ]
        return states

    def predict(self, docs: Union[Doc, List[Doc]]) -> List:
        """Copied from the EntityRecognizer's predict, but calls our greedy_parse"""
        if isinstance(docs, Doc):
            docs = [docs]
        if not any(len(doc) for doc in docs):
            result = self.moves.init_batch(docs)
            return result

        # Assert we are working with the 'ner' component and not 'beam_ner'
        assert self.cfg["beam_width"] == 1

        # Galileo's version of greedy_parse (ner's method that handles prediction)
        return self.greedy_parse(docs, drop=0.0)

    def pipe(self, docs: Union[Doc, List[Doc]], batch_size: int = 256) -> Generator:
        """Copy of spacy EntityRecognizer's pipe

        This method copies spacy's EntityRecognizer's pipe defined in the Parser
        superclass, but notice that in calling predict from this scope it calls the
        GalileoEntityRecognizer defined above predict method.
        """
        error_handler = self.get_error_handler()
        for batch in minibatch(docs, size=batch_size):
            batch_in_order = list(batch)
            try:
                by_length = sorted(batch, key=lambda doc: len(doc))
                for subbatch in minibatch(by_length, size=max(batch_size // 4, 2)):
                    subbatch = list(subbatch)
                    parse_states = self.predict(subbatch)
                    self.set_annotations(subbatch, parse_states)
                yield from batch_in_order
            except Exception as e:
                error_handler(self.name, self, batch_in_order, e)

    def __call__(self, doc: Doc) -> Doc:
        """Copy of TrainablePipe's __call__

        As the EntityRecognizer inherits from TrainablePipe it also inherits this
        method which calls 'predict' when the language is used as a callable.
        We need to overwrite to call our greedy_parse eventually

        e.g: nlp("some text") # calls under the hood __call__ for every pipe component
        """
        error_handler = self.get_error_handler()
        try:
            scores = self.predict([doc])
            self.set_annotations([doc], scores)
            return doc
        except Exception as e:
            error_handler(self.name, self, [doc], e)


@Language.factory("galileo_ner")
def create_galileo_ner(nlp: Language, name: str) -> GalileoEntityRecognizer:
    return GalileoEntityRecognizer(nlp.get_pipe("ner"))


def log_input_examples(examples: List[Example], split: str) -> None:
    """Logs a list of Spacy Examples using the dataquality client"""
    if not dataquality.get_data_logger().logger_config.labels:
        raise GalileoException(
            "Galileo does not have any logged labels. Did you forget "
            "to call watch(nlp) before log_input_examples(...)?"
        )
    text = []
    text_token_indices = []
    gold_spans = []
    ids = []
    for i, example in enumerate(examples):
        # For the most part reference has all the information we want to log
        data = example.reference
        # but predicted is the Doc that will be passed along to the spacy models, and
        # crucially holds the "id" user_data we attach
        text.append(data.text)
        text_token_indices.append(
            [(token.idx, token.idx + len(token)) for token in data]
        )
        gold_spans.append(
            [
                {"start": ent.start_char, "end": ent.end_char, "label": ent.label_}
                for ent in data.ents
            ]
        )
        # We add ids to the doc.user_data to be along for the ride through spacy
        # The predicted doc is the one that the model will see
        example.predicted.user_data["id"] = i
        ids.append(i)
    dataquality.log_input_data(
        text=text,
        text_token_indices=text_token_indices,
        gold_spans=gold_spans,
        ids=ids,
        split=split,
    )


def watch(nlp: Language) -> None:
    """Stores the nlp object before calling watch on the ner component within it

    We need access to the nlp object so that during training we can capture the
    model's predictions over the raw text by running nlp("user's text") and looking
    at the results

    :param nlp: The spacy nlp Language component.
    :return: None
    """
    assert (
        config.current_project_id and config.current_run_id
    ), "You must initialize dataquality first! Use dataquality.login()"

    ner = nlp.get_pipe("ner")

    if "O" not in ner.move_names:
        raise GalileoException(
            "Missing the 'O' tag in the model's moves, are you sure you have"
            "already called 'nlp.begin_training()'?"
        )

    text_ner_logger_config.user_data["nlp"] = nlp
    dataquality.set_labels_for_run(ner.move_names)
    dataquality.set_tagging_schema(TaggingSchema.BILOU)

    nlp.add_pipe("galileo_ner")
    nlp.remove_pipe("ner")
    nlp.rename_pipe("galileo_ner", "ner")


def unwatch(nlp: Language) -> None:
    """Returns spacy nlp Language component to its original unpatched state"""
    # the following code may work after the following spacy bug is addressed
    # https://github.com/explosion/spaCy/issues/10429
    # for now we pass
    # old_moves = nlp.get_pipe("ner").move_names
    # old_model = nlp.get_pipe("ner").model
    # nlp.remove_pipe("ner")
    # nlp.add_pipe("ner", config={"moves": old_moves, "model": model})


def _convert_spacy_ner_logits_to_probs(logits: np.ndarray, pred: int) -> List[float]:
    """Converts ParserStepModel per token logits to probabilities.

    Not all logits outputted by the spacy model are valid probabilities, for this reason
    spacy will ignore potential actions even if they might've had the largest prob mass.
    To accoutn for this, we first sort the logits for each token and then zero out
    all logits larger than the predicted logit (as these must've been ignored by spacy
    or else they would've become the prediction). Finally we take the softmax to convert
    them to probabilities.
    """
    # Sort in descending order
    argsorted_sample_logits = np.flip(np.argsort(logits))

    # Get all logit indices where pred_logit > logit
    # These are 'valid' because spacy ignored all logits > pred_logit
    # as it they were determined to not be possible given the current state.
    valid_logit_indices = argsorted_sample_logits[
        np.where(argsorted_sample_logits == pred)[0][0] :
    ]

    valid_probs = softmax(logits[valid_logit_indices])

    # non valid_logit_indices should be set to 0
    probs = np.zeros(logits.shape)
    probs[valid_logit_indices] = valid_probs

    return probs.tolist()


def _convert_spacy_ents_for_doc_to_predictions(docs: List[Doc]) -> List[List[int]]:
    """Converts spacy's representation of ner spans to their per token predictions.

    Uses some spacy utility code to convert from start/end/label representation to the
    BILUO per token corresponding tagging scheme.

    """
    prediction_indices = []
    for doc in docs:
        # perhaps there are utility functions to help do this step
        pred_output = offsets_to_biluo_tags(
            doc, [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        )
        pred_output_ind = [
            SpacyPatchState.model_logger.logger_config.labels.index(tok_pred)
            for tok_pred in pred_output
        ]
        prediction_indices.append(pred_output_ind)
    return prediction_indices


def galileo_parser_step_forward(
    parser_step_model: Model, X: List[StateClass], is_train: bool
) -> Tuple[np.ndarray, Callable]:
    scores, backprop_fn = SpacyPatchState.orig_parser_step_forward(
        parser_step_model, X, is_train
    )
    logits = scores[..., 1:]  # Throw out the -U token

    assert logits.shape == (len(X), parser_step_model.nO - 1)  # type: ignore
    # Logits are returned for all docs, one token at a time. So as we continue
    # to call parser_step_forward some docs will have finished and not have
    # any remaining tokens to make predictions on. We assume that the
    # outputted scores match in order to the inputted X: List[StateClass]
    # eg. X == [StateClass_0, StateClass_1] then scores == [2, 22]
    model_logger = SpacyPatchState.model_logger  # for readability
    model_logger_idxs = []

    for i, state in enumerate(X):
        # In case the order of X is different than the original X of docs
        # Assumes passed in data has the "id" user_data appended, which we
        # automatically append with our log_training call.
        model_logger_idx = model_logger.ids.index(state.doc.user_data["id"])
        model_logger_idxs.append(model_logger_idx)

        text_ner_logger_config.user_data["_spacy_state_for_pred"][
            model_logger_idx
        ] = state.copy()

        # Math is easier on a sample_logits level
        sample_logits = logits[i]

        model_logger.probs[model_logger_idx].append(sample_logits)

    ner = text_ner_logger_config.user_data["nlp"].get_pipe("ner")
    ner.transition_states(
        [
            text_ner_logger_config.user_data["_spacy_state_for_pred"][idx]
            for idx in model_logger_idxs
        ],
        scores,
    )

    # if we have are at the end of the batch
    if all(
        [
            len(model_logger.probs[i]) == len(model_logger.emb[i])
            for i in range(len(model_logger.ids))
        ]
    ):
        # Do the final transition to be able to use spacy to get predictions
        ner = text_ner_logger_config.user_data["nlp"].get_pipe("ner")
        docs_copy = [
            state.doc.copy()
            for state in text_ner_logger_config.user_data["_spacy_state_for_pred"]
        ]

        ner.set_annotations(
            docs_copy, text_ner_logger_config.user_data["_spacy_state_for_pred"]
        )

        predictions_for_docs = _convert_spacy_ents_for_doc_to_predictions(docs_copy)
        probabilities_for_docs: List[List] = [
            [] for _ in range(len(predictions_for_docs))
        ]

        for doc_idx, logits_for_doc in enumerate(model_logger.probs):
            for token_idx, token_logits in enumerate(logits_for_doc):
                probs = _convert_spacy_ner_logits_to_probs(
                    token_logits, predictions_for_docs[doc_idx][token_idx]
                )
                probabilities_for_docs[doc_idx].append(probs)

        for i in range(len(probabilities_for_docs)):
            probabilities_for_docs[i] = np.array(probabilities_for_docs[i])
        model_logger.probs = probabilities_for_docs
        model_logger.log()

    return scores, backprop_fn


def galileo_transition_based_parser_forward(
    transition_based_parser_model: Model, X: List[Doc], is_train: bool
) -> Tuple[ParserStepModel, Callable]:
    (
        parser_step_model,
        backprop_fn,
    ) = SpacyPatchState.orig_transition_based_parser_forward(
        transition_based_parser_model, X, is_train
    )

    model_logger = SpacyPatchState.model_logger = TextNERModelLogger()

    if not all(["id" in doc.user_data for doc in X]):
        raise GalileoException(
            "One of your model's docs is missing a galileo generated "
            "id. Did you first log your docs/examples with us?"
        )

    model_logger.ids = [doc.user_data["id"] for doc in X]
    model_logger.epoch = text_ner_logger_config.user_data["epoch"]
    model_logger.split = text_ner_logger_config.user_data["split"]
    model_logger.probs = [[] for _ in range(len(X))]

    text_ner_logger_config.user_data["_spacy_state_for_pred"] = [None] * len(X)

    assert parser_step_model.tokvecs.shape == (sum([len(doc) for doc in X]), 64)
    # Embeddings for all docs are concatenated together, so
    # need to split a [46, 64] matrix into [[8, 64], [17, 64], [21, 64]]
    # given that len(doc_0) == 8, len(doc_1) == 17, len(doc_2) == 21
    # Crucially, we assume the order of tokvecs == order of X
    # This is also called a "ragged" array?
    tokens_already_seen = 0
    for doc in X:
        model_logger.emb.append(
            parser_step_model.tokvecs[
                tokens_already_seen : tokens_already_seen + len(doc)
            ]
        )

    return patch_parser_step_forward(parser_step_model), backprop_fn


def patch_parser_step_forward(parser_step_model: ParserStepModel) -> ParserStepModel:
    """A basic way to patch the _func forward method of ParserStepModel model"""
    SpacyPatchState.orig_parser_step_forward = parser_step_model._func
    parser_step_model._func = galileo_parser_step_forward
    return parser_step_model


def patch_transition_based_parser_forward(transition_based_parser_model: Model) -> None:
    """A basic way to patch the _func forward method of the TransitionBasedParser"""

    if not isinstance(transition_based_parser_model, Model):
        parser_type = str(type(transition_based_parser_model))
        raise GalileoException(f"Expected a Thinc model. Received {parser_type}")
    if not hasattr(transition_based_parser_model, "_func"):
        raise GalileoException(
            "Expected your TransitionBasedParser Thinc model to have a "
            "forward function at _func"
        )

    # If the name changes in future Spacy versions we should assume the model also
    # changed and investigate if our integration still works.
    assert transition_based_parser_model.name == "parser_model"

    SpacyPatchState.orig_transition_based_parser_forward = (
        transition_based_parser_model._func
    )
    # https://github.com/python/mypy/issues/2427
    galileo_transition = galileo_transition_based_parser_forward
    transition_based_parser_model._func = galileo_transition  # type: ignore
