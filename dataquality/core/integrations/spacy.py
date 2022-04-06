from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Generator, List, Tuple, Union

import numpy as np
import thinc
from spacy.language import Language
from spacy.ml.parser_model import ParserStepModel
from spacy.ml.parser_model import precompute_hiddens as State2Vec
from spacy.pipeline._parser_internals.stateclass import StateClass
from spacy.pipeline.ner import EntityRecognizer
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import minibatch
from thinc.api import set_dropout_rate
from wrapt import CallableObjectProxy

import dataquality
from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.loggers.model_logger.text_ner import TextNERModelLogger
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split, conform_split
from dataquality.utils.spacy_integration import (
    convert_spacy_ents_for_doc_to_predictions,
    convert_spacy_ner_logits_to_valid_logits,
    validate_obj,
    validate_spacy_version,
)


def log_input_examples(examples: List[Example], split: Split) -> None:
    """Logs a list of Spacy Examples using the dataquality client"""
    split = conform_split(split)
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
        validate_obj(example, Example, "reference")
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
    validate_spacy_version()
    if not (config.current_project_id and config.current_run_id):
        raise GalileoException(
            "You must initialize dataquality first! "
            "Use dataquality.init(project_name='my_cool_project', "
            "run_name='my_awesome_run', task_type='text_ner')"
        )
    # TODO: Replace with the following in the future
    # if not (config.current_project_id and config.current_run_id):
    #     dataquality.init(task_type=TaskType.text_ner)
    #     warnings.warn("No run initialized with dataquality.init(...). "
    #                   "Creating one with the project name `{}` and run_name `{}`")
    ner = nlp.get_pipe("ner")

    if "O" not in ner.move_names:  # type: ignore
        raise GalileoException(
            "Missing the 'O' tag in the model's moves, are you sure you have"
            "already called 'nlp.begin_training()' or "
            "`nlp.initialize(training_examples)`?"
        )

    text_ner_logger_config.user_data["nlp"] = nlp
    dataquality.set_labels_for_run(ner.move_names)  # type: ignore
    dataquality.set_tagging_schema(TaggingSchema.BILOU)

    nlp.add_pipe("galileo_ner")
    nlp.remove_pipe("ner")
    nlp.rename_pipe("galileo_ner", "ner")


def unwatch(nlp: Language) -> None:
    """Returns spacy nlp Language component to its original unpatched state"""
    raise GalileoException(
        "Coming soon! Discussing here: "
        "https://github.com/explosion/spaCy/discussions/10443"
    )


class GalileoEntityRecognizer(CallableObjectProxy):
    """An EntityRecognizer proxy using the wrapt library.

    We subclass the CallableObjectProxy class because we override the __call__
    method. The wrapped 'ner' object is accessible via "self.__wrapped__".
    """

    def __init__(self, ner: EntityRecognizer):
        super().__init__(ner)
        validate_obj(ner, check_type=EntityRecognizer, has_attr="model")
        if isinstance(ner, GalileoEntityRecognizer):
            raise GalileoException(
                "Seems like your ner component has already "
                "been watched. Make sure to call `watch` "
                "on a fresh `nlp` spacy Language."
            )
        # Assert we are working with the 'ner' component and not 'beam_ner'
        if self.cfg["beam_width"] != 1:
            raise GalileoException(
                f"Your EntityRecognizer's beam width is set to "
                f"{self.cfg['beam_width']}. Galileo currently "
                f"expects a beam width of 1 (the 'ner' default)."
            )

        ner.model = GalileoTransitionBasedParserModel(ner.model)

    def greedy_parse(self, docs: List[Doc], drop: float = 0.0) -> List:
        """Python-land implementation of the greedy_parse method in the ner component

        Transcribes the greedy_parse method in Parser to python. This allows us to call
        the Thinc model's forward function, which we patch, rather than the ridiculous
        C-math copy of it. See spacy.ml.parser_model.predict_states for the C-math fn
        that gets called eventually to do the predictions by the EntitiyRecognizer's
        greedy_parse method
        https://github.com/explosion/spaCy/blob/ed561cf428494c2b7a6790cd4b91b5326102b59d/spacy/ml/parser_model.pyx#L93
        """
        self._ensure_labels_are_added(docs)
        set_dropout_rate(self.model, drop)
        batch = self.moves.init_batch(docs)
        step_model = self.model.predict(docs)

        states = list(batch)
        non_final_states = states.copy()
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
        # Ignore mypy because Doc implements __len__ but Spacy doesn't realize
        # https://github.com/explosion/spaCy/blob/master/spacy/tokens/doc.pyx#L489
        if not any(filter(lambda doc: len(doc), docs)):  # type: ignore
            result = self.moves.init_batch(docs)
            return result

        # Galileo's version of greedy_parse (ner's method that handles prediction)
        return self.greedy_parse(docs, drop=0.0)

    def pipe(self, docs: Union[Doc, List[Doc]], batch_size: int = 256) -> Generator:
        """Copy of spacy EntityRecognizer's pipe

        This method copies spacy's EntityRecognizer's pipe defined in the Parser
        superclass, but notice that in calling predict from this scope it calls the
        GalileoEntityRecognizer defined above predict method.
        """
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
                self.get_error_handler()(self.name, self, batch_in_order, e)

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
            return error_handler(self.name, self, [doc], e)


@Language.factory("galileo_ner")
def create_galileo_ner(nlp: Language, name: str) -> GalileoEntityRecognizer:
    return GalileoEntityRecognizer(nlp.get_pipe("ner"))


class ThincModelWrapper(CallableObjectProxy):
    """A Thinc Model obj wrapper using the wrapt library.

    wrapt primer: https://wrapt.readthedocs.io/en/latest/wrappers.html
    Wrapping an obj allows us to manipulate/watch its behavior. Compared to
    overwriting some of its functions, using an obj wrapper allows us to also
    maintain some state in between wrapped function calls. This also gives us a
    method of extending cython def classes to make them watchable. For example
    with the spacy EntityRecognizer we can wrap the class and create our own forward
    pass that calls methods we are able to log probabilities from. Successfully
    wrapping an obj relies on the ability to overwrite the ref to that obj with
    the wrapped obj, so if the obj references is immutable the better approach
    would be to overwrite the obj's behavior directly.
    wrapped object are accessible via 'self.__wrapped'

    This Thinc Model wrapper patches the 'self._func' method (the forward pass).
    """

    def __init__(self, model: thinc.model.Model):
        super().__init__(model)
        validate_obj(model, check_type=thinc.model.Model, has_attr="_func")

        self._self_orig_forward = model._func
        # https://github.com/python/mypy/issues/2427
        model._func = self._self_forward  # type: ignore

    def _self_forward(
        self, model: thinc.model.Model, X: Any, is_train: bool
    ) -> Tuple[Any, Any]:
        """Overwrite this to patch the Thinc model's forward fn"""


class GalileoTransitionBasedParserModel(ThincModelWrapper):
    expected_model_name: str = "parser_model"

    def __init__(self, model: thinc.model.Model):
        super().__init__(model)
        if not model.name == GalileoTransitionBasedParserModel.expected_model_name:
            raise GalileoException(
                "Expected the TransitionBasedParser Thinc Model to "
                f"be called {GalileoTransitionBasedParserModel.expected_model_name}. "
                f"Instead received {model.name}. This may indicate "
                f"that the spacy architecture has changed and is no "
                f"longer compatible with this Galileo integration."
            )
        assert isinstance(model, thinc.model.Model)
        assert not isinstance(model, GalileoTransitionBasedParserModel)

    def _self_forward(
        self, model: thinc.model.Model, X: Any, is_train: bool
    ) -> Tuple[ParserStepModel, Callable]:
        parser_step_model, backprop_fn = self._self_orig_forward(
            model, X, is_train=is_train
        )

        if not all(["id" in doc.user_data for doc in X]):
            raise GalileoException(
                "One of your model's docs is missing a galileo generated "
                "id. Did you first log your docs/examples with us using, "
                "for example, "
                "`log_input_examples(training_examples, split='training')`? "
                "Make sure to then continue using 'training_examples'"
            )

        model_logger = TextNERModelLogger()
        helper_data = model_logger.log_helper_data
        helper_data["logits"] = {doc.user_data["id"]: [None] * len(doc) for doc in X}
        helper_data["embs"] = {doc.user_data["id"]: [None] * len(doc) for doc in X}
        helper_data["spacy_states"] = defaultdict(list)
        helper_data["spacy_states_end_idxs"] = defaultdict(list)
        helper_data["already_logged"] = False
        return GalileoParserStepModel(parser_step_model, model_logger), backprop_fn


class GalileoParserStepModel(ThincModelWrapper):
    expected_model_name: str = "parser_step_model"

    def __init__(self, model: ParserStepModel, model_logger: TextNERModelLogger):
        super().__init__(model)
        if not model.name == GalileoParserStepModel.expected_model_name:
            raise GalileoException(
                "Expected the ParserStepModel Thinc Model "
                f"to be called {GalileoParserStepModel.expected_model_name}. "
                f"Instead received {model.name}. This may indicate "
                f"that the spacy architecture has changed and is no "
                f"longer compatible with this Galileo integration."
            )
        assert isinstance(model, ParserStepModel)
        assert not isinstance(
            model, GalileoParserStepModel
        ), "trying to patch an already patched model"

        self._self_model_logger = model_logger
        # state2vec is the embedding model/layer
        model.state2vec = GalileoState2Vec(model.state2vec, self._self_model_logger)

    def _self_get_state_end_idx(self, state_idx: int, states: List[StateClass]) -> int:
        state = states[state_idx]
        state_id = state.doc.user_data["id"]
        next_state = states[state_idx + 1] if state_idx + 1 < len(states) else None
        next_state_id = next_state.doc.user_data["id"] if next_state else None

        # If the next state is from the same doc, return its start
        if next_state and next_state_id == state_id:
            return next_state.queue[0]
        # Otherwise, we're at the end of this doc, so return its length
        return len(state.doc)

    def _self_initialize_helper_data(self, states: List[StateClass]) -> None:
        # TODO: might need to assert that states has a particular structure
        helper_data = self._self_model_logger.log_helper_data
        for state_idx, state in enumerate(states):
            doc_id = state.doc.user_data["id"]
            state_end_idx = self._self_get_state_end_idx(state_idx, states)
            helper_data["spacy_states"][doc_id].append(None)
            helper_data["spacy_states_end_idxs"][doc_id].append(state_end_idx)

    def _self_is_helper_data_filled(self) -> bool:
        """Should be enough to check that logits is filled"""
        helper_data = self._self_model_logger.log_helper_data
        is_doc_filled = []
        for doc_id, doc_logits in helper_data["logits"].items():
            doc_embs = helper_data["embs"][doc_id]
            is_doc_filled.append(
                all(
                    [
                        token_logits is not None and token_embs is not None
                        for token_logits, token_embs in zip(doc_logits, doc_embs)
                    ]
                )
            )
        return all(is_doc_filled)

    def _self_fill_helper_data(
        self, states: List[StateClass], scores: np.ndarray
    ) -> None:
        helper_data = self._self_model_logger.log_helper_data
        logits = scores[..., 1:]  # Throw out the -U token
        ner = text_ner_logger_config.user_data["nlp"].get_pipe("ner")
        for state_idx, state in enumerate(states):
            doc_id = state.doc.user_data["id"]
            state_cur_token_idx = state.queue[0]

            helper_data["logits"][doc_id][state_cur_token_idx] = logits[state_idx]

            for chunk_idx, state_end_idx in enumerate(
                helper_data["spacy_states_end_idxs"][doc_id]
            ):
                if state_cur_token_idx < state_end_idx:
                    # check if this is a pre final state
                    if state_cur_token_idx == state_end_idx - 1:
                        final_state = state.copy()
                        ner.transition_states(
                            [final_state], scores[state_idx : state_idx + 1, :]
                        )
                        helper_data["spacy_states"][doc_id][chunk_idx] = final_state
                        helper_data["spacy_states_end_idxs"][doc_id][chunk_idx] += 1
                    break

    def _self_set_annotations(self, docs: Dict[int, Doc]) -> None:
        helper_data = self._self_model_logger.log_helper_data
        ner = text_ner_logger_config.user_data["nlp"].get_pipe("ner")
        for doc_id, doc in docs.items():
            states_for_doc = helper_data["spacy_states"][doc_id]
            ner.set_annotations([doc] * len(states_for_doc), states_for_doc)

    def _self_get_valid_logits(self, docs: Dict[int, Doc]) -> DefaultDict:
        helper_data = self._self_model_logger.log_helper_data
        docs_predictions = convert_spacy_ents_for_doc_to_predictions(
            docs, self._self_model_logger.logger_config.labels
        )
        docs_valid_logits = defaultdict(list)
        for doc_id, doc_logits in helper_data["logits"].items():
            for token_idx, token_logits in enumerate(doc_logits):
                valid_token_logits = convert_spacy_ner_logits_to_valid_logits(
                    token_logits, docs_predictions[doc_id][token_idx]
                )
                docs_valid_logits[doc_id].append(valid_token_logits)
        return docs_valid_logits

    def _self_populate_model_logger(self, docs_valid_logits: DefaultDict) -> None:
        model_logger = self._self_model_logger
        helper_data = model_logger.log_helper_data
        model_logger.ids = list(model_logger.ids)  # for mypy's sake
        for doc_id, doc_valid_logits in docs_valid_logits.items():
            model_logger.logits.append(np.array(doc_valid_logits))
            doc_embs = helper_data["embs"][doc_id]
            model_logger.emb.append(np.array(doc_embs))
            model_logger.ids.append(doc_id)

    def _self_get_docs_copy(self) -> Dict[int, Doc]:
        spacy_states = self._self_model_logger.log_helper_data["spacy_states"]
        return {
            doc_id: states_for_doc[0].doc.copy()
            for doc_id, states_for_doc in spacy_states.items()
        }

    def _self_forward(
        self,
        parser_step_model: ParserStepModel,
        states: List[StateClass],
        is_train: bool,
    ) -> Tuple[np.ndarray, Callable]:
        """Patches the ParserStepModel's forward func

        The forward function is called with a token from each chunked doc in the
        sample. So the max num times its called is equal to the max num tokens in any
        doc's chunk.
        """
        helper_data = self._self_model_logger.log_helper_data
        if not helper_data["spacy_states"]:
            self._self_initialize_helper_data(states)

        parser_step_model.state2vec._self_states = states
        # Let spacy run its original forward on the data
        scores, backprop_fn = self._self_orig_forward(
            parser_step_model, states, is_train
        )

        self._self_fill_helper_data(states, scores.copy())

        if not helper_data["already_logged"] and self._self_is_helper_data_filled():
            docs_copy = self._self_get_docs_copy()
            self._self_set_annotations(docs_copy)
            docs_valid_logits = self._self_get_valid_logits(docs_copy)

            # Now that we have valid logits and embs, fill out the log
            self._self_populate_model_logger(docs_valid_logits)
            self._self_model_logger.log()
            helper_data["already_logged"] = True

        return scores, backprop_fn


class GalileoState2Vec(CallableObjectProxy):
    def __init__(self, model: State2Vec, model_logger: TextNERModelLogger):
        super().__init__(model)
        validate_obj(model, State2Vec, "__call__")
        assert not isinstance(
            model, GalileoState2Vec
        ), "trying to patch an already patched model"
        self._self_model_logger = model_logger
        self._self_states: List[StateClass] = []  # Set externally by developer

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Overwrites forward to capture embeddings and add to model_logger.

        Note that self._self_states needs to be set by developer before this is called.
        """
        embeddings, embeddings_bp = self.__wrapped__(*args, **kwargs)

        helper_data = self._self_model_logger.log_helper_data
        for state_idx, state in enumerate(self._self_states):
            doc_id = state.doc.user_data["id"]
            state_cur_token_idx = state.queue[0]
            helper_data["embs"][doc_id][state_cur_token_idx] = embeddings[
                state_idx
            ].copy()

        return embeddings, embeddings_bp
