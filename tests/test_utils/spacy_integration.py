import hashlib
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import vaex
from spacy.language import Doc, Language
from spacy.training import Example
from spacy.util import minibatch
from thinc.model import Model
from tqdm import tqdm
from vaex.dataframe import DataFrameLocal

import dataquality
from dataquality.integrations.spacy import log_input_examples, watch
from tests.conftest import SUBDIRS, TEST_PATH

MINIBATCH_SZ = 3


def train_model(
    nlp: Language,
    training_examples: List[Example],
    num_epochs: int = 5,
) -> None:
    """Trains a model and logs the data, embeddings, and probabilities.

    Args:
        nlp (Language): The spacy model, uninitialized and unwatched
        training_examples (List[Example]): The training examples.
        num_epochs (int, optional): The number of epochs to train for. Defaults to 5.
    """
    optimizer = nlp.initialize(lambda: training_examples)

    # Galileo code
    watch(nlp)
    log_input_examples(training_examples, "training")
    log_input_examples(training_examples, "test")

    training_losses = []
    for itn in range(num_epochs):
        dataquality.set_epoch(itn)
        batches = minibatch(training_examples, MINIBATCH_SZ)

        dataquality.set_split("training")
        for batch in tqdm(batches):
            training_loss = nlp.update(batch, drop=0.5, sgd=optimizer)
            training_losses.append(training_loss["ner"])

        dataquality.set_split("test")
        nlp.evaluate(training_examples, batch_size=MINIBATCH_SZ)

    # TODO: need to support the following line for inference
    # nlp('Thank you for your subscription renewal')

    # Evaluation Loop
    # TODO: Should change this with an actual evaluation loop

    logger = dataquality.get_data_logger(dataquality.config.task_type)
    logger.upload()

    return training_losses


def load_ner_data_from_local(
    split: str,
    inf_name_or_epoch: Union[str, int] = "",
) -> Tuple[pd.DataFrame, DataFrameLocal, DataFrameLocal]:
    """Loads post-logging locally created files.

    Returns: data, emb, and prob vaex dataframes
    """
    split_output_data = {}
    for subdir in SUBDIRS:
        file_path = (
            f"{TEST_PATH}/{split}/{inf_name_or_epoch}/{subdir}/{subdir}."
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
        split_output_data["data"].to_pandas_df(),
        split_output_data["emb"],  # can't convert nested arrays to pandas df
        split_output_data["prob"],  # can't convert nested arrays to pandas df
    )


def _calculate_emb_from_doc(doc: Doc) -> np.ndarray:
    hash_seed = int(hashlib.sha256(doc.text.encode("utf-8")).hexdigest(), 16) % 10**8
    np.random.seed(hash_seed)
    return np.random.rand(len(doc), 64)


class MockParserStepModel(Model):
    def __init__(self, docs: List[Doc]) -> None:
        self._func = self.mock_parser_step_model_forward

        tokvecs = []
        for doc in docs:
            tokvecs.extend(_calculate_emb_from_doc(doc))
        self.tokvecs = np.array(tokvecs)

    def mock_parser_step_model_forward(self, X, *args, **kwargs) -> None:
        """Returns logits that should construct an actual state"""


class MockTransitionBasedParserModel(Model):
    def __init__(self) -> None:
        self._func = self.mock_transition_based_parser_model_forward

    def mock_transition_based_parser_model_forward(self, X, *args, **kwargs) -> None:
        """Mocks TransitionBasedParser Model's forward func

        Returns a MockParserStepModel and a noop backprop_fn
        """
        # TODO: probably need to clean this up
        return MockParserStepModel(X), lambda d_output: 0.0
        # TODO: Does noop backprop_fn need to return some number?
        # I would guess it returns d_input
