from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import vaex
from spacy.language import Language
from spacy.training import Example
from spacy.util import minibatch
from tqdm import tqdm
from vaex.dataframe import DataFrameLocal

import dataquality
from dataquality.integrations.spacy import log_input_examples, watch
from dataquality.schemas.split import Split
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
    log_input_examples(training_examples, Split.training)
    log_input_examples(training_examples, Split.test)

    training_losses = []
    for itn in range(num_epochs):
        dataquality.set_epoch(itn)
        batches = minibatch(training_examples, MINIBATCH_SZ)

        dataquality.set_split(Split.training)
        for batch in tqdm(batches):
            training_loss = nlp.update(batch, drop=0.5, sgd=optimizer)
            training_losses.append(training_loss["ner"])

        dataquality.set_split(Split.test)
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
