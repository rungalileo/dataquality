TODO: Add tests for spacy start

from typing import Callable, Generator
from unittest.mock import MagicMock, patch
from dataquality.schemas.split import Split

import spacy
import vaex
from spacy.training import Example
from spacy.util import minibatch

import dataquality as dq
from dataquality import DataQuality
from dataquality.clients.api import ApiClient
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import DEFAULT_PROJECT_ID, DEFAULT_RUN_ID, LOCATION
from tests.test_utils.spacy_integration import train_model
from tests.test_utils.spacy_integration_constants import NER_TRAINING_DATA


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_spacy_txt(
    mock_valid_user: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    set_test_config: Callable,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    cleanup_after_use: Generator,
) -> None:
    mock_get_project_by_name.return_value = {"id": DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")
    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))
    optimizer = nlp.initialize(lambda: training_examples)
    num_epochs = 1
    MINIBATCH_SZ = 3
    with DataQuality(
        nlp,
        train_data=training_examples,
        task="text_ner",
    ) as dq:
        training_losses = []
        for itn in range(num_epochs):
            dq.set_epoch(itn)
            batches = minibatch(training_examples, MINIBATCH_SZ)
            dq.set_split(Split.training)
            for batch in batches:
                training_loss = nlp.update(batch, drop=0.5, sgd=optimizer)
                training_losses.append(training_loss["ner"])

            dq.set_split(Split.test)
            nlp.evaluate(training_examples, batch_size=MINIBATCH_SZ)
        ThreadPoolManager.wait_for_threads()
        assert len(vaex.open(f"{LOCATION}/training/0/*.hdf5"))
