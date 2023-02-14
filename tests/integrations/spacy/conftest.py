from typing import Callable, Generator, List

import numpy as np
import pytest
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.training import Example

from dataquality.integrations.spacy import watch
from dataquality.schemas.task_type import TaskType
from tests.test_utils.spacy_integration_constants import NER_TRAINING_DATA
from tests.test_utils.spacy_integration_constants_inference import NER_INFERENCE_DATA

spacy.util.fix_random_seed()


@pytest.fixture(scope="function")
def nlp(set_test_config: Callable, cleanup_after_use: Generator) -> Language:
    set_test_config(task_type=TaskType.text_ner)
    np.random.seed(42)
    spacy.util.fix_random_seed(42)
    nlp = spacy.blank("en")
    nlp.add_pipe("ner")
    return nlp


@pytest.fixture(scope="function")
def training_examples(nlp: Language) -> List[Example]:
    training_examples = []
    for text, annotations in NER_TRAINING_DATA:
        doc = nlp.make_doc(text)
        training_examples.append(Example.from_dict(doc, annotations))

    return training_examples


@pytest.fixture(scope="function")
def nlp_init(nlp: Language, training_examples: List[Example]) -> Language:
    nlp.initialize(lambda: training_examples)
    return nlp


@pytest.fixture(scope="function")
def nlp_watch(nlp_init: Language) -> Language:
    watch(nlp_init)
    return nlp_init


@pytest.fixture(scope="function")
def inference_docs(nlp: Language) -> List[Doc]:
    return [nlp.make_doc(text) for text in NER_INFERENCE_DATA]
