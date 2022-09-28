from unittest.mock import MagicMock, patch
from typing import Callable
import copy

import pytest

import dataquality as dq
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from tests.utils.mock_request import (
    EXISTING_PROJECT,
    EXISTING_RUN,
    mocked_create_project_run,
    mocked_get_project_run,
    mocked_login,
    mocked_login_requests,
    mocked_missing_project_run,
    mocked_missing_run,
)
from dataquality import config

from datasets import Dataset, load_metric
from .utils.hf_datasets_mock import mock_dataset
from dataquality.integrations.trainer_callback import DQCallback
from dataquality.exceptions import GalileoException


tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-distilbert")
model = AutoModelForSequenceClassification.from_pretrained(
    "hf-internal-testing/tiny-random-distilbert"
)
metric = load_metric("accuracy")

def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["text"], padding="max_length", max_length=201, truncation=True
    )

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax( axis=1)
    return metric.compute(predictions=predictions, references=labels)

encoded_train_dataset = mock_dataset.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

encoded_test_dataset = mock_dataset.map(
    lambda x: preprocess_function(x, tokenizer), batched=True
)

# Training arguments and training part
metric_name = "accuracy"
batch_size = 16

args_default = TrainingArguments(
    f"finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
   
)


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end_to_end_without_callback(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Training on a dataset"""    
    args =  copy.deepcopy(args_default)

   
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
       
    )

    trainer.train()


@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_end_to_end_with_callback(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests creating a new project and run"""
    global encoded_train_dataset,encoded_test_dataset
    dq.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id
    # 🔭🌕 Galileo logging
    dq.init(task_type="text_classification")
    logger = dq.get_data_logger()
    args =  copy.deepcopy(args_default)
    # 🔭🌕 Galileo logging
    dqcallback = DQCallback()
    #TODO: double check the labels 
    encoded_train_dataset = dqcallback.add_idx_col(encoded_train_dataset)
    encoded_test_dataset = dqcallback.add_idx_col(encoded_test_dataset)
    args.remove_unused_columns = False
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        # 🔭🌕 Galileo logging
        callbacks=[dqcallback],
        data_collator=dqcallback.collate_fn
    )

    trainer.train()
    # TODO: test with finish
    dq.finish()

@patch("requests.post", side_effect=mocked_create_project_run)
@patch("requests.get", side_effect=mocked_get_project_run)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_remove_unused_columns(
    mock_valid_user: MagicMock,
    mock_requests_get: MagicMock,
    mock_requests_post: MagicMock,
    set_test_config: Callable,
) -> None:
    """Base case: Tests remove unused columns not set to false"""
    dq.init(task_type="text_classification")
    assert config.current_run_id
    assert config.current_project_id
    dq.init(task_type="text_classification")
    logger = dq.get_data_logger()
    args =  copy.deepcopy(args_default)

    callback_args = {"embedding_layer":"str|layer?"}
    #optional, default search layer is there
    #log found layer xyz
    dqcallback = DQCallback(**callback_args)
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[dqcallback]
    )
    with pytest.raises(GalileoException) as e:
            trainer.train()
    assert e.value.args[0] == (
        "TrainerArgument remove_unused_columns must be false"
    )
    

