from functools import partial
from typing import Dict, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
    Trainer,
)

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.seq2seq.hf import watch
from dataquality.schemas.split import Split

EVAL_METRIC = "f1"
# Generation params
# The default values in Generation Config
# Check more out here: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/text_generation#transformers.GenerationConfig
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.2  # Keep this low for now
TOP_P = 1
TOP_K = 50
# Generate params
GENERATE_ON_TRAIN = False
# Tokenization params
MAX_INPUT_LENGTH = 32
MAX_TARGET_LENGTH = 16

try:
    import evaluate
    from evaluate import EvaluationModule
except ImportError:
    raise GalileoException(
        "⚠️ Huggingface evaluate library not installed "
        "please run `pip install dataquality[evaluate]` "
        "to enable metrics computation."
    )


# Taken from the docs of the trainer module:
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/
# text-classification/run_glue.py#L434
def preprocess_function(
    input_data: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> BatchEncoding:
    # Delay padding until batching - allowing for dynamic padding
    return tokenizer(
        input_data["text"], padding=False, max_length=max_length, truncation=True
    )


def compute_metrics(metric: EvaluationModule, eval_pred: EvalPrediction) -> Dict:
    predictions, labels = np.array(eval_pred.predictions), np.array(eval_pred.label_ids)
    predictions = predictions.argmax(axis=1)
    return metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )


def get_trainer(
    dd: DatasetDict,
    model_checkpoint: str,
    num_train_epochs: int,
    model_max_length: int,
) -> Tuple[Trainer, DatasetDict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True, model_max_length=model_max_length
    )
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

    # encoded_datasets = dd.map(
    #     lambda x: preprocess_function(x, tokenizer, max_padding_length), batched=True
    # )

    # Training arguments and training part
    metric = evaluate.load(EVAL_METRIC)
    # We use the users chosen evaluation metric by preloading it into the partial
    partial(compute_metrics, metric)

    generation_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        # Whether we use multinomial sampling
        do_sample=TEMPERATURE >= 1e-5,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    watch(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        generate_training_data=GENERATE_ON_TRAIN,
        max_input_tokens=MAX_INPUT_LENGTH,
        max_target_tokens=MAX_TARGET_LENGTH,
    )

    return model, dd


def do_train(
    trainer: Trainer,
    encoded_data: DatasetDict,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> Trainer:
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore

    inf_names = [k for k in encoded_data if k not in Split.get_valid_keys()]
    for inf_name in inf_names:
        dq.set_split(Split.inference, inference_name=inf_name)
        trainer.predict(test_dataset=encoded_data[inf_name])
    dq.finish(wait=wait, create_data_embs=create_data_embs)
    return trainer
