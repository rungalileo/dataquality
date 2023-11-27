from contextlib import nullcontext
from dataclasses import asdict
from typing import ContextManager, Dict, TextIO, Tuple, Union

import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    T5ForConditionalGeneration,
)

import dataquality as dq
from dataquality.exceptions import GalileoException
from dataquality.integrations.seq2seq.core import watch
from dataquality.integrations.seq2seq.schema import (
    Seq2SeqGenerationConfig,
    Seq2SeqTrainingConfig,
)
from dataquality.schemas.seq2seq import Seq2SeqModelType
from dataquality.schemas.split import Split
from dataquality.utils.torch import cleanup_cuda


def validate_cols(ds: Dataset, input_col: str, target_col: str) -> None:
    """Validates that the input and target columns are in the dataset"""
    template = (
        "{col} column {val} not found in dataset. "
        "Please check the DatasetConfig to ensure the {col_name} is correct."
        "If you are using a custom formatter, please ensure that the "
        "{col_name} is being set correctly.\n\n"
    )
    error_msg = ""
    if input_col not in ds.column_names:
        error_msg += template.format(col="Input", val=input_col, col_name="input_col")
    if target_col not in ds.column_names:
        error_msg += template.format(
            col="Target", val=target_col, col_name="target_col"
        )

    if error_msg:
        raise GalileoException(error_msg)


def tokenize(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    input_col: str,
    target_col: str,
    max_input_length: int,
    max_target_length: int,
) -> Dataset:
    validate_cols(ds, input_col, target_col)

    def _tokenize(row: Dict) -> BatchEncoding:
        """Tokenize the input and outputs

        Creates the following columns

        Input cols:
        - id
        - input_ids
        - attention_mask

        Output cols:
        - labels
        """
        model_inputs = tokenizer(
            row[input_col],
            truncation=True,
            max_length=max_input_length,
            padding=False,
            return_tensors=None,
        )
        labels = tokenizer(
            row[target_col],
            truncation=True,
            max_length=max_target_length,
            padding=False,
            return_tensors=None,
        ).input_ids

        model_inputs["labels"] = labels
        model_inputs["id"] = row["id"]
        return model_inputs

    ds_tokenized = ds.map(
        lambda x: _tokenize(x),
        remove_columns=ds.column_names,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    return ds_tokenized


def get_trainer(
    dd: DatasetDict,
    input_col: str,
    target_col: str,
    training_config: Seq2SeqTrainingConfig,
    generation_config: Seq2SeqGenerationConfig,
) -> Tuple[PreTrainedModel, Dict[str, DataLoader]]:
    """Sets up the model and tokenizer for training

    Note that for now this fn is a misnomer since our initial implementation
    is not using the Trainer class from transformers. We will likely refactor
    this in the future to use the Trainer class.

    For now, this function sets up the model and tokenizer, tokenizes the data,
    for each split, calls the DQ watch function, and returns the model and
    and tokenized dataset dict.
    """
    max_input_tokens = training_config.max_input_tokens
    max_target_tokens = training_config.max_target_tokens
    tokenizer = AutoTokenizer.from_pretrained(
        training_config.model, use_fast=True, model_max_length=max_input_tokens
    )
    model = T5ForConditionalGeneration.from_pretrained(training_config.model)

    # Setup the dataloader
    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)

    dataloaders = {}
    for key in dd.keys():
        shuffle = key in ["train", "training"]
        ds_tokenized = tokenize(
            dd[key],
            tokenizer,
            input_col,
            target_col,
            max_input_tokens,
            max_target_tokens,
        )
        dl = DataLoader(
            ds_tokenized,
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=training_config.batch_size,
            pin_memory=True,
        )
        dataloaders[key] = dl

    hf_generation_config = GenerationConfig(
        **asdict(generation_config),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    watch(
        tokenizer=tokenizer,
        model_type=Seq2SeqModelType.encoder_decoder,
        model=model,
        generation_config=hf_generation_config,
        generation_splits=generation_config.generation_splits,
        max_input_tokens=max_input_tokens,
        max_target_tokens=max_target_tokens,
    )
    return model, dataloaders


def do_train(
    model: PreTrainedModel,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    training_config: Seq2SeqTrainingConfig,
    wait: bool,
) -> PreTrainedModel:
    # training and evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataloader = dataloaders.get(Split.training) or dataloaders.get(Split.train)
    eval_dataloader = dataloaders.get(Split.validation) or dataloaders.get(Split.val)
    test_dataloader = dataloaders.get(Split.test)

    optimizer = Adafactor(
        model.parameters(),
        lr=training_config.learning_rate,
        scale_parameter=False,
        relative_step=False,
    )

    if not train_dataloader:
        raise ValueError("Training data must be provided for Seq2Seq `auto`")

    skip_train = training_config.epochs == 0
    train_context: Union[TextIO, ContextManager[None]] = (
        torch.no_grad() if skip_train else nullcontext()
    )  # simply defining the context, setting the type is to make the linter happy

    # If skip_train=True, we add 1 epoch so we can do inference and still log the data
    for epoch in range(training_config.epochs + int(skip_train)):
        dq.set_epoch_and_split(split=Split.train, epoch=epoch)
        model.eval() if skip_train else model.train()
        train_epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            ids = batch["id"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}

            with train_context:
                outputs = model(**batch)

            logits = outputs.logits  # Shape - [bs, bs_seq_ln, vocab]
            dq.log_model_outputs(logits=logits, ids=ids)

            loss = outputs.loss / training_config.accumulation_steps

            if not skip_train:
                loss.backward()
                # Grad Accumulation
                if ((step + 1) % training_config.accumulation_steps == 0) or (
                    (step + 1) == len(train_dataloader)
                ):
                    optimizer.step()
                    optimizer.zero_grad()

            step_loss = loss.detach().cpu().item()
            train_epoch_loss += step_loss

        train_epoch_loss = train_epoch_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.Tensor([train_epoch_loss])).float()
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        if not eval_dataloader:
            continue

        # Val step
        model.eval()
        dq.set_epoch_and_split(split=Split.validation, epoch=epoch)
        eval_epoch_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(eval_dataloader)):
                ids = batch["id"]
                batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
                outputs = model(**batch)
                logits = outputs.logits  # Shape - [bs, bs_seq_ln, vocab]
                dq.log_model_outputs(logits=logits, ids=ids)

                loss = outputs.loss
                eval_step_loss = loss.cpu().item()
                eval_epoch_loss += eval_step_loss

            # Look just at the loss in aggregate!
            eval_epoch_loss = eval_epoch_loss / len(eval_dataloader)
            eval_ppl = torch.exp(torch.Tensor([eval_epoch_loss])).item()

        print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")

    # After training, do test step
    if test_dataloader:
        model.eval()
        dq.set_epoch_and_split(split=Split.test, epoch=epoch)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dataloader)):
                ids = batch["id"]
                batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
                outputs = model(**batch)
                logits = outputs.logits  # Shape - [bs, bs_seq_ln, vocab]
                dq.log_model_outputs(logits=logits, ids=ids)

    # Cleanup all unused data on the GPU and any references
    # to that data
    cleanup_cuda(optimizer=optimizer, tensors=[logits, loss, batch, outputs])
    dq.finish(
        wait=wait,
        create_data_embs=training_config.create_data_embs,
        data_embs_col=training_config.data_embs_col,
    )
    return model
