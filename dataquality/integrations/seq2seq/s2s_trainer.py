from typing import Dict, List, Optional, Tuple

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
    Trainer,
)

import dataquality as dq
from dataquality.integrations.seq2seq.hf import watch
from dataquality.schemas.split import Split
from dataquality.utils.torch import cleanup_cuda

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
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

# Training params
LR = 3e-4
ACCUMULATION_STEPS = 4
BATCH_SIZE = 4


def tokenize(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    input_col: str,
    target_col: str,
    max_input_length: int,
    max_target_length: int,
) -> Dataset:
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
    model_checkpoint: str,
    input_col: str = "text",
    target_col: str = "label",
    max_input_tokens: Optional[int] = None,
    max_target_tokens: Optional[int] = None,
    generation_splits: Optional[List[str]] = None,
) -> Tuple[PreTrainedModel, Dict[str, DataLoader]]:
    """Sets up the model and tokenizer for training

    Note that for now this fn is a misnomer since our initial implementation
    is not using the Trainer class from transformers. We will likely refactor
    this in the future to use the Trainer class.

    For now, this function sets up the model and tokenizer, tokenizes the data,
    for each split, calls the DQ watch function, and returns the model and
    and tokenized dataset dict.
    """
    max_input_tokens = max_input_tokens or MAX_INPUT_LENGTH
    max_target_tokens = max_target_tokens or MAX_TARGET_LENGTH
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_fast=True, model_max_length=max_input_tokens
    )
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)

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
            batch_size=BATCH_SIZE,
            pin_memory=True,
        )
        dataloaders[key] = dl

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
        generation_splits=generation_splits,
        max_input_tokens=max_input_tokens,
        max_target_tokens=max_target_tokens,
    )

    return model, dataloaders


def do_train(
    model: PreTrainedModel,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    num_epochs: int,
    wait: bool,
    create_data_embs: Optional[bool] = None,
) -> Trainer:
    # training and evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataloader = dataloaders.get(Split.training) or dataloaders.get(Split.train)
    eval_dataloader = dataloaders.get(Split.validation) or dataloaders.get(Split.val)
    test_dataloader = dataloaders.get(Split.test)

    optimizer = Adafactor(
        model.parameters(), lr=LR, scale_parameter=False, relative_step=False
    )

    if not train_dataloader:
        raise ValueError("Training data must be provided for Seq2Seq `auto`")

    for epoch in range(num_epochs):
        dq.set_epoch_and_split(split=Split.train, epoch=epoch)
        model.train()
        train_epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader)):
            ids = batch["id"]
            batch = {k: v.to(device) for k, v in batch.items() if k != "id"}
            outputs = model(**batch)
            logits = outputs.logits  # Shape - [bs, bs_seq_ln, vocab]
            dq.log_model_outputs(logits=logits, ids=ids)

            loss = outputs.loss / ACCUMULATION_STEPS

            loss.backward()
            # Grad Accumulation
            if ((step + 1) % ACCUMULATION_STEPS == 0) or (
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

        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )

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

    cleanup_cuda(optimizer, [batch, outputs])
    dq.finish(wait=wait, create_data_embs=create_data_embs)
    return model
