from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast
from vaex import DataFrame

from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig
from dataquality.schemas.seq2seq import (
    AlignedTokenData,
    ModelGeneration,
    Seq2SeqInputCols,
    Seq2SeqModelType,
)
from dataquality.utils.seq2seq.decoder_only import extract_tokenized_responses
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
    process_sample_logprobs,
)
from dataquality.utils.seq2seq.offsets import (
    add_input_cutoff_to_df,
    add_target_cutoff_to_df,
    align_response_tokens_to_character_spans,
    align_tokens_to_character_spans,
)


class BaseSeq2SeqDataFormatter(ABC):
    def __init__(self, logger_config: Seq2SeqLoggerConfig) -> None:
        self.logger_config = logger_config

    @abstractmethod
    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        pass

    @abstractmethod
    def format_text(
        self,
        text: List[str],
        ids: List[int],
        tokenizer: PreTrainedTokenizerFast,
        max_tokens: Optional[int],
        split_key: str,
    ) -> Tuple[AlignedTokenData, List[List[str]], List[str]]:
        """Tokenize and align the `text` samples

        `format_text` tokenizes and computes token alignments for
        each samples in `text`. Different logic is applied depending on the
        model architecture (EncoderDecoder vs. DecoderOnly).

        In the end, we return AlignedTokenData and the target token strings
        (corresponding to `token_label_str` in `Seq2SeqDataLogger`). For both
        EncoderDecoder and DecoderOnly models the output is expected
        to be token alignment and string data over just the <Target> tokens
        in the Seq2Seq task. Note though that the input `text` samples are
        different between the two model architectures. See their respective
        implementations for further details.

        Additional information computed / variable assignements:
            - Assign the necessary `self.logger_config` fields
            - Compute token_label_str: the per token str representation
            of each sample (List[str]), saved and used for high DEP tokens.
            - In Decoder-Only: Decode the response tokens to get the str
            representation of the response (i.e. the target show in the UI).

        Parameters:
        -----------
        texts: List[str]
            batch of str samples. For EncoderDecoder model's these are exactly
            the targets vs. for DecoderOnly model's each sample is the full
            formatted_prompt
        ids: List[int]
            sample ids - used for logger_config assignment
        tokenizer: PreTrainedTokenizerFast
        max_tokens: Optional[int]
        split_key: str

        Return:
        -------
        batch_aligned_data: AlignedTokenData
            Aligned token data for *just* target tokens, based on `text`
        token_label_str: List[List[str]]
            The target tokens (as strings) - see `Seq2SeqDataLogger.token_label_str`
        targets: List[str]
            The decoded response tokens - i.e. the string representation of the
            Targets for each sample. Note that this is only computed for
            Decoder-Only models. Returns [] for Encoder-Decoder
        """
        pass

    @torch.no_grad()
    @abstractmethod
    def generate_sample(
        self,
        input_str: str,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        max_input_tokens: int,
        generation_config: GenerationConfig,
        input_id: Optional[int] = None,
        split_key: Optional[str] = None,
    ) -> ModelGeneration:
        """Generate and extract model logprobs

        Tokenize the input string and then use `hf.generate`
        to generate just the output tokens.

        We don't rely on the scores returned by hf since these
        can be altered by hf internally depending on the generation
        config / can be hard to parse in the case of BEAM Search.

        Instead, we pass the generated output back through the model
        to extract the token logprobs. We effectively ask the model
        to evaluate its own generation - which is identical to generation
        because of causal language modeling.

        Parameters:
        -----------
        input_str: str
            Input string context used to seed the generation
        tokenizer: PreTrainedTokenizerFast
        max_input_tokens: int
            the max number of tokens to use for tokenization
        model: PreTrainedModel
        generation_config: GenerationConfig
            Users generation config specifying the parameters for generation

        Return:
        -------
        model_generation: ModelGeneration
            - generated_ids: np.ndarray of shape - [seq_len]
            - generated_token_logprobs: np.ndarray of shape - [seq_len]
            - generated_top_logprobs: List[List[Tuple[str, float]]]
        """
        pass

    @staticmethod
    def process_generated_logits(
        generated_logits: torch.Tensor,
        generated_ids: np.ndarray,
        tokenizer: PreTrainedTokenizerFast,
    ) -> ModelGeneration:
        logprobs = (
            torch.nn.functional.log_softmax(generated_logits, dim=-1).cpu().numpy()
        )
        top_logprobs_indices = get_top_logprob_indices(logprobs)

        gen_logprob_data = process_sample_logprobs(
            logprobs,
            sample_labels=generated_ids,
            sample_top_indices=top_logprobs_indices,
            tokenizer=tokenizer,
        )

        return ModelGeneration(
            generated_ids=generated_ids, generated_logprob_data=gen_logprob_data
        )


class EncoderDecoderDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for EncoderDecoder models

    Logging input data for EncoderDecoder models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(tokenizer, ...)` functions in `dataquality.integrations.seq2seq.core`
    2. A two column (i.e. completion) dataset (pandas/huggingface etc) with string
        'text' (model <Input> / <Instruction> / <Prompt>, ...) and 'label' (model
        <Target> / (<Completion> / ...) columns + a data sample id column.
        Ex: Billsum dataset, with `text` <Input> and `summary` as the <Label>
        id  text	                        summary
        0	SECTION 1. LIABILITY ...	    Shields a business entity ...
        1	SECTION 1. SHORT TITLE.\n\n ...	Human Rights Information Act ...
        2	SECTION 1. SHORT TITLE.\n\n ...	Jackie Robinson Commemorative Coin ...
        3	SECTION 1. NONRECOGNITION ...	Amends the Internal Revenue Code to ...
        4	SECTION 1. SHORT TITLE.\n\n ...	Native American Energy Act - (Sec. 3...

        You can log your dataset via the `dq.log_dataset` function, passing in the
        column mapping as necessary for `text`, `label`, and `id`
        `dq.log_dataset(ds, text="text", label="summary", id="id")`

    Putting it all together:
        from dataquality.integrations.seq2seq.core import set_tokenizer
        from datasets import load_dataset
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        ds = load_dataset("billsum")
        # Add `id` column to each dataset split as the idx
        ds = ds.map(lambda x,idx : {"id":idx},with_indices=True)
        dq.init("seq2seq")
        # You can either use `set_tokenizer()` or `watch()`
        set_tokenizer(
            tokenizer,
            "encoder_decoder",
            max_input_tokens=512,
            max_target_tokens=128
        )
        dq.log_dataset(ds["train"], label="summary", split="train")

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer as to match the tokenization process to your training process.

    NOTE 2: Unlike DecoderOnly models, EncoderDecoder models explicitly separate the
    processing of the <Input> and <Target> data. Therefore, we do not need any
    additional information to isolate / extract information on the <Target> data.
    """

    def format_text(
        self,
        text: List[str],
        ids: List[int],
        tokenizer: PreTrainedTokenizerFast,
        max_tokens: Optional[int],
        split_key: str,
    ) -> Tuple[AlignedTokenData, List[List[str]], List[str]]:
        """Further validation for Encoder-Decoder

        For Encoder-Decoder we need to:
            - Save the target token ids: Equivalent to ground truth, it allows us to
                compare with the predictions and get perplexity and DEP scores
            - Save the target tokens: Decoding of the ids, to identify the tokens
            - Save the offsets and positions of the target tokens: allows us to extract
                token level information and align the tokens with the full sample text

        We achieve this by:
            - Tokenize the target texts using `max_target_tokens`
            - From the tokenized outputs generate the corresponding token alignments
                (i.e. label_offsets and lable_positions).
            - Save ground-truth token id in `id_to_tokens` map, mapping
                sample id to tokenized label (sample_id -> List[token_id])
        """
        targets = text
        use_special_tokens = True  # use this var to align encoding and decoding
        encoded_data = tokenizer(
            targets,
            return_offsets_mapping=True,
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=use_special_tokens,
        )
        token_label_ids = encoded_data["input_ids"]
        # Need to decode row by row otherwise each row is joined into one string
        token_label_str = [
            tokenizer.batch_decode(
                row,
                skip_special_tokens=not use_special_tokens,
                clean_up_tokenization_spaces=True,
            )
            for row in token_label_ids
        ]

        batch_aligned_data = align_tokens_to_character_spans(
            encoded_data["offset_mapping"]
        )

        # Save the token_ids in the config (to share with the model logger)
        id_to_tokens = dict(zip(ids, token_label_ids))
        self.logger_config.id_to_tokens[split_key].update(id_to_tokens)

        return batch_aligned_data, token_label_str, []

    @torch.no_grad()
    def generate_sample(
        self,
        input_str: str,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        max_input_tokens: int,
        generation_config: GenerationConfig,
        input_id: Optional[int] = None,
        split_key: Optional[str] = None,
    ) -> ModelGeneration:
        """Generate response for a single sample - Encoder Decoder"""

        # Shape - [1, seq_len]
        # We make sure to tokenize with the same max_length as they did during training
        input_ids = tokenizer(
            input_str,
            truncation=True,
            max_length=max_input_tokens,
            verbose=False,
            return_tensors="pt",
        )["input_ids"].to(model.device)

        gen_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )

        # Strip the beginning <pad> token and keep as Tensor
        gen_ids = gen_ids[..., 1:]

        # Pass the generated output through the model to get the logits
        model_outputs = model(input_ids=input_ids, labels=gen_ids)
        logits = model_outputs.logits

        # Remove singleton batch dimension
        logits = logits.squeeze(0)  # [seq_len, vocab_size]
        gen_ids = gen_ids.squeeze(0).cpu().numpy()  # [seq_len]

        return self.process_generated_logits(
            logits, generated_ids=gen_ids, tokenizer=tokenizer
        )

    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the input strings.

        When using Encoder-Decoder models, the input tokens are truncated
        based on the respective Encoders max_lengths OR the user specified
        max_length (note: these may be different between Encoder and Decoder
        - see `max_input_tokens` vs. `max_target_tokens).

        This function adds one column to the df:
          - 'input_cutoff': the position of the last character in the input.
        """
        tokenizer = self.logger_config.tokenizer
        max_input_length = self.logger_config.max_input_tokens
        df = add_input_cutoff_to_df(
            df, tokenizer, Seq2SeqInputCols.input, max_input_length
        )

        return df


class DecoderOnlyDataFormatter(BaseSeq2SeqDataFormatter):
    """Seq2Seq data logger for DecoderOnly models

    Logging input data for DecoderOnly models requires:
    1. tokenizer: This must be an instance of PreTrainedTokenizerFast from huggingface
        (ie T5TokenizerFast or GPT2TokenizerFast, etc). Your tokenizer should have an
        `.is_fast` property that returns True if it's a fast tokenizer.
        This class must implement the `encode`, `decode`, and `encode_plus` methods

        You can set your tokenizer via either the seq2seq `set_tokenizer()` or
        `watch(tokenizer, ...)` functions in `dataquality.integrations.seq2seq.core`
    2. A two column (i.e. completion) dataset (pandas/huggingface etc) with string
        'text' (model <Input> / <Instruction> / <Prompt>, ...) and 'label' (model
        <Target> / (<Completion> / ...) columns + a data sample id column.
        Ex: Billsum dataset, with `text` <Input> and `summary` as the <Label>
        id  text	                        summary
        0	SECTION 1. LIABILITY ...	    Shields a business entity ...
        1	SECTION 1. SHORT TITLE.\n\n ...	Human Rights Information Act ...
        2	SECTION 1. SHORT TITLE.\n\n ...	Jackie Robinson Commemorative Coin ...
        3	SECTION 1. NONRECOGNITION ...	Amends the Internal Revenue Code to ...
        4	SECTION 1. SHORT TITLE.\n\n ...	Native American Energy Act - (Sec. 3...

        You can log your dataset via the `dq.log_dataset` function, passing in the
        column mapping as necessary for `text`, `label`, and `id`
        `dq.log_dataset(ds, text="text", label="summary", id="id")`

    Putting it all together:
        from dataquality.integrations.seq2seq.core import set_tokenizer
        from datasets import load_dataset
        from transformers import T5TokenizerFast

        tokenizer = T5TokenizerFast.from_pretrained("t5-small")
        ds = load_dataset("billsum")
        # Add `id` column to each dataset split as the idx
        ds = ds.map(lambda x,idx : {"id":idx},with_indices=True)
        dq.init("seq2seq")
        # You can either use `set_tokenizer()` or `watch()`
        set_tokenizer(
            tokenizer,
            "encoder_decoder",
            max_input_tokens=512,
            max_target_tokens=128
        )
        dq.log_dataset(ds["train"], label="summary", split="train")

    NOTE: We assume that the tokenizer you provide is the same tokenizer used for
    training. This must be true in order to align inputs and outputs correctly. Ensure
    all necessary properties (like `add_eos_token`) are set before setting your
    tokenizer as to match the tokenization process to your training process.

    NOTE 2: Unlike DecoderOnly models, EncoderDecoder models explicitly separate the
    processing of the <Input> and <Target> data. Therefore, we do not need any
    additional information to isolate / extract information on the <Target> data.
    """

    def format_text(
        self,
        text: List[str],
        ids: List[int],
        tokenizer: PreTrainedTokenizerFast,
        max_tokens: Optional[int],
        split_key: str,
    ) -> Tuple[AlignedTokenData, List[List[str]], List[str]]:
        """Further formatting for Decoder-Only

        Text is the formatted prompt of combined input/target

        Tokenize text using the user's `max_input_tokens`. From
        the tokenized outputs generate the corresponding token alignments
        (i.e. label_offsets and lable_positions).

        Save the tokenized labels for each sample as `id_to_tokens`. This
        is essential during model logging for extracting GT token label
        information.

        We also save a `formatted_prompt_lengths` map used later to remove
        padding tokens.
        """
        # For decoder-only the text is the formatted prompt (input/target combined)
        formatted_prompts = text
        max_input_tokens = max_tokens
        use_special_tokens = True  # use this var to align encoding and decoding
        encoded_data = tokenizer(
            formatted_prompts,
            max_length=max_tokens,
            truncation=True,
            add_special_tokens=use_special_tokens,
        )
        # Tokenized input/target combination
        tokenized_formatted_prompts = encoded_data["input_ids"]

        # Split each sample based on the location of the response template
        # This is equivalent to `tokenized_labels` in encoder-decoder
        assert self.logger_config.response_template  # Necessary for linting
        tokenized_labels = extract_tokenized_responses(
            tokenized_formatted_prompts, self.logger_config.response_template
        )

        # Empty initialization
        batch_aligned_data = AlignedTokenData([], [])
        token_label_str = []
        targets = []

        # Decode then re-tokenize just the response labels to get correct offsets
        for token_label_ids in tqdm(
            tokenized_labels,
            leave=False,
            desc="Aligning string characters with tokenizer representation",
        ):
            # Detokenize to save the token_str in the df (for ex for high DEP tokens)
            token_label_str.append(
                tokenizer.batch_decode(
                    token_label_ids,
                    skip_special_tokens=not use_special_tokens,
                    clean_up_tokenization_spaces=True,
                )
            )

            (
                response_aligned_data,
                response_str,
            ) = align_response_tokens_to_character_spans(
                tokenizer,
                token_label_ids,
                max_input_tokens,
            )
            batch_aligned_data.append(response_aligned_data)
            targets.append(response_str)

        # Save the tokenized response labels for each samples
        id_to_tokens = dict(zip(ids, tokenized_labels))
        self.logger_config.id_to_tokens[split_key].update(id_to_tokens)

        # Save the length of the formatted prompt - used later to remove padding
        formatted_prompt_lengths = [
            len(prompt) for prompt in tokenized_formatted_prompts
        ]
        id_to_formatted_prompt_length = dict(zip(ids, formatted_prompt_lengths))
        self.logger_config.id_to_formatted_prompt_length[split_key].update(
            id_to_formatted_prompt_length
        )

        return batch_aligned_data, token_label_str, targets

    @torch.no_grad()
    def generate_sample(
        self,
        input_str: str,
        tokenizer: PreTrainedTokenizerFast,
        model: PreTrainedModel,
        max_input_tokens: int,
        generation_config: GenerationConfig,
        input_id: Optional[int] = None,
        split_key: Optional[str] = None,
    ) -> ModelGeneration:
        """Generate response for a single sample - Decoder Only"""

        # Retrieve the length of the response tokens
        assert split_key, "Decoder-Only models require a valid `split_key`"
        assert input_id is not None, (
            "Decoder-Only models require the `sample_id` "
            "that is being used for generation"
        )
        response_labels = self.logger_config.id_to_tokens[split_key][input_id]
        num_response_labels = len(response_labels)

        # Shape - [1, seq_len]
        # We make sure to tokenize with the same max_length as they did during training
        formatted_prompt_ids = tokenizer(
            input_str,
            truncation=True,
            max_length=max_input_tokens,
            verbose=False,
            return_tensors="pt",
        )["input_ids"]

        len_prompt_ids_without_response = (
            len(formatted_prompt_ids[0]) - num_response_labels
        )
        prompt_ids_without_response = formatted_prompt_ids[
            :, :len_prompt_ids_without_response
        ].to(model.device)

        # This returns ALL the ids (prompt + response)
        full_gen_ids = model.generate(
            input_ids=prompt_ids_without_response,
            generation_config=generation_config,
        )

        # Remove batch dimension
        full_gen_ids_cpu = full_gen_ids.cpu().numpy()
        gen_ids = full_gen_ids_cpu[0, len(prompt_ids_without_response[0]) :]

        # Pass generated output through the model to get the logits
        model_outputs = model(input_ids=full_gen_ids, labels=full_gen_ids.clone())

        logits = model_outputs.logits
        # Shift sample tokens such that tokens < n predict token n.
        response_logits = logits[0, -(len(gen_ids) + 1) : -1]

        return self.process_generated_logits(
            response_logits, generated_ids=gen_ids, tokenizer=tokenizer
        )

    def set_input_cutoff(self, df: DataFrame) -> DataFrame:
        """Calculate the cutoff index for the inputs

        Set the cutoff for the Input to just be the entire sample
            i.e. the length of `input`
        """
        # Assign input_cutoff always to be the full strings
        df[Seq2SeqInputCols.input_cutoff.value] = df[Seq2SeqInputCols.input].str.len()

        target_offsets_colname = Seq2SeqInputCols.token_label_offsets
        if target_offsets_colname in df.get_column_names():
            df = add_target_cutoff_to_df(df, target_offsets_colname)

        return df


FORMATTER_MAP: Dict[Seq2SeqModelType, Type[BaseSeq2SeqDataFormatter]] = {
    Seq2SeqModelType.encoder_decoder: EncoderDecoderDataFormatter,
    Seq2SeqModelType.decoder_only: DecoderOnlyDataFormatter,
}


def get_data_formatter(
    model_type: Seq2SeqModelType, logger_config: Seq2SeqLoggerConfig
) -> BaseSeq2SeqDataFormatter:
    """Returns the data formatter for the given model_type"""
    return FORMATTER_MAP[model_type](logger_config)
