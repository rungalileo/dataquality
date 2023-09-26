import pyarrow as pa
import torch
import vaex
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.schemas.seq2seq import (
    TOP_LOGPROBS_SCHEMA,
    ModelGeneration,
)
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as S2SOC
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
    process_sample_logprobs,
)
from dataquality.utils.seq2seq.offsets import align_tokens_to_character_spans


@torch.no_grad()
def generate_sample_output(
    input_str: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    max_input_tokens: int,
    generation_config: GenerationConfig,
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
    max_input_tokens: the max number of tokens to use for tokenization
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
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()

    # Remove singleton dimensions
    logprobs = logprobs.squeeze()  # [seq_len, vocab_size]
    gen_ids = gen_ids.squeeze().cpu().numpy()  # [seq_len]

    top_logprobs_indices = get_top_logprob_indices(logprobs)

    gen_logprob_data = process_sample_logprobs(
        logprobs,
        sample_labels=gen_ids,
        sample_top_indices=top_logprobs_indices,
        tokenizer=tokenizer,
    )

    return ModelGeneration(
        generated_ids=gen_ids, generated_logprob_data=gen_logprob_data
    )


def add_generated_output_to_df(
    df: vaex.DataFrame,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerFast,
    max_input_tokens: int,
    generation_config: GenerationConfig,
) -> vaex.DataFrame:
    """Generates model outputs over df and extracts the logprob data

    Using the user's model we generate the output for each sample in the df and the
    corresponding logprob data. We use a vaex register function to batch the processing
    across df; however, we generate model outputs one sample at a time.

    We specifically add the following 5 columns to the df:
        - generated_output: str
        - generated_token_label_positions: pa.array
        - generated_token_label_offsets: pa.array
        - generated_token_logprobs: pa.array
        - generated_top_logprobs: pa.array

    Note: We return a pa.StructArray to extract multiple columns of info
    at once through vaex. We then have to seperate the Struct into individual
    columns.

    Parameters:
    -----------
    df: vaex.DataFrame
        Dataframe with the input data that we want to generate based on
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerFast
    max_input_tokens: the max number of tokens to use for tokenizing
    generation_config: GenerationConfig
        Users generation config specifying the parameters for generation

    Return:
    -------
    df: vaex.DataFrame
        Updated Dataframe with the generated columns added (see above)
    """
    model.eval()

    @vaex.register_function()
    def generate_batch_outputs(texts: pa.array) -> pa.StructArray:
        """Generated model outputs for a batch of text inputs

        For each sample in the batch, extract all of the necessary
        information to populate the vaex df. We return a somewhat specail
        StructArray, because you can only return an expression for a single
        column in vaex, but here we extract the content for 5 columns -
        (see above). By creating aStructArray (which is just an arrow dictionary)
        we can then xtract out each one into an independent column after.
        """
        generated_outputs = []
        generated_token_label_positions = []
        generated_token_label_offsets = []
        generated_token_logprobs = []
        generated_top_logprobs = []

        for sample in texts:
            # Generate and extract model outputs
            sample_generation = generate_sample_output(
                input_str=str(sample),
                model=model,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
                generation_config=generation_config,
            )

            output = tokenizer.decode(
                sample_generation.generated_ids, skip_special_tokens=True
            )
            generated_outputs.append(output)
            # Re-tokenize the data to get the token position offsets
            encoded_data = tokenizer([output], return_offsets_mapping=True)
            aligned_data = align_tokens_to_character_spans(
                encoded_data["offset_mapping"]
            )
            # aligned_data assumes batches, so for single samples it returns
            # batched list of length == 1.
            generated_token_label_positions.append(
                aligned_data.token_label_positions[0]
            )
            generated_token_label_offsets.append(aligned_data.token_label_offsets[0])

            generated_logprob_data = sample_generation.generated_logprob_data
            generated_token_logprobs.append(generated_logprob_data.token_logprobs)
            generated_top_logprobs.append(generated_logprob_data.top_logprobs)

        # Ensure correct pyarrow format for top_logprobs
        generated_top_logprobs = pa.array(
            generated_top_logprobs, type=TOP_LOGPROBS_SCHEMA
        )
        return pa.StructArray.from_arrays(
            arrays=[
                generated_outputs,
                generated_token_label_positions,
                generated_token_label_offsets,
                generated_token_logprobs,
                generated_top_logprobs,
            ],
            names=S2SOC.generated_cols(),
        )

    df[S2SOC.generation_data.value] = df.func.generate_batch_outputs(df[S2SIC.text])
    # 'generation_data' is a col of type StructArray
    # struct.flatten creates a column per key
    df = df.struct.flatten(S2SOC.generation_data.value)
    # the new column name will be `generation_data_<key>` so we rename them
    for col in S2SOC.generated_cols():
        df.rename(f"{S2SOC.generation_data}_{col}", col)

    return df
