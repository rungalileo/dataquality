import math
from typing import Optional

import numpy as np
import pyarrow as pa
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast
from vaex import DataFrame

from dataquality.loggers.data_logger.seq2seq.formatters import BaseSeq2SeqDataFormatter
from dataquality.schemas.seq2seq import (
    GENERATION_BATCH_SIZE,
    TOP_LOGPROBS_SCHEMA,
    BatchGenerationData,
)
from dataquality.schemas.seq2seq import Seq2SeqInputCols as S2SIC
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as S2SOC
from dataquality.utils import tqdm
from dataquality.utils.seq2seq.offsets import align_tokens_to_character_spans


def generate_on_batch(
    texts: pa.array,
    ids: pa.array,
    formatter: BaseSeq2SeqDataFormatter,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    max_input_tokens: int,
    generation_config: GenerationConfig,
    split_key: Optional[str] = None,
) -> BatchGenerationData:
    """Generate over a batch of text inputs

    We use model to generate the output for each text sample *individually* and the
    corresponding logprob + token alignment data. Returns the processed batch
    data to be added to the dataframe.

    Parameters:
    -----------
    texts: pa.array of strs
        batch of str input strings that we want to generate on

    Return:
    -------
    generated_data: BatchGenerationData
        BatchGenerationData object with the processed generation data for the batch
        of text inputs.
    """
    generated_outputs = []
    generated_token_label_positions = []
    generated_token_label_offsets = []
    generated_token_logprobs = []
    generated_top_logprobs = []

    for sample, sample_id in zip(texts, ids):
        # Generate and extract model outputs
        sample_generation = formatter.generate_sample(
            str(sample),
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=max_input_tokens,
            generation_config=generation_config,
            input_id=sample_id.as_py(),  # Convert to int from pyarrow type
            split_key=split_key,
        )

        # Tokenize *exactly* the ids generated. This means
        # we will display special characters such as <eos>
        output = tokenizer.decode(
            sample_generation.generated_ids, skip_special_tokens=False
        )
        generated_outputs.append(output)
        # Re-tokenize the data to get the token position offsets
        # TODO adding <bos> tokens can mess with alignment in the UI. So we
        #   avoid adding special_tokens. This may apply to not just generation!
        encoded_data = tokenizer(
            [output], return_offsets_mapping=True, add_special_tokens=False
        )
        aligned_data = align_tokens_to_character_spans(
            encoded_data["offset_mapping"], disable_tqdm=True
        )
        # aligned_data assumes batches, so for single samples it returns
        # batched list of length == 1.
        generated_token_label_positions.append(aligned_data.token_label_positions[0])
        generated_token_label_offsets.append(aligned_data.token_label_offsets[0])

        generated_logprob_data = sample_generation.generated_logprob_data
        generated_token_logprobs.append(generated_logprob_data.token_logprobs)
        generated_top_logprobs.append(generated_logprob_data.top_logprobs)

    return BatchGenerationData(
        generated_outputs=generated_outputs,
        generated_token_label_positions=generated_token_label_positions,
        generated_token_label_offsets=generated_token_label_offsets,
        generated_token_logprobs=generated_token_logprobs,
        generated_top_logprobs=generated_top_logprobs,
    )


def add_generated_output_to_df(
    df: DataFrame,
    generation_column: str,
    formatter: BaseSeq2SeqDataFormatter,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    max_input_tokens: int,
    generation_config: GenerationConfig,
    split_key: Optional[str] = None,
) -> DataFrame:
    """Generates model outputs over df and extracts the logprob data

    Using the user's model we generate the output for each sample in the df and the
    corresponding logprob data. We generate in batches of Input text using vaex's
    `evaluate_iterator`. This avoids brining the full `S2SIC.input` into memory;
    however, we do end up materializing the full logprob and token alignemnt data
    for the generated outputs.

    We specifically add the following 5 columns to the df:
        - generated_output: str
        - generated_token_label_positions: pa.array
        - generated_token_label_offsets: pa.array
        - generated_token_logprobs: pa.array
        - generated_top_logprobs: pa.array

    NOTE: Although we bring into memory quite a bit of information about the generated
    outputs, in general users won't be generated over very many samples (on the order of
    100-1000s because it simply takes too much time to do much more). Nevertheless, we
    should monitor this for memory issues.

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
    # When generating it is important to set `use_cache = True`.
    # - WHAT? Caching stores intermediate token activations / representations.
    #   During autoregressive generation, the cache is updated each time a token
    #   is generated.
    # - WHY? Caching prevents re-computing token information during auto-regressive
    #   generation, DRAMATICALLY speeding up performance. Every time a new token is
    #   generated, we only need to do the forward pass for a single new token, as we
    #   leverage the cached information to compute transformer based attention.
    model_cache_flag = model.config.use_cache
    model.config.use_cache = True

    generated_data = BatchGenerationData()

    num_batches = math.ceil(len(df) / GENERATION_BATCH_SIZE)
    for _, _, chunk in tqdm(
        df.evaluate_iterator(
            [generation_column, S2SIC.id.value],
            chunk_size=GENERATION_BATCH_SIZE,
            parallel=False,
        ),
        total=num_batches,
        desc="Batched Model Generation",
    ):
        texts, ids = chunk
        batch_generated_data = generate_on_batch(
            texts=texts,
            ids=ids,
            formatter=formatter,
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=max_input_tokens,
            generation_config=generation_config,
            split_key=split_key,
        )

        generated_data.extend_from(batch_generated_data)

    # Assign the vaex columns manually for now!
    df[S2SOC.generated_output.value] = np.array(generated_data.generated_outputs)
    df[S2SOC.generated_token_label_positions.value] = pa.array(
        generated_data.generated_token_label_positions
    )
    df[S2SOC.generated_token_label_offsets.value] = pa.array(
        generated_data.generated_token_label_offsets
    )
    df[S2SOC.generated_token_logprobs.value] = pa.array(
        generated_data.generated_token_logprobs
    )
    df[S2SOC.generated_top_logprobs.value] = pa.array(
        generated_data.generated_top_logprobs, type=TOP_LOGPROBS_SCHEMA
    )

    # Reset the cache flag for the model
    model.config.use_cache = model_cache_flag

    return df
