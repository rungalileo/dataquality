import copy
import re
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import col, udf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sparknlp.base.finisher import Finisher
from sparknlp.base.light_pipeline import LightPipeline
from sparknlp.internal import AnnotatorTransformer
from sparknlp.pretrained.pretrained_pipeline import PretrainedPipeline

MAX_TOKEN_LEN = 512

PipelineLike = Union[Pipeline, LightPipeline, PipelineModel, PretrainedPipeline]


def extract_begin_end(token_list: List[Row]) -> List[List[int]]:
    """Extracts the begin and end of each token in a list of tokens
    :param token_list: A list of tokens
    :return: A list of lists of integers, where each sublist contains the begin and
        end of a token
    """
    # For some labels begin is equal to end
    return [[row.begin, row.end + 1] for row in token_list]


def extract_embeddings(embeddings_list: List[Row]) -> List[List[float]]:
    """Extracts the embeddings from a list of embeddings
    :param embeddings_list: A list of embeddings
    :return: A list of lists of floats, where each sublist contains the embeddings
        of a token
    """
    return [row.embeddings for row in embeddings_list]


def extract_ner_result(ner_list: List[Row]) -> List:
    """Extracts the NER result from a list of NER results
    :param ner_list: A list of NER results
    :return: A list of strings, where each string is the NER result of a token
    """
    return [row.result for row in ner_list]


def extract_ner_gold_spans(ner_list: List[Row]) -> List[dict]:
    """Extracts the NER gold spans from a list of NER results
    :param ner_list: A list of NER results
    :return: A list of dictionaries, where each dictionary contains the start, end
        and label of a gold span
    """
    spans = []
    current_span = None
    for row in ner_list:
        # Check if the row is a beginning of a named entity (B-)
        if row.result.startswith("B-"):
            # If there is an existing span, we add it to the list
            if current_span:
                spans.append(current_span)
            # Start a new span

            label = row.result
            match = re.match(r"^\w?-(.*)", label)
            if match:
                label = match.group(1)
            current_span = {
                "start": row.begin,
                "end": row.end + 1,
                "label": label,
            }

        # Check if the row is inside a named entity (I-) and there's an active span
        elif row.result.startswith("I-") and current_span:
            # Extend the current span to include the current row
            current_span["end"] = row.end + 1

        # If it's outside of a named entity (O)
        elif row.result == "O":
            # If there's an active span, we add it to the list and reset current_span
            if current_span:
                spans.append(current_span)
                current_span = None

    # Add any remaining span after the loop
    if current_span:
        spans.append(current_span)

    return spans


def convert_ner_to_list(labels: List[str]) -> Callable:
    """Converts a list of NER results to a list of lists of floats
    :param labels: A list of NER labels
    :return: A function that converts a list of NER results to a list of lists of
        floats
    """
    labels = labels

    def convert_ner_to_list(ner_list: List[Row]) -> List[List[float]]:
        return [
            [float(row.metadata.get(tag, "0")) for tag in labels] for row in ner_list
        ]

    return convert_ner_to_list


def prepare_df_for_dq(
    df: DataFrame, labels: List[str], embeddings_col: str = "embeddings"
) -> DataFrame:
    """Prepares a dataframe for DQ
    :param df: The pyspark dataframe to be prepared
    :param labels: The list of labels
    :param embeddings_col: The name of the embeddings column (default: embeddings)
    :return: The prepared dataframe
    """
    embeddings_udf = udf(extract_embeddings, ArrayType(ArrayType(FloatType())))
    convert_ner_to_list_udf = udf(
        convert_ner_to_list(labels), ArrayType(ArrayType(FloatType()))
    )

    df = df.withColumn(
        embeddings_col, embeddings_udf(col(embeddings_col))
    )  # Do select instead
    df = df.withColumn(
        "ner_list", convert_ner_to_list_udf(col("ner"))
    )  # Do select instead
    df = df.drop("document").drop("sentence").drop("entities").drop("token")
    return df


def add_labels_df(df: DataFrame) -> DataFrame:
    """Adds labels to a dataframe
    :param df: The pyspark dataframe to be prepared
    :return: The prepared dataframe"""
    token_udf = udf(extract_begin_end, ArrayType(ArrayType(IntegerType())))
    gold_spans_udf = udf(
        extract_ner_gold_spans,
        ArrayType(
            StructType(
                [
                    StructField("start", IntegerType(), True),
                    StructField("end", IntegerType(), True),
                    StructField("label", StringType(), True),
                ]
            )
        ),
    )
    extract_ner_result_udf = udf(extract_ner_result, ArrayType(StringType()))

    df = df.withColumn("ner_start_end", token_udf(col("ner")))
    df = df.withColumn("gold_spans", gold_spans_udf(col("label")))
    df = df.withColumn("label", extract_ner_result_udf(col("label")))
    df = df.drop("sentence")
    df = df.drop("document")

    df = df.drop("ner")
    df = df.drop("pos")
    return df


def pad_df(
    df: pd.DataFrame,
    emb_col: str = "embeddings",
    label_col: str = "ner_list",
    max_length: int = 512,
) -> pd.DataFrame:
    """Pads a dataframe
    :param df: The pandas dataframe to be padded
    :param emb_col: The name of the embeddings column (default: embeddings)
    :param label_col: The name of the labels column (default: ner_list)
    :param max_length is the MAX_TOKEN_LEN to make sure we don't have very
        large documents
    :return: The padded dataframe
    """
    df["embs_padded"] = [
        np.pad(
            row,
            pad_width=((0, max_length - len(row)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        for row in df[emb_col]
    ]
    df["probs_padded"] = [
        np.pad(
            row,
            pad_width=((0, max_length - len(row)), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        for row in df[label_col]
    ]
    return df


# Function to align gold spans with token boundaries
def align_gold_spans(
    gold_spans: List[Dict], token_boundaries_in: List[List[List[int]]]
) -> List[List[List[int]]]:
    """Aligns gold spans with token boundaries
    :param gold_spans: A list of dictionaries, where each dictionary contains the
        start, end and label of a gold span
    :param token_boundaries_in: A list of lists of integers, where each sublist
        contains the begin and end of a token
    :return: A list of lists of integers, where each sublist contains the begin and
        end of a token"""
    token_boundaries = copy.deepcopy(token_boundaries_in)
    # Iterate over each list of gold spans in the same order
    for idx, gold_list in enumerate(gold_spans):
        token_list = token_boundaries[idx]
        token_idx_start = 0

        # Iterate over each gold span in the list
        for gold in gold_list:
            gold_start, gold_end = gold["start"], gold["end"]

            # Iterate over token boundaries to find the alignment
            for i in range(token_idx_start, len(token_list)):
                start: int = token_list[i][0]
                end: int = token_list[i][1]

                # If start of token matches start of gold span, update index
                if start == gold_start:
                    token_idx_start = i

                # If end of token is greater or equal
                # to end of gold span, update and break
                if end >= gold_end:
                    if end > gold_end:
                        token_list[i - 1][1] = gold_end
                    break

    return token_boundaries


def get_stages(pipe: PipelineLike) -> Any:
    """Get all stages of a pipeline
    :param pipe: The pipeline to get the stages from
    :return: A list of stages
    """
    if isinstance(pipe, Pipeline):
        stages = pipe.getStages()
    elif isinstance(pipe, LightPipeline):
        stages = pipe.pipeline_model.stages
    elif isinstance(pipe, PipelineModel):
        stages = pipe.stages
    elif isinstance(pipe, PretrainedPipeline):
        stages = pipe.model.stages
    else:
        raise TypeError(
            """pipe must be one of Pipeline, LightPipeline, PipelineModel
            PretrainedPipeline"""
        )
    return stages


def get_stage_of_class(
    pipe: PipelineLike, classes_to_filter: Union[List, Tuple]
) -> List[Any]:
    """Get all stages of a pipeline which are one of the classes in classes_to_filter
    :param pipe: The pipeline to get the stages from
    :param classes_to_filter: A list of classes to filter the stages by
    :return: A list of stages
    """
    # Get all stages of pipeline which are one of the classes in classes_to_filter
    if isinstance(classes_to_filter, list):
        classes_to_filter = tuple(classes_to_filter)

    stages = get_stages(pipe)
    filterd_stages = [s for s in stages if isinstance(s, classes_to_filter)]
    return filterd_stages


def is_col_match(provider: AnnotatorTransformer, consumer: Finisher) -> bool:
    """Check if the output column of a provider matches the input column of a consumer
    :param provider: The provider to check
    :param consumer: The consumer to check
    :return: True if the output column of the provider matches the input column of the
        consumer, False otherwise"""
    return provider.getOutputCol() in consumer.getInputCols()


def is_provider_match(provider: AnnotatorTransformer, required_type: str) -> bool:
    """Check if the output type of a provider matches the required type
    :param provider: The provider to check
    :param required_type: The required type to check
    :return: True if the output type of the provider matches the required type,
        False otherwise"""
    return required_type in provider.outputAnnotatorType


def get_embedding_provider_stage_for_consumer(
    consumer: Finisher, pipeline: PipelineLike
) -> Any:
    """Get the embedding provider stage for a consumer
    :param consumer: The consumer to get the embedding provider stage for
    :param pipeline: The pipeline to get the embedding provider stage from
    :return: The embedding provider stage for the consumer
    """
    stages = get_stages(pipeline)
    #   consumer_storage_ref = consumer.getStorageRef()
    storage_ref_match = lambda x, y: x.getStorageRef() == y.getStorageRef()
    for s in stages:
        if (
            hasattr(s, "outputAnnotatorType")
            and is_provider_match(s, "word_embeddings")
            and hasattr(s, "getStorageRef")
            and storage_ref_match(s, consumer)
            and hasattr(s, "getOutputCol")
            and is_col_match(s, consumer)
        ):
            return s


def get_text_provider_stage_for_consumer(
    consumer: Finisher, pipeline: PipelineLike
) -> Any:
    """Get the text provider stage for a consumer
    :param consumer: The consumer to get the text provider stage for
    :param pipeline: The pipeline to get the text provider stage from
    :return: The text provider stage for the consumer"""
    stages = get_stages(pipeline)
    for s in stages:
        if (
            hasattr(s, "outputAnnotatorType")
            and is_provider_match(s, "document")
            and hasattr(s, "getOutputCol")
            and is_col_match(s, consumer)
        ):
            return s


def get_token_provider_stage_for_consumer(
    consumer: Finisher, pipeline: PipelineLike
) -> Any:
    """
    Get the token provider stage for a consumer
    :param consumer: The consumer to get the token provider stage for
    :param pipeline: The pipeline to get the token provider stage from
    :return: The token provider stage for the consumer
    """
    stages = get_stages(pipeline)
    for s in stages:
        if (
            hasattr(s, "outputAnnotatorType")
            and is_provider_match(s, "token")
            and hasattr(s, "getOutputCol")
            and is_col_match(s, consumer)
        ):
            return s


def find_ner_model(pipeline: PipelineLike) -> Any:
    """Find the NER model in a pipeline
    :param pipeline: The pipeline to find the NER model in
    :return: The NER model in the pipeline"""
    stages = get_stages(pipeline)
    stages_str = [stage.__repr__().lower() for stage in stages]
    # Common keywords associated with NER models
    ner_keywords = ["ner", "named", "entity", "recognition", "model"]

    # Convert all names in model list to lowercase and remove special characters
    cleaned_model_list = [
        re.sub(r"[^a-zA-Z0-9]", "", model.lower()) for model in stages_str
    ]

    # Search for matches in cleaned model list
    matches = [
        model
        for model in cleaned_model_list
        if all(keyword in model for keyword in ["ner", "model"])
    ]

    # If no matches using both 'ner' and 'model', try finding with just 'ner' keyword
    if not matches:
        matches = [
            model
            for model in cleaned_model_list
            if any(keyword in model for keyword in ner_keywords)
        ]
    if matches:
        # Return first match or None if no matches found
        for match in matches:
            model = stages[cleaned_model_list.index(match)]
            if (
                hasattr(model, "setIncludeAllConfidenceScores")
                and hasattr(model, "setIncludeConfidence")
                and hasattr(model, "getClasses")
            ):
                return model


def get_relevant_cols(
    ner_model: Finisher, pipeline: PipelineLike
) -> Tuple[str, str, str]:
    """Get the relevant columns for a NER model
    :param ner_model: The NER model to get the relevant columns for
    :param pipeline: The pipeline to get the relevant columns from
    :return: A tuple of the relevant columns for the NER model
    """
    # NER Takes in Documen/Token/Embed type cols
    emb = get_embedding_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    doc = get_text_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    tok = get_token_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    return emb, doc, tok
