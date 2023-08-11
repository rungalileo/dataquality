import copy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pyspark.sql.functions import col, udf
from pyspark.sql import Row, DataFrame
from sparknlp.internal import AnnotatorTransformer
from sparknlp.base.finisher import Finisher
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from sparknlp.pretrained import (
    LightPipeline,
    Pipeline,
    PipelineModel,
    PretrainedPipeline,
)


MAX_TOKEN_LEN = 512

PipelineLike = Union[Pipeline, LightPipeline, PipelineModel, PretrainedPipeline]


def extract_begin_end(token_list: list[Row]) -> List[List[int]]:
    # For some labels begin is equal to end
    return [[row.begin, row.end + 1] for row in token_list]


def extract_embeddings(embeddings_list: list[Row]) -> List[List[float]]:
    return [row.embeddings for row in embeddings_list]


def extract_ner_result(ner_list: List[Row]) -> List:
    return [row.result for row in ner_list]


def extract_ner_gold_spans(ner_list: List[Row]) -> List[dict]:
    spans = []
    current_span = None
    for row in ner_list:
        # Check if the row is a beginning of a named entity (B-)
        if "B-" in row.result:
            # If there is an existing span, we add it to the list
            if current_span:
                spans.append(current_span)
            # Start a new span
            current_span = {
                "start": row.begin,
                "end": row.end + 1,
                "label": row.result.split("-")[-1],
            }

        # Check if the row is inside a named entity (I-) and there's an active span
        elif "I-" in row.result and current_span:
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


def convert_ner_to_list(ner_list: List[Row], labels: List[str]) -> List[List[float]]:
    return [[float(row.metadata.get(tag, "0")) for tag in labels] for row in ner_list]


def prepare_df_for_dq(
    df: DataFrame, labels: List[str], embeddings_col="embeddings"
) -> DataFrame:
    embeddings_udf = udf(extract_embeddings, ArrayType(ArrayType(FloatType())))
    convert_ner_to_list_udf = udf(
        convert_ner_to_list, ArrayType(ArrayType(FloatType()))
    )

    df = df.withColumn(
        embeddings_col, embeddings_udf(col(embeddings_col))
    )  # Do select instead
    df = df.withColumn(
        "ner_list", convert_ner_to_list_udf(col("ner"), labels)
    )  # Do select instead
    df = df.drop("document").drop("sentence").drop("entities").drop("token")
    return df


def add_labels_df(df: DataFrame) -> DataFrame:
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
    df: pd.DataFrame, emb_col: str = "embeddings", label_col: str = "ner_list"
) -> pd.DataFrame:
    global MAX_TOKEN_LEN
    max_length = MAX_TOKEN_LEN
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
    gold_spans: List[Dict], token_boundaries_in: List[List[int]]
) -> List[List[int]]:
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
            "pipe must be one of Pipeline, LightPipeline, PipelineModel, PretrainedPipeline"
        )
    return stages


def get_stage_of_class(
    pipe: PipelineLike, classes_to_filter: Union[List, Tuple]
) -> List[Any]:
    # Get all stages of pipeline which are one of the classes in classes_to_filter
    if isinstance(classes_to_filter, list):
        classes_to_filter = tuple(classes_to_filter)

    stages = get_stages(pipe)
    filterd_stages = [s for s in stages if isinstance(s, classes_to_filter)]
    return filterd_stages


def is_col_match(provider: AnnotatorTransformer, consumer: Finisher) -> bool:
    return provider.getOutputCol() in consumer.getInputCols()


def is_provider_match(provider: AnnotatorTransformer, required_type: str) -> bool:
    return required_type in provider.outputAnnotatorType


def get_embedding_provider_stage_for_consumer(
    consumer: Finisher, pipeline: PipelineLike
) -> Any:
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
    stages = get_stages(pipeline)
    for s in stages:
        if (
            hasattr(s, "outputAnnotatorType")
            and is_provider_match(s, "token")
            and hasattr(s, "getOutputCol")
            and is_col_match(s, consumer)
        ):
            return s


import re


def find_ner_model(pipeline: PipelineLike) -> Any:
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
            model = stages[cleaned_model_list.index(matches)]
            if hasattr(model, "setIncludeAllConfidenceScores") and hasattr(
                model, "setIncludeConfidence"
            ):
                return model


def get_relevant_cols(
    ner_model: Finisher, pipeline: PipelineLike
) -> Tuple[str, str, str]:
    # NER Takes in Documen/Token/Embed type cols
    emb = get_embedding_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    doc = get_text_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    tok = get_token_provider_stage_for_consumer(ner_model, pipeline).getOutputCol()
    return emb, doc, tok
