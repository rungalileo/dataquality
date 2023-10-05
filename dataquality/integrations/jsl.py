from typing import Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, monotonically_increasing_id, row_number, size

import dataquality as dq
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.jsl import (
    PipelineLike,
    add_labels_df,
    align_gold_spans,
    find_ner_model,
    get_relevant_cols,
    pad_df,
    prepare_df_for_dq,
)

chunksize = 512
max_token_length = 512


class JSLProject:
    """
    Represents a project for John Snow Labs for PySpark integration
    """

    def __init__(
        self,
        project_name: str,
        run_name: str,
        task_type: TaskType = TaskType.text_ner,
        url: Optional[str] = None,
        pipeline: Optional[PipelineLike] = None,
        dataset: Optional[DataFrame] = None,
        finish: bool = True,
    ) -> None:
        """Initializes a new project for a John Snow Labs NER pipeline and dataframe.
        It can be execute in the following way:
        ```
        project = JSLProject(project_name, run_name, url="console.domain.com")
        project.watch(pipeline)
        project.evaluate(dataset)
        project.finish()
        ```

        :param project_name: The name of the project
        :param run_name: The name of the run
        :param task_type: The task type of the project (supported: text_ner)
        :param url: The url of the console
        :param pipeline: The pipeline to be evaluated (optional)
        :param dataset: The dataset to be evaluated (optional if pipeline is provided)
        """
        self.project_name = project_name
        self.run_name = run_name
        self.task_type = task_type
        self.url = url
        self.setup()
        if pipeline:
            self.watch(pipeline)
        if dataset and pipeline:
            self.evaluate(dataset, finish=finish)

    def setup(self) -> None:
        if self.url:
            dq.set_console_url(self.url)
        dq.init(
            project_name=self.project_name,
            run_name=self.run_name,
            task_type=self.task_type,
        )
        if self.task_type == TaskType.text_ner:
            dq.set_tagging_schema(TaggingSchema.BIO)

    def watch(self, pipeline: PipelineLike, text_col: str = "text") -> None:
        """
        Watches a pipeline and sets the labels and relevant columns for the project.
        :param pipeline: The pipeline to be evaluated
        :param text_col: The text column of the dataset
        """
        model = find_ner_model(pipeline)
        model.setIncludeAllConfidenceScores(True)
        model.setIncludeConfidence(True)
        self.labels = model.getClasses()
        self.emb_col, _, _ = get_relevant_cols(model, pipeline)
        self.text_col = text_col
        self.pipeline = pipeline
        return dq.set_labels_for_run(self.labels)

    def evaluate(
        self,
        df: DataFrame,
        split: Split = Split.training,
        finish: bool = False,
        pipeline: Optional[PipelineLike] = None,
        batch_size: int = 2048,
        max_token_length: int = 1024,
    ) -> Optional[str]:
        """
        Evaluates a dataset and logs the results to the console.
        :param df: The pyspark dataset to be evaluated
        :param split: The split of the dataset (default: training)
        :param finish: Whether to finish the run after evaluation (default: False)
        :param pipeline: The pipeline to be evaluated (optional if pipeline is provided)
        :param batch_size: The batch size for the evaluation (default: 2048)
        :param max_token_length: The maximum token length for
            the evaluation (default: 1024)
        """
        if pipeline:
            self.watch(pipeline)
        if not self.pipeline:
            raise ValueError("No pipeline found. Please call watch(pipeline) first.")
        df = df.withColumn(
            "id", row_number().over(Window.orderBy(monotonically_increasing_id())) - 1
        )

        df = df.withColumn("token_length", size("token"))
        ds_len = df.count()
        split = Split.training
        chunksize = batch_size
        for chunk in range(0, ds_len, chunksize):
            df_chunk = df.filter(
                (col("id") >= chunk)
                & (col("id") < (chunk + chunksize))
                & (col("token_length") < max_token_length)
            )  # .persist()
            try:
                df_chunk_pred = self.pipeline.transform(df_chunk)
                df_chunk_dq = prepare_df_for_dq(
                    df_chunk_pred, self.labels, self.emb_col
                )
                df_chunk_dq_labels = add_labels_df(df_chunk_dq)
                df_formatted = df_chunk_dq_labels.toPandas()
                df_padded = pad_df(df_formatted)
                ner_start_end = df_padded["ner_start_end"].tolist()
                gold_spans = (
                    df_padded["gold_spans"]
                    .apply(lambda x: list(map(lambda y: y.asDict(), x)))
                    .tolist()
                )

                token_indices = align_gold_spans(gold_spans, ner_start_end)
                dq.log_data_samples(
                    texts=df_padded["text"].tolist(),
                    ids=(df_padded["id"]).tolist(),
                    text_token_indices=token_indices,
                    gold_spans=gold_spans,
                    split=split,
                )
                dq.log_model_outputs(
                    ids=(df_padded["id"]).tolist(),
                    embs=df_padded["embs_padded"].tolist(),
                    logits=df_padded["probs_padded"].tolist(),
                    split=split,
                    epoch=0,
                )
            except Exception as e:
                print(e)
            del df_chunk

        if finish:
            return dq.finish()
        return None

    def finish(self) -> Optional[str]:
        """
        Finishes the run and returns the run id.
        """
        return dq.finish()
