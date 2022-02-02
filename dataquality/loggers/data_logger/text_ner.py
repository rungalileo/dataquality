from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


@unique
class GalileoDataLoggerAttributes(str, Enum):
    text = "text"
    text_token_indices = "text_token_indices"
    text_token_indices_flat = "text_token_indices_flat"
    gold_spans = "gold_spans"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataLoggerAttributes))


class TextNERDataLogger(BaseGalileoDataLogger):
    """
    Class for logging input data/metadata of Text NER models to Galileo.

    * text: The raw text inputs for model training. List[str]
    * text_token_indices: TODO: Nidhi
    * gold_spans: Gold spans for the text_tokenized. The list of spans in a sample with
    their start and end indexes, and the label. This matches the text_tokenized format
    Indexes start at 0 and are [inclusive, exclusive) for [start, end) respectively.
    List[List[dict]].
    NOTE: Max 5 spans per text sample. More than 5 will throw an error
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[int]]
    * split: The split of training/test/validation
    * meta: Dict[str, List]. Any metadata information you want to log at a per sample
    (text input) level. This could be a string (len <= 50), a float or an int.
    Each sample can have up to 50 meta fields.
    # number of samples in the list must be the same length as the number of text
    # samples logged
    Format {"sample_importance": [0.2, 0.5, 0.99, ...]}

    ex:
    .. code-block:: python

        labels = ["B-PER", "I-PER", "B-LOC", "I-LOC", "O"]
        dataquality.set_labels_for_run(labels = labels)

        # One of (IOB2, BIO, IOB, BILOU, BILOES)
        dataquality.set_tagging_schema(tagging_schema: str = "BIO")

        text: List[str] = [
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ]
        TODO: Nidhi update docs for text_token_indices

        # Gold spans, user created. The start and end index of a true span, with the
        # label in question. Each sentence may have 1 or more labels, so we hold a list
        # of spans per sentence. The start and end index reference the unnested list
        # of text_tokenized.
        gold_spans: List[List[dict]] = [
            [
                {"start":4, "end":7, "label":"person"}  # [joe], [bi, ##den]
            ],
            [
                {"start":0, "end":3, "label":"person"},    # [joe], [bi, ##den]
                {"start":6, "end":10, "label":"location"}  # [unite, ##d], [state, ##s]
            ]
        ]
        ids: List[int] = [0, 1]
        split = "training"

        dataquality.log_input_data(
            text=text, text_token_indices=text_token_indices,
            gold_spans=gold_spans, ids=ids, split=split
        )
    """

    DATA_FOLDER_EXTENSION = {"emb": "hdf5", "prob": "hdf5", "data": "arrow"}

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        text: List[str] = None,
        text_token_indices: List[List[Tuple[int, int]]] = None,
        gold_spans: List[List[Dict]] = None,
        ids: List[int] = None,
        split: str = None,
        meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param text_token_indices: TODO: Nidhi add definition here
        :param gold_spans: The model-level gold spans over the `text_tokenized`
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(meta)
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.text_token_indices = (
            text_token_indices if text_token_indices is not None else []
        )
        self.gold_spans = gold_spans if gold_spans is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.text_token_indices_flat: List[List[int]] = []

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataLoggerAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * Text and Labels must both exist (unless split is 'inference' in which case
        labels must be None)
        * Text and Labels must be the same length
        * If ids exist, it must be the same length as text/labels
        :return: None
        """
        super().validate()

        assert self.logger_config.labels, (
            "You must set your labels before logging input data. "
            "See dataquality.set_labels_for_run"
        )

        assert self.logger_config.tagging_schema, (
            "You must set your tagging schema before logging input data. "
            "See dataquality.set_tagging_schema"
        )

        text_tokenized_len = len(self.text_token_indices)
        text_len = len(self.text)
        gold_span_len = len(self.gold_spans)
        id_len = len(self.ids)

        if self.ids:
            assert id_len == text_len, (
                f"Ids exists but are not the same length as text and labels. "
                f"(ids, text) ({id_len}, {text_len})"
            )
        else:
            self.ids = list(range(text_len))

        if self.split == Split.inference.value:
            assert not gold_span_len, "You cannot have labels in your inference split!"
        else:
            assert gold_span_len and text_len, (
                f"Both text and gold spans for your logger must be set, but got"
                f" text:{bool(text_len)}, labels:{bool(text_len)}"
            )

            assert text_len == text_tokenized_len == gold_span_len, (
                f"labels, text, and tokenized text must be the same length, but got"
                f"(labels, text, text_token) ({gold_span_len},{text_len}, "
                f"{text_tokenized_len})"
            )

        for sample_id, sample_spans, sample_indices in zip(
            self.ids, self.gold_spans, self.text_token_indices
        ):
            for span in sample_spans:
                assert isinstance(span, dict), "individual spans must be dictionaries"
                assert "start" in span and "end" in span and "label" in span, (
                    "gold_spans must have a 'start', 'end', and 'label', but got "
                    f"{span.keys()}"
                )
                assert (
                    span["start"] <= span["end"]
                ), f"end index must be >= start index, but got {span}"

            updated_spans = self._extract_gold_spans(sample_spans, sample_indices)

            span_key = self.logger_config.get_span_key(str(self.split), sample_id)
            self.logger_config.gold_spans[span_key] = [
                (span["start"], span["end"], span["label"]) for span in updated_spans
            ]
            # Flatten the List[Tuple[int,int]] to List[int]
            # https://github.com/python/mypy/issues/6040
            flattened_indices = list(sum(sample_indices, ()))  # type: ignore
            self.text_token_indices_flat.append(flattened_indices)

        # Free up the memory, we don't need it anymore
        del self.text_token_indices

        self.validate_metadata(batch_size=text_len)

    def _extract_gold_spans(
        self, gold_spans: List[Dict], token_indicies: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        TODO: Nidhi add description of logic

        gold_spans = [{"start": 21, "end": 32, "label": "nothing"}]
        token_indicies = [
            (0, 4),  (5, 7), (8, 11), (12, 16), (17, 19), (21, 24), (25, 28), (29, 32)
        ]
        new_gold_spans =  [{'start': 5, 'end': 8, 'label': 'nothing'}]
        """
        new_gold_spans: List[Dict] = []
        for span in gold_spans:
            start = span["start"]
            end = span["end"]
            new_start, new_end = None, None
            for token_idx, tokens in enumerate(token_indicies):
                token_start, token_end = tokens[0], tokens[1]
                if start == token_start:
                    new_start = token_idx
                if end == token_end:
                    new_end = token_idx + 1
            if (
                new_start and new_end
            ):  # Handle edge case of where sentence > allowed_max_length
                new_gold_spans.append(
                    {
                        "start": new_start,
                        "end": new_end,
                        "label": span["label"],
                    }
                )

        return new_gold_spans

    def _get_input_dict(self) -> Dict[str, Any]:
        """NER is a special case where we need to log 2 different input data files

        The first is at the sentence level (id, split, text, **meta)
        The second is at the span level. This is because each sentence will have
        an arbitrary number of spans so we wont be able to create a structured
        dataframe (column numbers wont align).

        So the span-level dataframe is a row per span, with a 'sentence_id' linking
        back to the sentence.

        This function will be used for the sentence level, as that enables the parent's
        `log()` function to behave exactly as expected.
        """
        return dict(
            id=self.ids,
            split=self.split,
            text=self.text,
            text_token_indices=pa.array(self.text_token_indices_flat),
            data_schema_version=__data_schema_version__,
            **self.meta,
        )

    @classmethod
    def process_in_out_frames(
        cls, in_frame: DataFrame, out_frame: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Processes input and output dataframes from logging

        NER is a different case where the input data is logged at the sample level,
        but output data is logged at the span level, so we need to process it
        differently

        We don't have span IDs so we don't need to validate uniqueness
        We don't join the input and output frames
        Splits the dataframes into prob, emb, and input data for uploading to minio
        """

        prob, emb, _ = cls.split_dataframe(out_frame)
        return prob, emb, in_frame

    @classmethod
    def split_dataframe(cls, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the dataframe into logical grouping for minio storage

        NER is a different case, where we store the text samples as "data" and
        all of the span level data is split into only "emb" and "prob". This function
        will only return 2 modified dataframes, where the third is expected to be the
        input data logged by the user
        """
        df_copy = df.copy()
        df_copy["id"] = vaex.vrange(0, len(df_copy))
        # Separate out embeddings and probabilities into their own files
        prob_cols = [
            "id",
            "sample_id",
            "split",
            "epoch",
            "is_gold",
            "is_pred",
            "span_start",
            "span_end",
            "gold",
            "pred",
            "data_error_potential",
        ]
        prob = df_copy[prob_cols]
        emb = df_copy[["id", "emb"]]
        return prob, emb, df_copy

    @classmethod
    def validate_labels(cls) -> None:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )
