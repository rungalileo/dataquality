import itertools
from enum import Enum, unique
from typing import Dict, List, Optional, Tuple, Union

import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame

from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.dataframe import BaseLoggerInOutFrames
from dataquality.schemas.ner import NERColumns as NERCols
from dataquality.schemas.ner import TaggingSchema
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

    * text_token_indices: Token boundaries of text. List[List[Tuple(int, int)]].
    Used to convert the gold_spans into token level spans internally. For each sample,
    the boundary of a token will contain the start and end character index of word in
    the `text` to which the said token belongs.

    * gold_spans: Gold spans for the text at character level indices.
    The list of spans in a sample with their start and end indexes, and the label.
    Indexes start at 0 and are [inclusive, exclusive) for [start, end) respectively.
    List[List[dict]].

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

        gold_spans: List[List[dict]] = [
            [
                {"start":17, "end":27, "label":"person"}  # "Joe Biden"
            ],
            [
                {"start":0, "end":10, "label":"person"},    # "Joe Biden"
                {"start":30, "end":41, "label":"location"}  # "United States"
            ]
        ]

        text_token_indices: [[(0, 3), (4, 13), (14, 16), (17, 20), (21, 27), (21, 27)],
                [...]]
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
        :param text_token_indices: Token boundaries of text. List[Tuple(int, int)].
        Used to convert the gold_spans into token level spans internally.
        t[0] indicates the start index of the span and t[1] is the end index (exclusive)
        :param gold_spans: The model-level gold spans over the char index of `text`
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

        for sample_id, sample_spans, sample_indices, sample_text in zip(
            self.ids, self.gold_spans, self.text_token_indices, self.text
        ):
            self._validate_sample_spans(sample_spans, sample_indices, sample_text)

            updated_spans = self._extract_gold_spans(sample_spans, sample_indices)

            sample_key = self.logger_config.get_sample_key(Split(self.split), sample_id)
            self.logger_config.gold_spans[sample_key] = [
                (span["start"], span["end"], span["label"]) for span in updated_spans
            ]
            # Unpadded length of the sample. Used to extract true predicted spans
            # which are padded by the model
            self.logger_config.sample_length[sample_key] = len(sample_indices)
            # Flatten the List[Tuple[int,int]] to List[int]
            flattened_indices = list(itertools.chain(*sample_indices))
            self.text_token_indices_flat.append(flattened_indices)

        # Free up the memory, we don't need it anymore
        del self.text_token_indices

        self.validate_metadata(batch_size=text_len)

    def _validate_sample_spans(
        self,
        sample_spans: List[Dict],
        sample_indices: List[Tuple[int, int]],
        sample_text: str,
    ) -> None:
        """Validates spans of a sample"""
        clean_labels = self._clean_labels()
        max_end_idx, max_start_idx = 0, 0
        for span in sample_spans:
            assert span["label"] in clean_labels, (
                f"'{span['label']}' is absent in the provided labels {clean_labels}. "
                f"Set all labels for run with `dataquality.set_labels_for_run`"
            )

            assert isinstance(span, dict), "individual spans must be dictionaries"
            assert "start" in span and "end" in span and "label" in span, (
                "gold_spans must have a 'start', 'end', and 'label', but got "
                f"{span.keys()}"
            )
            assert (
                span["start"] < span["end"]
            ), f"end index must be >= start index, but got {span}"
            max_end_idx = max(span["end"], max_end_idx)
            max_start_idx = max(span["start"], max_start_idx)

        assert max_start_idx <= sample_indices[-1][0], (
            f"span start idx: {max_start_idx}, does not align with provided token "
            f"boundaries {sample_indices}"
        )
        assert max_end_idx <= sample_indices[-1][1], (
            f"span end idx: {max_end_idx}, does not align with provided token "
            f"boundaries {sample_indices}"
        )
        assert max_end_idx <= len(sample_text), (
            f"span end idx: {max_end_idx} "
            f"overshoots text length of {len(sample_text)} {sample_text}"
        )

    def _extract_gold_spans(
        self, gold_spans: List[Dict], token_indices: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        This function converts gold spans that were character indexed into gold spans
            that are token indexed.
        This is done to align with the predicted spans of the model, and compute DEP
        gold_spans = [{'start': 17, 'end': 29, 'label': 'ACTOR'}]
        token_indices = [
            (0, 4), (5, 11), (12, 16), (17, 22), (17, 22), (23, 29), (23, 29)
            ]
        new_gold_spans =  [{'start': 3, 'end': 7, 'label': 'ACTOR'}]
        """
        new_gold_spans: List[Dict] = []
        for span in gold_spans:
            start = span["start"]
            end = span["end"]
            new_start, new_end = None, None
            for token_idx, token in enumerate(token_indices):
                token_start, token_end = token
                if start == token_start and new_start is None:
                    new_start = token_idx
                if end == token_end:
                    new_end = token_idx + 1
            if (
                new_start is not None and new_end is not None
            ):  # Handle edge case of where sentence > allowed_max_length
                new_gold_spans.append(
                    {
                        "start": new_start,
                        "end": new_end,
                        "label": span["label"],
                    }
                )
        assert len(new_gold_spans) == len(gold_spans), (
            f"error in span alignment, "
            f"cannot find all gold spans: "
            f"{gold_spans} in token boundaries: {token_indices}"
        )
        return new_gold_spans

    def _get_input_df(self) -> DataFrame:
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
        df_len = len(self.text)
        inp = dict(
            id=self.ids,
            split=[Split(self.split).value] * df_len,
            text=self.text,
            text_token_indices=pa.array(self.text_token_indices_flat),
            data_schema_version=[__data_schema_version__] * df_len,
            **self.meta,
        )
        df = vaex.from_dict(inp)
        return df

    @classmethod
    def process_in_out_frames(
        cls,
        in_frame: DataFrame,
        out_frame: DataFrame,
        prob_only: bool,
        epoch_or_inf_name: str = None,
    ) -> BaseLoggerInOutFrames:
        """Processes input and output dataframes from logging

        NER is a different case where the input data is logged at the sample level,
        but output data is logged at the span level, so we need to process it
        differently

        We don't have span IDs so we don't need to validate uniqueness
        We don't join the input and output frames
        We do need to split take only the rows from in_frame from this split
        Splits the dataframes into prob, emb, and input data for uploading to minio
        """

        prob, emb, _ = cls.split_dataframe(out_frame, prob_only)
        return BaseLoggerInOutFrames(prob=prob, emb=emb, data=in_frame)

    @classmethod
    def split_dataframe(
        cls, df: DataFrame, prob_only: bool
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the dataframe into logical grouping for minio storage

        NER is a different case, where we store the text samples as "data" and
        all of the span level data is split into only "emb" and "prob". This function
        will only return 2 modified dataframes, where the third is expected to be the
        input data logged by the user
        """
        df_copy = df.copy()
        df_copy["id"] = vaex.vrange(0, len(df_copy), dtype="int64")
        # Separate out embeddings and probabilities into their own files
        prob_cols = [
            NERCols.id.value,
            NERCols.sample_id.value,
            NERCols.split.value,
            NERCols.epoch.value,
            NERCols.is_gold.value,
            NERCols.is_pred.value,
            NERCols.span_start.value,
            NERCols.span_end.value,
            NERCols.gold.value,
            NERCols.pred.value,
            NERCols.data_error_potential.value,
            NERCols.galileo_error_type.value,
        ]
        prob = df_copy[prob_cols]
        emb_cols = (
            [NERCols.id.value] if prob_only else [NERCols.id.value, NERCols.emb.value]
        )
        emb = df_copy[emb_cols]
        return prob, emb, df_copy

    @classmethod
    def validate_labels(cls) -> None:
        """Validates and cleans labels

        For NER, labels will come with their tags and internal values, such as:
        ['[PAD]', '[CLS]', '[SEP]', 'O', 'B-ACTOR', 'I-ACTOR', 'B-YEAR', 'B-TITLE',
        'B-GENRE', 'I-GENRE', 'B-DIRECTOR', 'I-DIRECTOR', 'B-SONG', 'I-SONG', 'B-PLOT',
        'I-PLOT', 'B-REVIEW', 'B-CHARACTER', 'I-CHARACTER', 'B-RATING',
        'B-RATINGS_AVERAGE', 'I-RATINGS_AVERAGE', 'I-TITLE', 'I-RATING', 'B-TRAILER',
        'I-TRAILER', 'I-REVIEW', 'I-YEAR']

        But we want only the true tag values (preserving order):
        ['ACTOR', 'YEAR', 'TITLE', 'GENRE', 'DIRECTOR', 'SONG', 'PLOT', 'REVIEW',
        'CHARACTER', 'RATING', 'RATINGS_AVERAGE', 'TRAILER']
        """
        cls.logger_config.labels = cls._clean_labels()

    @classmethod
    def _clean_labels(cls) -> List[str]:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )
        clean_labels = [
            i.split("-", maxsplit=1)[1]
            for i in cls.logger_config.labels
            if i and cls.is_valid_span_label(i)
        ]
        clean_labels = list(dict.fromkeys(clean_labels))  # Remove dups, keep order
        return clean_labels

    @classmethod
    def is_valid_span_label(cls, label: str) -> bool:
        """Denotes if a span label is valid based on our allowed tagging schemas

        B = Before the sequence
        I = In the sequence
        L/E = Last/Ending character of the sequence
        S/U = Single/Unit element of a sequence
        """
        return (
            label.startswith("B")
            or label.startswith("I")
            or label.startswith("L")
            or label.startswith("E")
            or label.startswith("S")
            or label.startswith("U")
        )

    @classmethod
    def set_tagging_schema(cls, tagging_schema: TaggingSchema) -> None:
        cls.logger_config.tagging_schema = tagging_schema
