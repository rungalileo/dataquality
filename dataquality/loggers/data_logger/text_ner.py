import json
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Tuple, Union

from vaex.dataframe import DataFrame

from dataquality.loggers.data_logger.base_data_logger import BaseGalileoDataLogger
from dataquality.loggers.logger_config.text_ner import text_ner_logger_config
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


@unique
class GalileoDataLoggerAttributes(str, Enum):
    text = "text"
    text_tokenized = "text_tokenized"
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
    * text_tokenized: Tokenized text _as the model sees it_. This tokenization
    should match the input that is fed to the model, created by the user-specific
    tokenizer. This requires tokenizing each individual word instead of the entire
    sentence at once. List[List[List[str]]]
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

        text_inputs: List[str] = [
            "The president is Joe Biden",
            "Joe Biden addressed the United States on Monday"
        ]
        TODO: Update docs for text_token_indicies
        # Taken from the user's tokenizer (at the word level). 2 sentences, 2 elements
        # in the list. Each element is a list of tokens, a token is a list of strings
        text_tokenized: List[List[List[str]]] = [
            [
               # span indexes
               # 0        1       2            3       4        5     6          7
                ["the"], ["pres", "##ident"], ["is"], ["joe"], ["bi", "##den"], ["."]
            ],
            [
               # 0        1     2          3          4         5
                ["joe"], ["bi", "##den"], ["address", "##ed"], ["the"],
               # 6        7        8        9        10      11          12
                ["unite", "##d"], ["state", "##s"], ["on"], ["monday"], ["."]
            ]
        ]
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
            text_inputs, text_tokenized, gold_spans, ids, split
        )

    labels = ["B-PER", "I-PER", "B-LOC", "I-LOC", "O"]
    dataquality.set_labels_for_run(labels_list = labels)

    # One of (IOB2, BIO, IOB, BILOU, BILOES)
    dataquality.set_tagging_schema(tagging_schema: str = "BIO")
    """

    __logger_name__ = "text_ner"
    logger_config = text_ner_logger_config

    def __init__(
        self,
        text: List[str] = None,
        text_token_indicies: List[List[Tuple[int, int]]] = None,
        gold_spans: List[List[Dict]] = None,
        ids: List[int] = None,
        split: str = None,
        meta: Optional[Dict[str, List[Union[str, float, int]]]] = None,
    ) -> None:
        """Create data logger.

        :param text: The raw text inputs for model training. List[str]
        :param text_token_indicies: FIXME: Definition here
        :param gold_spans: The model-level gold spans over the `text_tokenized`
        :param ids: Optional unique indexes for each record. If not provided, will
        default to the index of the record. Optional[List[Union[int,str]]]
        :param split: The split for training/test/validation
        """
        super().__init__(meta)
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.text_token_indicies = (
            text_token_indicies if text_token_indicies is not None else []
        )
        self.gold_spans = gold_spans if gold_spans is not None else []
        self.ids = ids if ids is not None else []
        self.split = split

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataLoggerAttributes.get_valid()

    def log(self) -> None:
        # This will handle validation and logging the sentence level data
        # (see `_get_input_dict()` below)
        super().log()
        # Now we need to handle the span-level dataframe



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

        text_tokenized_len = len(self.text_token_indicies)
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

        new_gold_spans: List[List[Dict]] = []
        for sample_id, sample_spans, sample_indicies in zip(
            self.ids, self.gold_spans, self.text_token_indicies
        ):
            assert len(sample_spans) <= self.logger_config.max_spans, (
                f"Galileo does not support more than {self.logger_config.max_spans} "
                f"spans in a sample input."
            )

            for span in sample_spans:
                assert isinstance(span, dict), "individual spans must be dictionaries"
                assert "start" in span and "end" in span and "label" in span, (
                    "gold_spans must have a 'start', 'end', and 'label', but got "
                    f"{span.keys()}"
                )
                assert (
                    span["start"] <= span["end"]
                ), f"end index must be >= start index, but got {span}"

            updated_spans = self._extract_gold_spans(sample_spans, sample_indicies)
            new_gold_spans.append(updated_spans)

            span_key = self.logger_config.get_span_key(str(self.split), sample_id)
            self.logger_config.gold_spans[span_key] = [
                (span["start"], span["end"], span["label"]) for span in updated_spans
            ]

        self.gold_spans = new_gold_spans

        self.validate_metadata(batch_size=text_len)

    def _extract_gold_spans(
        self, gold_spans: List[Dict], token_indicies: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        gold_spans = [{"start_idx": 21, "end_idx": 32, "label": "nothing"}]
        token_indicies = [(0, 4),  (5, 7), (8, 11), (12, 16), (17, 19), (21, 24), (25, 28), (29, 32)]
        new_gold_spans =  [{'start_idx': 5, 'end_idx': 8, 'label': 'nothing'}]
        """
        new_gold_spans: List[Dict] = []
        for span in gold_spans:
            start_idx = span["start_idx"]
            end_idx = span["end_idx"]
            new_start_idx, new_end_idx = None, None
            for token_idx, tokens in enumerate(token_indicies):
                token_start_idx, token_end_idx = tokens[0], tokens[1]
                if start_idx == token_start_idx:
                    new_start_idx = token_idx
                if end_idx == token_end_idx:
                    new_end_idx = token_idx + 1
            if (
                new_start_idx and new_end_idx
            ):  # Handle edge case of where sentence > allowed_max_length
                new_gold_spans.append(
                    {
                        "start_idx": new_start_idx,
                        "end_idx": new_end_idx,
                        "label": span["label"],
                    }
                )

        return new_gold_spans

    def _get_span_input_dict(self) -> Dict[str, Any]:
        """Extracts the span-level data for logging"""


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
            **self.meta,
            # text_token_indicies=json.dumps(self.text_token_indicies),
            # data_schema_version=__data_schema_version__,
            # gold_spans=json.dumps(self.gold_spans)
            # if self.split != Split.inference.value
            # else None,
        )

    @classmethod
    def split_dataframe(cls, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Splits the singular dataframe into its 3 components

        Gets the probability df, the embedding df, and the "data" df containing
        all other columns
        """
        df_copy = df.copy()
        # Separate out embeddings and probabilities into their own files
        dep_cols = [f"dep_{i}" for i in range(cls.logger_config.max_spans)]
        prob = df_copy[["id"] + dep_cols]
        gold_emb = [f"gold_emb_{i}" for i in range(cls.logger_config.max_spans)]
        pred_emb = [f"pred_emb_{i}" for i in range(cls.logger_config.max_spans)]
        emb = df_copy[["id"] + gold_emb + pred_emb]
        ignore_cols = ["emb", "prob", "gold", "split_id"]
        other_cols = [i for i in df_copy.get_column_names() if i not in ignore_cols]
        data_df = df_copy[other_cols]
        return prob, emb, data_df

    @classmethod
    def validate_labels(cls) -> None:
        assert cls.logger_config.labels, (
            "You must set your config labels before calling finish. "
            "See `dataquality.set_labels_for_run`"
        )

        assert len(cls.logger_config.labels) == cls.logger_config.observed_num_labels, (
            f"You set your labels to be {cls.logger_config.labels} "
            f"({len(cls.logger_config.labels)} labels) but based on training, your "
            f"model is expecting {cls.logger_config.observed_num_labels} labels. "
            f"Use dataquality.set_labels_for_run to update your config labels."
        )
