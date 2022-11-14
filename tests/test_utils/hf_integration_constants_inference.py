from typing import List
from unittest import mock

import datasets
from transformers import BatchEncoding

UNADJUSTED_TOKEN_DATA_INF = {
    "id": [4, 5, 6, 7, 8],
    "tokens": [
        ["'", "''", "Andrew", "Noble", "''", "'", "-", "fisico", "britannico"],
        ["Eliminato", "al", "4T", "da", "Andy", "Murray", "[3]"],
        ["'", "''", "Suzuki", "''", "'"],
        ["Seekirchen", "am", "Wallersee"],
        ["Ha", "mutuato", "il", "nome", "dal", "capoluogo", "Tarfaya", "."],
    ],
}


ADJUSTED_TOKEN_DATA_INF = {
    "input_ids": [
        [
            101,
            1005,
            1005,
            1005,
            4080,
            7015,
            1005,
            1005,
            1005,
            1011,
            27424,
            11261,
            28101,
            11639,
            11261,
            102,
        ],
        [
            101,
            12005,
            22311,
            3406,
            2632,
            1018,
            2102,
            4830,
            5557,
            6264,
            1031,
            1017,
            1033,
            102,
        ],
        [101, 1005, 1005, 1005, 14278, 1005, 1005, 1005, 102],
        [101, 6148, 4313, 8661, 2572, 23550, 19763, 102],
        [
            101,
            5292,
            14163,
            26302,
            3406,
            6335,
            2053,
            4168,
            17488,
            6178,
            4747,
            19098,
            3995,
            16985,
            7011,
            3148,
            1012,
            102,
        ],
    ],
    "attention_mask": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "text_token_indices": [
        [
            (0, 1),
            (2, 4),
            (2, 4),
            (5, 11),
            (12, 17),
            (18, 20),
            (18, 20),
            (21, 22),
            (23, 24),
            (25, 31),
            (25, 31),
            (32, 42),
            (32, 42),
            (32, 42),
        ],
        [
            (0, 9),
            (0, 9),
            (0, 9),
            (10, 12),
            (13, 15),
            (13, 15),
            (16, 18),
            (19, 23),
            (24, 30),
            (31, 34),
            (31, 34),
            (31, 34),
        ],
        [(0, 1), (2, 4), (2, 4), (5, 11), (12, 14), (12, 14), (15, 16)],
        [(0, 10), (0, 10), (0, 10), (11, 13), (14, 23), (14, 23)],
        [
            (0, 2),
            (3, 10),
            (3, 10),
            (3, 10),
            (11, 13),
            (14, 18),
            (14, 18),
            (19, 22),
            (23, 32),
            (23, 32),
            (23, 32),
            (23, 32),
            (33, 40),
            (33, 40),
            (33, 40),
            (41, 42),
        ],
    ],
    "bpe_tokens": [
        [
            "[CLS]",
            "'",
            "'",
            "'",
            "andrew",
            "noble",
            "'",
            "'",
            "'",
            "-",
            "fis",
            "##ico",
            "brit",
            "##ann",
            "##ico",
            "[SEP]",
        ],
        [
            "[CLS]",
            "eli",
            "##mina",
            "##to",
            "al",
            "4",
            "##t",
            "da",
            "andy",
            "murray",
            "[",
            "3",
            "]",
            "[SEP]",
        ],
        ["[CLS]", "'", "'", "'", "suzuki", "'", "'", "'", "[SEP]"],
        ["[CLS]", "seek", "##ir", "##chen", "am", "waller", "##see", "[SEP]"],
        [
            "[CLS]",
            "ha",
            "mu",
            "##tua",
            "##to",
            "il",
            "no",
            "##me",
            "dal",
            "cap",
            "##ol",
            "##uo",
            "##go",
            "tar",
            "##fa",
            "##ya",
            ".",
            "[SEP]",
        ],
    ],
    "text": [
        "' '' Andrew Noble '' ' - fisico britannico",
        "Eliminato al 4T da Andy Murray [3]",
        "' '' Suzuki '' '",
        "Seekirchen am Wallersee",
        "Ha mutuato il nome dal capoluogo Tarfaya .",
    ],
}


TOKENIZED_DATA = {
    "input_ids": [
        [
            101,
            1005,
            1005,
            1005,
            4080,
            7015,
            1005,
            1005,
            1005,
            1011,
            27424,
            11261,
            28101,
            11639,
            11261,
            102,
        ],
        [
            101,
            12005,
            22311,
            3406,
            2632,
            1018,
            2102,
            4830,
            5557,
            6264,
            1031,
            1017,
            1033,
            102,
        ],
        [101, 1005, 1005, 1005, 14278, 1005, 1005, 1005, 102],
        [101, 6148, 4313, 8661, 2572, 23550, 19763, 102],
        [
            101,
            5292,
            14163,
            26302,
            3406,
            6335,
            2053,
            4168,
            17488,
            6178,
            4747,
            19098,
            3995,
            16985,
            7011,
            3148,
            1012,
            102,
        ],
    ],
    "attention_mask": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
}


BATCH_WORD_IDS = [
    [None, 0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 8, 8, None],
    [None, 0, 0, 0, 1, 2, 2, 3, 4, 5, 6, 6, 6, None],
    [None, 0, 1, 1, 2, 3, 3, 4, None],
    [None, 0, 0, 0, 1, 2, 2, None],
    [None, 0, 1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 7, None],
]
BATCH_TOKENS = [
    [
        "[CLS]",
        "'",
        "'",
        "'",
        "andrew",
        "noble",
        "'",
        "'",
        "'",
        "-",
        "fis",
        "##ico",
        "brit",
        "##ann",
        "##ico",
        "[SEP]",
    ],
    [
        "[CLS]",
        "eli",
        "##mina",
        "##to",
        "al",
        "4",
        "##t",
        "da",
        "andy",
        "murray",
        "[",
        "3",
        "]",
        "[SEP]",
    ],
    ["[CLS]", "'", "'", "'", "suzuki", "'", "'", "'", "[SEP]"],
    ["[CLS]", "seek", "##ir", "##chen", "am", "waller", "##see", "[SEP]"],
    [
        "[CLS]",
        "ha",
        "mu",
        "##tua",
        "##to",
        "il",
        "no",
        "##me",
        "dal",
        "cap",
        "##ol",
        "##uo",
        "##go",
        "tar",
        "##fa",
        "##ya",
        ".",
        "[SEP]",
    ],
]


class BatchTokens:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens


# We mock the BatchEncoding so we do not need to use a real huggingface tokenizer
mock_batch_encoding_inf = BatchEncoding(TOKENIZED_DATA)
mock_batch_encoding_inf.word_ids = lambda batch_index: BATCH_WORD_IDS[batch_index]
mock_batch_encoding_inf._encodings = [BatchTokens(tokens) for tokens in BATCH_TOKENS]

# We mock the tokenizer so we don't need to use the real tokenizer from hf
mock_tokenizer_inf = mock.Mock()
mock_tokenizer_inf.batch_encode_plus.return_value = mock_batch_encoding_inf


# We mock the hf dataset so we don't need to download a dataset from hf
mock_ds_inf = datasets.Dataset.from_dict(UNADJUSTED_TOKEN_DATA_INF)
label_names = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]
