from typing import List
from unittest import mock

import datasets
from transformers import BatchEncoding

UNADJUSTED_TOKEN_DATA = {
    "tokens": [
        ["'", "''", "Andrew", "Noble", "''", "'", "-", "fisico", "britannico"],
        ["Eliminato", "al", "4T", "da", "Andy", "Murray", "[3]"],
        ["'", "''", "Suzuki", "''", "'"],
        ["Seekirchen", "am", "Wallersee"],
        ["Ha", "mutuato", "il", "nome", "dal", "capoluogo", "Tarfaya", "."],
    ],
    "ner_tags": [
        [0, 0, 1, 2, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 1, 2, 0],
        [0, 0, 3, 0, 0],
        [5, 6, 6],
        [0, 0, 0, 0, 0, 0, 5, 0],
    ],
    "langs": [
        ["it", "it", "it", "it", "it", "it", "it", "it", "it"],
        ["it", "it", "it", "it", "it", "it", "it"],
        ["it", "it", "it", "it", "it"],
        ["it", "it", "it"],
        ["it", "it", "it", "it", "it", "it", "it", "it"],
    ],
    "spans": [
        ["PER: Andrew Noble", "LOC: britannico"],
        ["PER: Andy Murray"],
        ["ORG: Suzuki"],
        ["LOC: Seekirchen am Wallersee"],
        ["LOC: Tarfaya"],
    ],
}

ADJUSTED_TOKEN_DATA = {
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
    "labels": [
        [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0],
        [0, 5, 6, 6, 6, 6, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 6, 0, 0],
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
    "gold_spans": [
        [
            {"end": 17, "label": "PER", "start": 5},
            {"end": 42, "label": "LOC", "start": 32},
        ],
        [{"end": 30, "label": "PER", "start": 19}],
        [{"end": 11, "label": "ORG", "start": 5}],
        [{"end": 23, "label": "LOC", "start": 0}],
        [{"end": 40, "label": "LOC", "start": 33}],
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


class BIOSequence:
    gold_sequences = [
        ["B-PER", "I-PER", "O", "I-PER", "B-MISC", "I-MISC", "B-PER"],
        ["O", "B-PER", "I-ORG", "O", "I-ORG"],
        ["B-PER", "O", "B-ORG", "I-ORG", "B-ORG", "I-ORG", "B-ORG", "O", "O"],
        ["B-PER", "I-PER", "I-ORG"],
        ["B-PER", "I-PER", "I-PER", "I-PER"],
    ]
    gold_spans = [
        [
            {"start": 0, "end": 2, "label": "PER"},
            {"start": 4, "end": 6, "label": "MISC"},
            {"start": 6, "end": 7, "label": "PER"},
        ],
        [{"start": 1, "end": 2, "label": "PER"}],
        [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 2, "end": 4, "label": "ORG"},
            {"start": 4, "end": 6, "label": "ORG"},
            {"start": 6, "end": 7, "label": "ORG"},
        ],
        [{"start": 0, "end": 2, "label": "PER"}],
        [{"start": 0, "end": 4, "label": "PER"}],
    ]


class BIOESSequence:
    gold_sequences = [
        ["B-PER", "I-PER", "E-PER", "I-PER", "B-MISC", "I-MISC", "S-PER"],
        ["O", "S-PER", "I-ORG", "O", "I-ORG"],
        ["S-PER", "O", "B-ORG", "I-ORG", "I-ORG", "E-ORG", "B-ORG", "O", "O"],
        ["B-PER", "I-PER", "I-ORG"],
        ["B-PER", "I-PER", "I-PER", "I-PER"],
    ]
    gold_spans = [
        [
            {"start": 0, "end": 3, "label": "PER"},
            {"start": 6, "end": 7, "label": "PER"},
        ],
        [{"start": 1, "end": 2, "label": "PER"}],
        [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 2, "end": 6, "label": "ORG"},
        ],
        [],
        [],
    ]


class BILOUSequence:
    gold_sequences = [
        ["B-PER", "I-PER", "L-PER", "I-PER", "B-MISC", "I-MISC", "U-PER"],
        ["O", "U-PER", "I-ORG", "O", "I-ORG"],
        ["U-PER", "O", "B-ORG", "I-ORG", "I-ORG", "L-ORG", "B-ORG", "O", "O"],
        ["B-PER", "I-PER", "I-ORG"],
        ["B-PER", "I-PER", "I-PER", "I-PER"],
    ]
    gold_spans = [
        [
            {"start": 0, "end": 3, "label": "PER"},
            {"start": 6, "end": 7, "label": "PER"},
        ],
        [{"start": 1, "end": 2, "label": "PER"}],
        [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 2, "end": 6, "label": "ORG"},
        ],
        [],
        [],
    ]


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
mock_batch_encoding = BatchEncoding(TOKENIZED_DATA)
mock_batch_encoding.word_ids = lambda batch_index: BATCH_WORD_IDS[batch_index]
mock_batch_encoding._encodings = [BatchTokens(tokens) for tokens in BATCH_TOKENS]

# We mock the tokenizer so we don't need to use the real tokenizer from hf
mock_tokenizer = mock.Mock()
mock_tokenizer.batch_encode_plus.return_value = mock_batch_encoding

# We mock the hf dataset so we don't need to download a dataset from hf
mock_ds = datasets.Dataset.from_dict(UNADJUSTED_TOKEN_DATA)
tag_names = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]
mock_ds.features["ner_tags"].feature.names = tag_names
