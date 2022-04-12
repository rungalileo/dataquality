TEXT_INPUTS = [
    "what movies star bruce willis",
    "show me films with drew barrymore from the 1980s",
    "what movies starred both al pacino and robert deniro",
    "find me all of the movies that starred harold ramis and bill murray",
    "find me a movie with a quote about baseball in it",
]


TEXT_TOKENS = [
    [(0, 4), (5, 11), (12, 16), (17, 22), (17, 22), (23, 29), (23, 29)],
    [
        (0, 4),
        (5, 7),
        (8, 13),
        (14, 18),
        (19, 23),
        (24, 33),
        (24, 33),
        (24, 33),
        (34, 38),
        (39, 42),
        (43, 48),
    ],
    [
        (0, 4),
        (5, 11),
        (12, 19),
        (20, 24),
        (25, 27),
        (28, 34),
        (28, 34),
        (28, 34),
        (35, 38),
        (39, 45),
        (39, 45),
        (46, 52),
        (46, 52),
    ],
    [
        (0, 4),
        (5, 7),
        (8, 11),
        (12, 14),
        (15, 18),
        (19, 25),
        (26, 30),
        (31, 38),
        (39, 45),
        (39, 45),
        (39, 45),
        (46, 51),
        (46, 51),
        (52, 55),
        (56, 60),
        (61, 67),
        (61, 67),
        (61, 67),
    ],
    [
        (0, 4),
        (5, 7),
        (8, 9),
        (10, 15),
        (16, 20),
        (21, 22),
        (23, 28),
        (29, 34),
        (35, 43),
        (44, 46),
        (47, 49),
    ],
]

GOLD_SPANS = [
    [{"start": 17, "end": 29, "label": "ACTOR"}],
    [
        {"start": 19, "end": 33, "label": "ACTOR"},
        {"start": 43, "end": 48, "label": "YEAR"},
    ],
    [
        {"start": 25, "end": 34, "label": "ACTOR"},
        {"start": 39, "end": 52, "label": "ACTOR"},
    ],
    [
        {"start": 39, "end": 51, "label": "ACTOR"},
        {"start": 56, "end": 67, "label": "ACTOR"},
    ],
    [],
]

LABELS = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "O",
    "B-ACTOR",
    "I-ACTOR",
    "B-YEAR",
    "B-TITLE",
    "B-GENRE",
    "I-GENRE",
    "B-DIRECTOR",
    "I-DIRECTOR",
    "B-SONG",
    "I-SONG",
    "B-PLOT",
    "I-PLOT",
    "B-REVIEW",
    "B-CHARACTER",
    "I-CHARACTER",
    "B-RATING",
    "B-RATINGS_AVERAGE",
    "I-RATINGS_AVERAGE",
    "I-TITLE",
    "I-RATING",
    "B-TRAILER",
    "I-TRAILER",
    "I-REVIEW",
    "I-YEAR",
]

NER_INPUT_DATA = {
    "my_text": TEXT_INPUTS,
    "my_spans": GOLD_SPANS,
    "my_id": list(range(len(TEXT_TOKENS))),
    "text_tokens": TEXT_TOKENS,
}

NER_INPUT_ITER = [
    {
        "my_text": NER_INPUT_DATA["my_text"][i],
        "my_spans": NER_INPUT_DATA["my_spans"][i],
        "my_id": NER_INPUT_DATA["my_id"][i],
        "text_tokens": NER_INPUT_DATA["text_tokens"][i],
    }
    for i in list(range(len(TEXT_TOKENS)))
]

NER_INPUT_TUPLES = list(zip(TEXT_INPUTS, GOLD_SPANS, list(range(5)), TEXT_TOKENS))
NER_INPUT_TUPLES = [tuple(i) for i in NER_INPUT_TUPLES]
