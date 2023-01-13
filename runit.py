## Local

import os

os.environ['GALILEO_CONSOLE_URL']="http://localhost:8088"
os.environ["GALILEO_USERNAME"]="user@example.com"
os.environ["GALILEO_PASSWORD"]="Th3secret_"


import dataquality as dq
from dataquality.schemas.task_type import TaskType
from dataquality import config 
from uuid import uuid4
import numpy as np
from time import sleep
from tqdm.notebook import tqdm

dq.configure()
dq.init("text_ner", "test-ner-run")


def log_inputs():
    text_inputs = ['what movies star bruce willis', 'show me films with drew barrymore from the 1980s', 'what movies starred both al pacino and robert deniro', 'find me all of the movies that starred harold ramis and bill murray', 'find me a movie with a quote about baseball in it']
    tokens = [[(0, 4), (5, 11), (12, 16), (17, 22), (17, 22), (23, 29), (23, 29)], [(0, 4), (5, 7), (8, 13), (14, 18), (19, 23), (24, 33), (24, 33), (24, 33), (34, 38), (39, 42), (43, 48)], [(0, 4), (5, 11), (12, 19), (20, 24), (25, 27), (28, 34), (28, 34), (28, 34), (35, 38), (39, 45), (39, 45), (46, 52), (46, 52)], [(0, 4), (5, 7), (8, 11), (12, 14), (15, 18), (19, 25), (26, 30), (31, 38), (39, 45), (39, 45), (39, 45), (46, 51), (46, 51), (52, 55), (56, 60), (61, 67), (61, 67), (61, 67)], [(0, 4), (5, 7), (8, 9), (10, 15), (16, 20), (21, 22), (23, 28), (29, 34), (35, 43), (44, 46), (47, 49)]]
    gold_spans = [[{'start': 17, 'end': 29, 'label': 'ACTOR'}], [{'start': 19, 'end': 33, 'label': 'ACTOR'}, {'start': 43, 'end': 48, 'label': 'YEAR'}], [{'start': 25, 'end': 34, 'label': 'ACTOR'}, {'start': 39, 'end': 52, 'label': 'ACTOR'}], [{'start': 39, 'end': 51, 'label': 'ACTOR'}, {'start': 56, 'end': 67, 'label': 'ACTOR'}], []]
    ids = [0, 1, 2, 3, 4]

    labels = ['[PAD]', '[CLS]', '[SEP]', 'O', 'B-ACTOR', 'I-ACTOR', 'B-YEAR', 'B-TITLE', 'B-GENRE', 'I-GENRE', 'B-DIRECTOR', 'I-DIRECTOR', 'B-SONG', 'I-SONG', 'B-PLOT', 'I-PLOT', 'B-REVIEW', 'B-CHARACTER', 'I-CHARACTER', 'B-RATING', 'B-RATINGS_AVERAGE', 'I-RATINGS_AVERAGE', 'I-TITLE', 'I-RATING', 'B-TRAILER', 'I-TRAILER', 'I-REVIEW', 'I-YEAR']
    dq.set_labels_for_run(labels)
    dq.set_tagging_schema("BIO")
    dq.log_data_samples(texts=text_inputs, text_token_indices=tokens, ids=ids, gold_spans=gold_spans, split="training")
    dq.log_data_samples(texts=text_inputs, text_token_indices=tokens, ids=ids, gold_spans=gold_spans, split="validation")
    dq.log_data_samples(texts=text_inputs, text_token_indices=tokens, ids=ids, gold_spans=gold_spans, split="test")

def log_outputs():
    num_classes = 28
    embs = [np.random.rand(119, 768) for _ in range(5)]
    logits= [np.random.rand(119, 28) for _ in range(5)]                                      
    ids= list(range(5))
    for epoch in tqdm(range(3)):
        for split in ["training"]:#, "test", "validation"]:
            dq.log_model_outputs(
                embs=embs, logits=logits, ids=ids, split=split, epoch=epoch
            )
    
def finish():
    dq.finish()
    
    
def runit():
    log_inputs()
    log_outputs()
    finish()
    
runit()
