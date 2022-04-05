#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import dataquality
from dataquality.schemas.split import Split
from dataquality.clients.api import ApiClient

# os.environ["GALILEO_API_URL"] = "https://api.dev.rungalileo.io"
# os.environ["GALILEO_USERNAME"] = "galileo@rungalileo.io"
# os.environ["GALILEO_PASSWORD"] = "A11a1una!"

# project_id = "44345019-4a66-43a8-800d-7d1b6a56379e"
# run_id = "a01e63e4-6718-4770-a0e4-eb349fb8ec5b"
# project_name = "original_ner_pytorch"
# run_name = "MIT_movies_BIO"

run_samples = "/Users/nikita/Downloads/Involve_debugging/text/text_samples.hdf5"
run_spans = "/Users/nikita/Downloads/Involve_debugging/data/data.hdf5"

# api_client = ApiClient()
# dataquality.configure()
# dataquality.login()


# In[3]:


import vaex
samples = vaex.open(run_samples)
spans = vaex.open(run_spans)


# In[4]:


samples[0:5]


# In[5]:


df = samples.join(spans, left_on='id', right_on='sample_id', how='inner', allow_duplication=True)
print(df.shape, spans.shape, samples.shape)
df = df.to_pandas_df(['text', 'is_gold','is_pred', 'span_start', 'span_end', 'gold_str', 'pred_str', 'span_text', 'sample_id'])


# In[6]:


df


# In[7]:


df[df['is_gold']!=df['is_pred']]


# In[8]:


gold_spans = spans[spans['is_gold']==True]
gold_spans_pd = gold_spans.to_pandas_df(['gold_str', 'pred_str'])
gold_spans_pd['gold_str'].hist(figsize=(20, 3))
gold_spans_pd.shape


# In[9]:


spans[spans['is_pred']==True]


# In[10]:


pred_spans = spans[spans['is_pred']==True]
pred_spans_pd = pred_spans.to_pandas_df(['gold_str', 'pred_str'])
pred_spans_pd['pred_str'].hist(figsize=(20, 3))
pred_spans_pd.shape


# In[11]:


# Run F1 using spacy (with and without Galileo)
    # If spacy.F1==0 in both: model issue (Involve to handle)
    # If spacy.F1==0 in Galileo: watch(nlp) is mutating the model OR ensure model was logged correctly (logging code verify)

# Internally test: use F1 of spacy (without Galileo), and use F1 of spacy (with Galileo) on MIT movies
    # In the Galileo run, compute the F1 using spacy utility

# Questions: 1. Is the F1 without Galileo observed on train or val data?
#


# In[12]:


#TODO: Add micro F1
precision = df[df['is_gold']==df['is_pred']].shape[0]/df[df['is_pred']==True].shape[0]
recall = df[df['is_gold']==df['is_pred']].shape[0]/df[df['is_gold']==True].shape[0]
recall, precision


# In[13]:


df_interest = df[['span_start', 'span_end', 'gold_str', 'text', 'sample_id']]
df_list = df_interest.values.tolist()

TRAIN_DATA = []

sample_map = {}

for x in df_list:
    sample_id = x[4]
    if sample_id in sample_map.keys():
        # add to the entities list within the map entry
        row_tuple = sample_map[sample_id]
        entities_map = row_tuple[1]
        entities_list = entities_map['entities']
        # get current entity
        current_entity = (x[0], x[1], x[2])
        entities_list.append(current_entity)
    else:
        entities_list = [(x[0], x[1], x[2])]
        entities_map = {'entities': entities_list}
        text = x[3]
        row_tuple = (text, entities_map)
        TRAIN_DATA.append(row_tuple)
        sample_map[sample_id] = row_tuple


# In[14]:


TRAIN_DATA


# In[15]:


# !pip install --upgrade dataquality
#!pip install --upgrade spacy==3.2.1
#!pip install ipdb
import json
import logging
import spacy
from spacy.tokens import Doc
from spacy.scorer import Scorer
import numpy as np
import random
import en_core_web_sm # Download language model
import yaml
import pandas as pd
import re
import urllib
import argparse
import sys
import os
from spacy.util import minibatch
import io
import pickle
from tqdm.auto import tqdm
from spacy.training import Example
spacy.prefer_gpu()
from pathlib import Path
import scrubadub
import mlflow
import mlflow.spacy
random.seed(0)
spacy.util.fix_random_seed()

import os
os.environ['GALILEO_CONSOLE_URL'] = "https://console.preprod.rungalileo.io"
os.environ["GALILEO_USERNAME"] = "atin@rungalileo.io"
os.environ["GALILEO_PASSWORD"] = "th3Secret_"

# os.environ['GALILEO_CONSOLE_URL'] = "https://console.dev.rungalileo.io"
# os.environ["GALILEO_USERNAME"] = "galileo@rungalileo.io"
# os.environ["GALILEO_PASSWORD"] = "A11a1una!"

from dataquality.core.integrations.spacy import watch, log_input_examples
import dataquality


# In[ ]:


def train_model(train_data):
    USE_GALILEO = True

    minibatch_size = 1
    num_epochs = 3

    """Model Definition"""

    model = None
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')


    if USE_GALILEO:
        dataquality.configure()
        dataquality.login()
        dataquality.init(
            task_type="text_ner",
            project_name="debugging_involve_low_f1",
            run_name="nikita_run"
        )

    def make_examples(data):
        examples = list()
        for text, annotations in tqdm(data):
                doc = nlp.make_doc(text)
                examples.append(Example.from_dict(doc, annotations))
        return examples

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        training_examples = make_examples(train_data)
        val_examples = make_examples(train_data)
        optimizer = nlp.initialize(lambda: training_examples) # alter begin_training()/resume_training()

        if USE_GALILEO:
            watch(nlp)
            log_input_examples(training_examples, split="training")
            log_input_examples(val_examples, split="validation")

        for i in range(num_epochs):

            if USE_GALILEO:
                dataquality.set_epoch(i)
                dataquality.set_split("training")
            if i == num_epochs - 1:
                print("debug")

            losses = {}
            t_ex = 0
            batches = minibatch(training_examples, minibatch_size)
            print("Starting training")
            for batch in tqdm(batches):
                    nlp.update(
                    batch,
                    drop=0.3,
                    sgd=optimizer,
                    losses=losses)
                    t_ex+=1

            if USE_GALILEO:
                dataquality.set_split("validation")

            print("Starting validation")
            results = nlp.evaluate(val_examples)
            print(results)
            print(f"Training Loss: {losses}")
            print(f"Epoch {i} Complete")

    if USE_GALILEO:
        dataquality.finish()

    return nlp

train_data = TRAIN_DATA
nlp = train_model(train_data)


# In[29]:

#
# num_predictions = 0
# for text, annotations in tqdm(train_data):
#     result = nlp(text)
#     for ent in result.ents:
#         print(ent)
#         print(ent.label_)
#     num_predictions += len(result.ents)
#
# print(num_predictions)


# In[17]:




