#Custom Callback to save our embeddings, logits, idx_ids and epoch
import os

import pandas as pd
 
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_metric, load_dataset,Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
from dataquality.integrations.trainer_callback import DQCallback
import dataquality as dq

import os 
os.environ["GALILEO_CONSOLE_URL"] = "https://console.dev.rungalileo.io/"

# 🔭🌕 Galileo logging
dq.login()



dataset_name="emotion"
ds = load_dataset(dataset_name)
num_labels = len(ds["train"].features["label"].names)

model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

metric = load_metric("accuracy")


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"],
                   padding="max_length",max_length=201 ,
                   truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


#We are gonna work on a subset for faster training
train_dataset=ds["train"]
test_dataset=ds["test"]

#'input_ids', 'attention_mask'
encoded_train_dataset = train_dataset.select(range(4000)).map(lambda x: preprocess_function(x,tokenizer),batched=True) 
encoded_test_dataset =  test_dataset.select(range(2000)).map(lambda x: preprocess_function(x,tokenizer),batched=True) 

#Training arguments and training part
metric_name = "accuracy"
batch_size= 16


args = TrainingArguments(
    f"finetuned",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
    report_to="galileo",
    # 🔭🌕 Galileo logging
    remove_unused_columns=False,
    )

# 🔭🌕 Galileo logging
dqcallback = DQCallback()
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    # 🔭🌕 Galileo logging
    callbacks=[dqcallback]
    data_collator=dqcallback._collate_fn,
)

trainer.train()


dq.finish() # 🔭🌕 Galileo logging