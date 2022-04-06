import dataquality
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
import os

os.environ['GALILEO_CONSOLE_URL'] = "https://127.0.0.1:8088"
# os.environ['GALILEO_CONSOLE_URL'] = "https://console.dev.rungalileo.io"
os.environ["GALILEO_USERNAME"] = "galileo@rungalileo.io"
os.environ["GALILEO_PASSWORD"] = "A11a1una!"

if __name__ == '__main__':
    dataquality.configure()
    dataquality.login()
    dataquality.init("text_classification", "hackathon", "explainatron_metadata")

    newsgroups = fetch_20newsgroups(subset="train",
                                    remove=('headers', 'footers', 'quotes'))
    dataset = pd.DataFrame()
    dataset["text"] = newsgroups.data
    label_ind = newsgroups.target_names
    dataset["label"] = [label_ind[i] for i in newsgroups.target]
    dataset = dataset[:100]
    dataquality.log_input_data(text=dataset['text'], labels=dataset['label'],
                               split="training", ids=list(range(len(dataset["text"]))))

    from random import choice
    from string import ascii_uppercase
    dataquality.set_labels_for_run(list(set(dataset["label"])))


    num_rows = len(dataset)
    dataquality.set_split("training")

    for epoch in range(5):
        dataquality.set_epoch(epoch)
        emb = np.random.rand(num_rows, 40)
        logits = np.random.rand(num_rows, len(set(dataset["label"])))
        ids = list(range(num_rows))
        dataquality.log_model_outputs(emb=emb, logits=logits, ids=ids, meta={"explainatron_meta": ["please_work"]*num_rows})

    dataquality.finish()
