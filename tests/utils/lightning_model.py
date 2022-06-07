from random import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.datasets import fetch_20newsgroups

import dataquality

NUM_RECORDS = 23


dataquality.config.task_type = "text_classification"


def introduce_label_errors(
    df: pd.DataFrame, column: str, shuffle_percent: int
) -> pd.DataFrame:
    arr = df[column].values
    shuffle = np.random.choice(
        np.arange(arr.shape[0]),
        round(arr.shape[0] * shuffle_percent / 100),
        replace=False,
    )
    arr[np.sort(shuffle)] = arr[shuffle]
    df[column] = arr
    return df


class NewsgroupDataset(torch.utils.data.Dataset):
    def __init__(self, split: str) -> None:
        newsgroups = fetch_20newsgroups(
            subset="train" if split == "training" else "test",
            remove=("headers", "footers", "quotes"),
        )

        self.dataset = pd.DataFrame()
        self.dataset["text"] = newsgroups.data
        self.dataset["label"] = newsgroups.target
        self.dataset = self.dataset[:NUM_RECORDS]
        self.glogger = dataquality.get_data_logger()

        # Shuffle some percentage of the training dataset
        # to force create mislabeled samples
        if split == "training":
            self.dataset = introduce_label_errors(self.dataset, "label", 11)

        #
        # 🔭 Logging Inputs with Galileo!
        #
        self.glogger.texts = self.dataset["text"]
        self.glogger.labels = self.dataset["label"]
        self.glogger.ids = list(range(len(self.dataset)))

    def __getitem__(self, idx):
        x = torch.tensor(np.random.randint(0, 100, size=50))
        attention_mask = x
        y = self.dataset["label"][idx]
        return idx, x, attention_mask, y

    def __len__(self):
        return len(self.dataset)


class TorchDistilBERTTorch(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask, x_idxs):
        # Fake model building for less memory usage
        logits = [[random() for _ in range(5)] for _ in range(NUM_RECORDS)]
        embs = [[random() for _ in range(10)] for _ in range(NUM_RECORDS)]

        # Logging with Galileo!
        self.glogger = dataquality.get_model_logger()(
            embs=embs, logits=logits, ids=x_idxs
        )

        return 0


class LightningDistilBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask, x_idxs):
        # Fake model building for less memory usage
        logits = [[random() for _ in range(5)] for _ in range(NUM_RECORDS)]
        embs = [[random() for _ in range(10)] for _ in range(NUM_RECORDS)]

        # Logging with Galileo!
        self.glogger = dataquality.get_model_logger()(
            embs=embs, logits=logits, ids=x_idxs
        )

        return 0

    def training_step(self, batch, batch_idx):
        """Model training step."""
        x_idxs, x, attention_mask, y = batch
        self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
        )

    def validation_step(self, batch, batch_idx):
        """Model validation step."""
        x_idxs, x, attention_mask, y = batch
        self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
        )

    def test_step(self, batch, batch_idx):
        """Model test step."""
        x_idxs, x, attention_mask, y = batch
        self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
        )

    def configure_optimizers(self):
        """Model optimizers."""


model = LightningDistilBERT()
torch_model = TorchDistilBERTTorch()
