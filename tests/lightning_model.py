import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_20newsgroups
from transformers import (
    AutoModel,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig

NUM_RECORDS = 23


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

        # Shuffle some percentage of the training dataset
        # to force create mislabeled samples
        if split == "training":
            self.dataset = introduce_label_errors(self.dataset, "label", 11)

        #
        # 🔭 Logging Inputs with Galileo!
        #
        self.gconfig = GalileoDataConfig(
            text=self.dataset["text"], labels=self.dataset["label"]
        )

        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.encodings = tokenizer(
            self.dataset["text"].tolist(), truncation=True, padding=True
        )

    def __getitem__(self, idx):
        x = torch.tensor(self.encodings["input_ids"][idx])
        attention_mask = torch.tensor(self.encodings["attention_mask"][idx])
        y = self.dataset["label"][idx]
        return idx, x, attention_mask, y

    def __len__(self):
        return len(self.dataset)


class LightningDistilBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=DistilBertConfig(num_labels=20)
        )
        self.feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased")

    def forward(self, x, attention_mask, x_idxs, epoch, split):
        out = self.model(x, attention_mask=attention_mask)
        log_probs = F.log_softmax(out.logits, dim=1)
        probs = F.softmax(out.logits, dim=1)
        encoded_layers = self.feature_extractor(x, return_dict=False)[0]

        # Logging with Galileo!
        self.g_model_config = GalileoModelConfig(
            emb=[i[0] for i in encoded_layers.tolist()],
            probs=probs,
            ids=x_idxs,
        )

        return log_probs

    def training_step(self, batch, batch_idx):
        """Model training step."""
        x_idxs, x, attention_mask, y = batch
        log_probs = self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
            epoch=self.current_epoch,
            split="training",
        )
        loss = F.nll_loss(log_probs, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """Model validation step."""
        x_idxs, x, attention_mask, y = batch
        log_probs = self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
            epoch=self.current_epoch,
            split="validation",
        )
        loss = F.nll_loss(log_probs, y)
        return loss

    def test_step(self, batch, batch_idx):
        """Model test step."""
        x_idxs, x, attention_mask, y = batch
        log_probs = self(
            x=x,
            attention_mask=attention_mask,
            x_idxs=x_idxs,
            epoch=self.current_epoch,
            split="test",
        )
        loss = F.nll_loss(log_probs, y)
        return loss

    def configure_optimizers(self):
        """Model optimizers."""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5
        )


model = LightningDistilBERT()
