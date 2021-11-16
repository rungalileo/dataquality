import dataquality

dataquality.login()

dataquality.init()

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import DistilBertTokenizerFast

#
# 🔭 Use the GalileoModelConfig and GalileoDataConfig to keep track of Galileo metrics for logging
#
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig


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
        self.dataset = self.dataset

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


import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from transformers import (
    AutoModel,
    DistilBertConfig,
    DistilBertForSequenceClassification,
)


class LightningDistilBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", config=DistilBertConfig(num_labels=20)
        )
        self.feature_extractor = AutoModel.from_pretrained("distilbert-base-uncased")
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x, attention_mask, x_idxs, epoch, split):
        out = self.model(x, attention_mask=attention_mask)
        log_probs = F.log_softmax(out.logits, dim=1)
        probs = F.softmax(out.logits, dim=1)
        encoded_layers = self.feature_extractor(x, return_dict=False)[0]

        #
        # 🔭 Logging model outputs with Galileo!
        #
        self.g_model_config = GalileoModelConfig(
            emb=[i[0] for i in encoded_layers.tolist()],
            probs=probs.tolist(),
            ids=x_idxs.tolist(),
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
        self.train_acc(torch.argmax(log_probs, 1), y)
        self.log("train_acc", self.train_acc, prog_bar=True)
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
        self.val_acc(torch.argmax(log_probs, 1), y)
        self.log("val_acc", self.val_acc, prog_bar=True)
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
        self.test_acc(torch.argmax(log_probs, 1), y)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Model optimizers."""
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5
        )


# Use the PyTorch Lightning Callback to log data to Galileo
from dataquality.core.integrations.lightning import DataQualityCallback

model = LightningDistilBERT()

train_dataloader = torch.utils.data.DataLoader(
    NewsgroupDataset("training"), batch_size=32, shuffle=True
)
validation_dataloader = torch.utils.data.DataLoader(
    NewsgroupDataset("validation"), batch_size=32, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    NewsgroupDataset("test"), batch_size=32, shuffle=True
)

trainer = pl.Trainer(
    max_epochs=5, num_sanity_val_steps=0, callbacks=[(DataQualityCallback())]
)

trainer.fit(model, train_dataloader, validation_dataloader)
trainer.test(model, test_dataloader)

dataquality.set_labels_for_run(list(map(str, range(model.model.num_labels))))

dataquality.finish()
