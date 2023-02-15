from typing import Callable, Generator
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
import vaex
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

import dataquality as dq
from dataquality.integrations.torch import unwatch, watch
from dataquality.schemas.task_type import TaskType
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import validate_unique_ids
from tests.conftest import LOCATION

train_iter = iter(AG_NEWS(split="train"))
tokenizer = get_tokenizer("basic_english")
train_iter = AG_NEWS(split="train")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

vocab(["here", "is", "an", "example"])


def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return int(x) - 1


text_pipeline("here is the an example")
label_pipeline("10")

""" Generate data batch and iterator """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


train_iter = AG_NEWS(split="train")
dataloader = DataLoader(
    train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch
)

""" Define the model """


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.classifier = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier.weight.data.uniform_(-initrange, initrange)
        self.classifier.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.classifier(embedded)


train_iter = AG_NEWS(split="train")
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
modeldq = TextClassificationModel(vocab_size, emsize, num_class).to(device)

""" Define functions to train the model and evaluate results"""


def train(dataloader, model):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        # uncommented to speed up testing
        # loss = criterion(predicted_label, label)
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        # uncommented to speed up testing
        # optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch  | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            # uncommented to speed up testing
            # loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


"""Split the dataset and run the model"""

# Hyperparameters
EPOCHS = 2  # epoch
LR = 5  # learning rate
BATCH_SIZE = 8  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

ag_train = to_map_style_dataset(AG_NEWS(split="train"))[:192]
ag_test = to_map_style_dataset(AG_NEWS(split="test"))[192:220]
train_df = pd.DataFrame(ag_train)
test_df = pd.DataFrame(ag_test)
train_df = train_df.reset_index().rename(columns={0: "label", 1: "text", "index": "id"})
train_df["id"] = train_df["id"] + 10000
test_df = test_df.reset_index().rename(columns={0: "label", 1: "text", "index": "id"})
labels = pd.concat([train_df["label"], test_df["label"]]).unique()
labels.sort()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_end_to_end_with_callback(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    global train_df, test_df

    set_test_config(default_task_type=TaskType.text_classification)
    dq.set_labels_for_run(labels)
    # Preprocessing
    dq.log_dataset(train_df, split="train")
    dq.log_dataset(test_df, split="test")

    train_dataloader_dq = DataLoader(
        ag_train,
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True,
        collate_fn=collate_batch,
        # persistent_workers=True,
        # pin_memory=False,
    )
    test_dataloader_dq = DataLoader(
        ag_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    unwatch(modeldq)
    # 🔭🌕 Logging the dataset with Galileo
    watch(
        modeldq,
        [train_dataloader_dq, test_dataloader_dq],
        classifier_layer="classifier",
    )
    split = "training"
    for epoch in range(0, EPOCHS):
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_epoch_and_split(epoch, split)
        train(train_dataloader_dq, modeldq)
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_split("test")
        evaluate(test_dataloader_dq, modeldq)
    unwatch()
    ThreadPoolManager.wait_for_threads()
    validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
    validate_unique_ids(vaex.open(f"{LOCATION}/{split}/1/*.hdf5"), "epoch")
    dq.finish()


@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(dq.clients.api.ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
def test_end_to_end_old_patch(
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    mock_valid_user: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    set_test_config(default_task_type=TaskType.text_classification)
    # Preprocessing
    global train_df, test_df

    dq.set_labels_for_run(labels)

    dq.log_dataset(train_df, split="train")
    dq.log_dataset(test_df, split="test")

    train_dataloader_dq = DataLoader(
        ag_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader_dq = DataLoader(
        ag_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    # 🔭🌕 Logging the dataset with Galileo
    watch(
        modeldq,
        [train_dataloader_dq, test_dataloader_dq],
        classifier_layer="classifier",
    )
    split = "training"
    for epoch in range(0, 2):
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_epoch_and_split(epoch, split)
        train(train_dataloader_dq, modeldq)
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_split("test")
        evaluate(test_dataloader_dq, modeldq)
    unwatch()
    ThreadPoolManager.wait_for_threads()
    validate_unique_ids(vaex.open(f"{LOCATION}/{split}/0/*.hdf5"), "epoch")
