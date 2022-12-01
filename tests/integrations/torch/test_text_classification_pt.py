from typing import Callable, Generator

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

import dataquality as dq
from dataquality.integrations.torch import watch
from dataquality.schemas.task_type import TaskType

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
    for (_label, _text) in batch:
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
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


""" Initate an instance """
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
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
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
            loss = criterion(predicted_label, label)
            loss
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


"""Split the dataset and run the model"""

# Hyperparameters
EPOCHS = 1  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
    split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
    split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)


""" Evaluate the model with test dataset"""


def test_dataset() -> None:
    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model)
    print("test accuracy {:8.3f}".format(accu_test))


def test_end_to_end_without_callback():
    global total_accu
    for epoch in range(1, EPOCHS + 1):
        train(train_dataloader, model)
        accu_val = evaluate(valid_dataloader, model)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val


def test_end_to_end_with_callback(
    cleanup_after_use: Generator, set_test_config: Callable
) -> None:
    set_test_config(default_task_type=TaskType.text_classification)
    global total_accu
    # Preprocessing
    ag_train = to_map_style_dataset(AG_NEWS(split="train"))[:500]
    ag_test = to_map_style_dataset(AG_NEWS(split="test"))[500:800]

    train_df = pd.DataFrame(ag_train)
    test_df = pd.DataFrame(ag_test)
    labels = train_df[0].unique()
    labels.sort()
    dq.set_labels_for_run(labels)
    train_df = train_df.reset_index().rename(
        columns={0: "label", 1: "text", "index": "id"}
    )
    train_df["id"] = train_df["id"] + 10000
    test_df = test_df.reset_index().rename(
        columns={0: "label", 1: "text", "index": "id"}
    )
    dq.log_dataset(train_df, split="train")
    dq.log_dataset(test_df, split="validation")

    train_dataloader_dq = DataLoader(
        ag_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )
    test_dataloader_dq = DataLoader(
        ag_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
    )

    # 🔭🌕 Logging the dataset with Galileo
    watch(modeldq, [train_dataloader_dq, test_dataloader_dq])

    for epoch in range(1, EPOCHS + 1):
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_epoch_and_split(epoch, "training")
        train(train_dataloader_dq, modeldq)
        # 🔭🌕 Logging the dataset with Galileo
        dq.set_split("validation")
        accu_val = evaluate(test_dataloader_dq, modeldq)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
