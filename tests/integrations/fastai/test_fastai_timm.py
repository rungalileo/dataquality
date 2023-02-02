from fastai.vision.all import *
from fastai.data.core import DataLoaders
from fastai.data.load import DataLoader as FastDataLoader
from dataquality.integrations.fastai import DQFastAiCallback
from fastai.optimizer import OptimWrapper
from torch import optim
from fastai.learner import Learner
from torch.utils.data import TensorDataset

import torch
import torch.nn as nn

# TODO
def test_end_to_end():
    return
    path = untar_data(URLs.PETS) / "images"
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        label_func=lambda x: x[0].isupper(),
        item_tfms=Resize(224),
        num_workers=3,
    )
    learn = vision_learner(dls, "resnet34", metrics=error_rate)
    dqc = DQFastAiCallback(disable_dq=False, labels=["nocat", "cat"])
    learn.add_cb(dqc)
    learn.fine_tune(1)


class Model(nn.Module):
    def __init__(self, input_dim, embedding_output_dim, classifier_output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_output_dim)
        self.fc = nn.Linear(embedding_output_dim, classifier_output_dim, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x


input_dim = 128
embedding_output_dim = 1
classifier_output_dim = 1

model = Model(input_dim, embedding_output_dim, classifier_output_dim)

weights = torch.tensor([[i] for i in range(input_dim)], dtype=torch.float32)
model.embedding.weight.data.copy_(weights)

model.fc.weight.data.fill_(1)

input_list = torch.arange(input_dim)

criterion = nn.MSELoss()
target = torch.ones(input_dim, 1)


def opt_func(params, **kwargs):
    return OptimWrapper(optim.SGD(params, lr=0.001))


def test_simple_model():
    train_dataset = TensorDataset(input_list.long(), target.float())
    dl = FastDataLoader(train_dataset, batch_size=4, num_workers=2, shuffle=True)

    dls = DataLoaders(dl, dl)
    learn = Learner(
        dls,
        model,
        nn.MSELoss()
        # , metrics=[accuracy]
    )
    dqc = DQFastAiCallback(disable_dq=True, layer=model.fc)
    learn.add_cb(dqc)
    learn.fine_tune(2)
