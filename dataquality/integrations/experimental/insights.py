import abc
import datetime
from abc import abstractmethod
from typing import Any, List

import tensorflow as tf
import torch
from spacy.language import Language
from transformers import Trainer

import dataquality as dq
from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.integrations.experimental.keras import unwatch as kerasunwatch
from dataquality.integrations.experimental.keras import watch as keraswatch
from dataquality.integrations.spacy import unwatch as spacyunwatch
from dataquality.integrations.spacy import watch as spacywatch
from dataquality.integrations.torch import unwatch as ptunwatch
from dataquality.integrations.torch import watch as ptwatch
from dataquality.integrations.transformers_trainer import unwatch as trainerunwatch
from dataquality.integrations.transformers_trainer import watch as trainerwatch
from dataquality.schemas.split import Split


class BaseInsights(abc.ABC):
    framework: str

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def enter(self) -> None:
        ...

    @abstractmethod
    def exit(self) -> None:
        ...


class TorchInsights(BaseInsights):
    framework = "pt"

    def enter(self) -> None:
        ptwatch(self.model)

    def exit(self) -> None:
        ptunwatch()


class TFInsights(BaseInsights):
    framework = "tf"

    def enter(self) -> None:
        keraswatch(self.model)

    def exit(self) -> None:
        kerasunwatch(self.model)


class SpacyInsights(BaseInsights):
    framework = "spacy"

    def enter(self) -> None:
        spacywatch(self.model)

    def exit(self) -> None:
        spacyunwatch(self.model)


class TrainerInsights(BaseInsights):
    framework = "transformers"

    def enter(self) -> None:
        trainerwatch(self.model)

    def exit(self) -> None:
        trainerunwatch(self.model)


def detect_model(model: Any) -> BaseInsights:
    if isinstance(model, torch.nn.Module):
        return TorchInsights(model)
    elif isinstance(model, tf.keras.layers.Layer):
        return TFInsights(model)
    elif isinstance(model, Language):
        return SpacyInsights(model)
    elif isinstance(model, Trainer):
        return TrainerInsights(model)
    else:
        raise GalileoException("Model type could not be detected model type")


class Insights:
    def __init__(
        self,
        model: Any,
        task: str = "text_classification",
        train_df: Any = None,
        test_df: Any = None,
        labels: List[str] = [],
    ) -> None:
        current_time_to_minute = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.cls = detect_model(model)
        if not config.current_project_id:
            self.task = task
            self.name = f"insights_{task}"
            self.run = f"{self.cls.framework}_{current_time_to_minute}"
            dq.init(self.task, self.name, self.run)
            dq.log_dataset(train_df, split=Split.train)
            dq.log_dataset(test_df, split=Split.test)
        else:
            self.task = str(config.task_type)
            self.name = str(config.current_project_id)
            self.run = str(config.current_run_id)
        if len(labels) > 0:
            dq.set_labels_for_run(labels)

        self.model = model

    def __enter__(self) -> Any:
        self.cls.enter()
        return dq

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.cls.exit()
        dq.finish()
        # return dq.metrics.get_dataframe(self.name, self.run, "train")
