import abc
import datetime
import inspect
import random
from typing import Any, Callable, Dict, List, Optional, Type

import dataquality as dq
from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.schemas.model import ModelFramework
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.name import COLORS


class BaseInsights(abc.ABC):
    """Base class for dq start integrations."""

    framework: ModelFramework
    watch: Optional[Callable]
    unwatch: Optional[Callable]
    call_finish: bool = True

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize the base class.
        :param model: The model to be tracked
        :param args: Positional arguments to be passed to the watch function
        :param kwargs: Keyword arguments to be passed to the watch function
        """

        self.model = model
        self.args, self.kwargs = args, kwargs

    def enter(self) -> None:
        """Call the watch function (called in __enter__)."""
        if hasattr(self, "watch") and self.watch:
            func_signature = inspect.signature(self.watch)
            func_kwargs = {
                k: v for k, v in self.kwargs.items() if k in func_signature.parameters
            }
            self.watch(self.model, **func_kwargs)

    def exit(self) -> None:
        """Call the unwatch function (called in __exit__)."""
        if hasattr(self, "unwatch") and self.unwatch:
            self.unwatch(self.model)

    def set_project_run(
        self,
        project: str = "",
        run: str = "",
        task: TaskType = TaskType.text_classification,
    ) -> None:
        """Set the project and run names. To the class.
        If project and run are not provided, generate them.
        :param project: The project name
        :param run: The run name
        :param task: The task type
        """
        self.task = task
        if project:
            self.project = project
        else:
            self.project = f"insights_{task}"
        if run:
            self.run = run
        else:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            random_word = random.choice(COLORS)
            self.run = f"{current_time}_{self.framework}_{random_word}"

    def init_project(
        self,
        task: TaskType,
        project: str = "",
        run: str = "",
    ) -> None:
        """Initialize the project and calls dq.init().
        :param task: The task type
        :param project: The project name
        :param run: The run name
        """
        self.set_project_run(project, run, task)

        dq.init(self.task, self.project, self.run)
        self.project_id = str(config.current_project_id)
        self.run_id = str(config.current_run_id)

    def setup_training(
        self,
        labels: Optional[List[str]],
        train_data: Any,
        test_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
    ) -> None:
        """Log dataset and labels to the run.
        :param labels: The labels
        :param train_data: The training dataset
        :param test_data: The test dataset
        :param val_data: The validation dataset
        """
        if labels is not None and len(labels):
            dq.set_labels_for_run(labels)
        if train_data is not None:
            dq.log_dataset(train_data, split=Split.train)
        if test_data is not None:
            dq.log_dataset(test_data, split=Split.test)
        if val_data is not None:
            dq.log_dataset(val_data, split=Split.validation)

    def validate(self, task_type: TaskType, labels: Optional[List[str]]) -> None:
        """Validate the task type and labels.
        :param task_type: The task type
        :param labels: The labels
        """
        assert (
            task_type is not None
        ), """keyword argument task_type is required,
    for example task_type='text_classification' """
        assert labels is not None and len(
            labels
        ), """keyword labels is required,
    for example labels=['neg','pos']"""


class TorchInsights(BaseInsights):
    framework = ModelFramework.torch

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        from dataquality.integrations.torch import unwatch as torch_unwatch
        from dataquality.integrations.torch import watch as torch_watch

        self.unwatch = torch_unwatch
        self.watch = torch_watch


class TFInsights(BaseInsights):
    framework = ModelFramework.keras

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        from dataquality.integrations.keras import unwatch as keras_unwatch
        from dataquality.integrations.keras import watch as keras_watch

        self.unwatch = keras_unwatch
        self.watch = keras_watch


class TrainerInsights(BaseInsights):
    framework = ModelFramework.hf

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        from dataquality.integrations.transformers_trainer import (
            unwatch as trainer_unwatch,
        )
        from dataquality.integrations.transformers_trainer import watch as trainer_watch

        self.unwatch = trainer_unwatch
        self.watch = trainer_watch


class AutoInsights(BaseInsights):
    framework = ModelFramework.auto
    call_finish = False
    auto_kwargs: Dict[str, Any]

    def __init__(self, model: Any) -> None:
        super().__init__(model)
        from dataquality.dq_auto.auto import auto

        self.auto = auto
        self.auto_kwargs = {}

    def setup_training(
        self,
        labels: Optional[List[str]],
        train_data: Any,
        test_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
    ) -> None:
        """Setup auto by creating the parameters for the auto function.
        :param labels: Labels for the training
        :param train_data: Training dataset
        :param test_data: Test dataset
        :param val_data: Validation dataset
        """
        auto_kwargs = self.auto_kwargs
        if self.task:
            assert self.task == TaskType.text_classification

        if labels is not None and len(labels):
            auto_kwargs["labels"] = labels
        if train_data is not None:
            auto_kwargs["train_data"] = train_data
        if test_data is not None:
            auto_kwargs["test_data"] = test_data
        if val_data is not None:
            auto_kwargs["validation_data"] = val_data
        if self.project:
            auto_kwargs["project_name"] = self.project
        if self.run:
            auto_kwargs["run_name"] = self.run
        if isinstance(self.model, str):
            auto_kwargs["hf_model"] = self.model

    def init_project(self, task: TaskType, project: str = "", run: str = "") -> None:
        """Initialize the project and run but dq init is not called.
        :param task: The task type
        :param project: The project name
        :param run: The run name
        """
        self.set_project_run(project, run, task)

    def enter(self) -> None:
        """Call auto function with the generated paramters."""
        self.auto(**self.auto_kwargs)

    def validate(self, task_type: TaskType, labels: Optional[List[str]] = []) -> None:
        """Validate the task type and labels.
        :param task_type: The task type
        :param labels: The labels
        """
        pass


def detect_model(model: Any, framework: Optional[ModelFramework]) -> Type[BaseInsights]:
    """Detect the model type in a lazy way and return the appropriate class.
    :param model: The model to inspect, if a string, it will be assumed to be auto
    :param framework: The framework to use, if provided it will be used instead of
         the model
    """
    if hasattr(model, "fit") or framework == ModelFramework.keras:
        return TFInsights
    elif hasattr(model, "register_forward_hook") or framework == ModelFramework.torch:
        return TorchInsights
    elif hasattr(model, "push_to_hub") or framework == ModelFramework.hf:
        return TrainerInsights
    elif isinstance(model, str) or framework == ModelFramework.auto:
        return AutoInsights
    else:
        raise GalileoException(
            f"Model type: {type(model)} could not be detected model type"
        )


class DataQuality:
    def __init__(
        self,
        model: Optional[Any] = None,
        task: TaskType = TaskType.text_classification,
        labels: Optional[List[str]] = None,
        train_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        val_data: Optional[Any] = None,
        project: str = "",
        run: str = "",
        framework: Optional[ModelFramework] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """DataQuality
        :param model: The model to inspect, if a string, it will be assumed to be auto
        :param task: Task type for example "text_classification"
        :param project: Project name
        :param run: Run name
        :param train_data: Training data
        :param test_data: Optional test data
        :param val_data: Optional: validation data
        :param labels: The labels for the run
        :param framework: The framework to use, if provided it will be used instead of
            inferring it from the model. For example, if you have a torch model, you
            can pass framework="torch". If you have a torch model, you can pass
            framework="torch"
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments

        .. code-block:: python

            from dataquality import DataQuality

            with DataQuality(model, "text_classification",
                             labels = ["neg", "pos"],
                             train_data = train_data) as dq:
                model.fit(train_data)

        If you want to train without a model, you can use the auto framework:

        .. code-block:: python

            from dataquality import DataQuality

            with DataQuality(labels = ["neg", "pos"],
                             train_data = train_data) as dq:
                dq.finish()
        """
        self.args, self.kwargs = args, kwargs
        self.model = model
        if model:
            cls = detect_model(model, framework)
        else:
            cls = AutoInsights

        self.cls = cls(model)
        self.cls.validate(task, labels)
        self.cls.init_project(task, project, run)
        self.cls.setup_training(labels, train_data, test_data, val_data)
        if cls == AutoInsights:
            self.cls.enter()
            self.finished = True

    def __enter__(self) -> Any:
        if getattr(self, "finished", False):
            return dq
        self.cls.enter()
        return dq

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.cls.exit()
        if self.cls.call_finish:
            dq.finish()

    def get_metrics(self, split: Split = Split.training) -> Dict[str, Any]:
        return dq.metrics.get_metrics(
            project_name=self.cls.project,
            run_name=self.cls.run,
            task=self.cls.task,
            split=split,
        )
