from transformers import GenerationConfig, PreTrainedModel

from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import get_task_type


def watch(model: PreTrainedModel, generation_config: GenerationConfig) -> None:
    """Seq2seq only. Log model generations for your run

    Iterates over a given dataset and logs the generations for each sample.
    `model` must be an instance of transformers PreTrainedModel and have a `generate`
      method.
    """
    task_type = get_task_type()
    assert task_type == TaskType.seq2seq, "This method is only supported for seq2seq"
    assert isinstance(
        model, PreTrainedModel
    ), "model must be an instance of transformers PreTrainedModel"
    assert model.can_generate(), "model must contain a `generate` method for seq2seq"
