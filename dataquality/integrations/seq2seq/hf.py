from transformers import GenerationConfig, PreTrainedModel

from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import get_task_type


def watch(
    model: PreTrainedModel,
    generation_config: GenerationConfig,
    generate_training_data: bool = False,
) -> None:
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

    seq2seq_logger_config.model = model
    seq2seq_logger_config.generation_config = generation_config

    generation_splits = {Split.validation, Split.test}
    if generate_training_data:
        generation_splits.add(Split.training)
    seq2seq_logger_config.generation_splits = generation_splits
