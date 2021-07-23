from torch import Tensor, nn

from galileo_python.log import Logger
from galileo_python.schemas.logger import LoggerMode


def galileo_module_hook(
    module: nn.Module,
    input: Tensor,
    output: Tensor,
    galileo_logger: Logger,
    logger_mode: LoggerMode,
) -> None:
    # https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    # https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    pass
