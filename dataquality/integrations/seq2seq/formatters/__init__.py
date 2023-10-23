from typing import Dict, Type

from dataquality.integrations.seq2seq.formatters.alpaca import AlpacaFormatter
from dataquality.integrations.seq2seq.formatters.base import (
    BaseFormatter,
    DefaultFormatter,
)

FORMATTER_MAPPING: Dict[str, Type[BaseFormatter]] = {
    AlpacaFormatter.name: AlpacaFormatter,
}


def get_formatter(name: str) -> BaseFormatter:
    """Returns the formatter for the given name

    If the name isn't found, returns the base formatter
    """
    return FORMATTER_MAPPING.get(name, DefaultFormatter)()  # type: ignore
