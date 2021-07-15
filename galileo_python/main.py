from typing import Optional
from uuid import UUID

from galileo_python.config import Config


class Galileo:
    def __init__(
        self,
        namespace: Optional[str],
        run_id: Optional[UUID],
        config: Optional[Config] = None,
        loading_from_run: bool = False,
        cross_validation: bool = False,
    ):
        pass
