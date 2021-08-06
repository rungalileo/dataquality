import os
from enum import Enum, unique
from typing import Dict, Optional

from pydantic import BaseModel

from dataquality.core.config import _Config


@unique
class AuthMethod(str, Enum):
    email = "email"


class Config(BaseModel):
    api_url: str = os.getenv("GALILEO_API_URL") or "https://api.rungalileo.io"
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[str] = None
    current_project: Optional[Dict] = None
    current_run: Optional[Dict] = None

    def update_file_config(self) -> None:
        _config = _Config()
        _config.write_config(self.dict())
