import os
from enum import Enum, unique
from typing import Optional

from pydantic import BaseModel


@unique
class AuthMethod(str, Enum):
    email = "email"


class Config(BaseModel):
    api_url: str = os.getenv("GALILEO_API_URL") or "https://api.rungalileo.io"
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[str] = None
