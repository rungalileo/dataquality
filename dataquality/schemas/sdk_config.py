from pydantic import BaseModel


class SDKConfig(BaseModel):
    api_url: str = "https://api.rungalileo.io"
