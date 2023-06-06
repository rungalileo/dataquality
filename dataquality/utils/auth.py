from typing import Dict, Optional


def headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        raise ValueError(
            "Missing token passed to headers utility! "
            "Check that you are logged into Galileo by running "
            "dataquality.config.email. "
            "Your account email should appear."
        )
    return {"Authorization": f"Bearer {token}"}
