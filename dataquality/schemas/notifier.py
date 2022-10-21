from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, StrictStr

from dataquality.schemas.actions import ActionType


class Notifier(BaseModel):
    action_type: ActionType
    content: Optional[Dict[str, Any]] = None


class Slack(Notifier):
    action_type = ActionType.send_slack_message
    token: StrictStr
    channel: StrictStr


class Email(Notifier):
    action_type = ActionType.send_email
    to_emails: List[EmailStr]
