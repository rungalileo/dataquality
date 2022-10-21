from enum import Enum, unique


@unique
class ActionType(str, Enum):
    send_email = "send_email"
    send_slack_message = "send_slack_message"
