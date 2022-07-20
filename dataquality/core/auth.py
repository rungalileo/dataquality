from typing import Optional

from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.exceptions import GalileoException

api_client = ApiClient()


def verify_jwt_token() -> None:
    """
    Set and verify the JWT token for the current user.
    """
    print(f"📡 {config.api_url.replace('api.','console.')}")
    print("🔭 Validating Galileo token...\n")
    if not config.token:
        config.token = input(
            "🔐 Authentication token not found. To skip this prompt in the future "
            "set the GALILEO_JWT_TOKEN environment variable.\nYou can get your JWT "
            "token from the console. \n"
            "Please enter your token: \n"
        )

    try:
        current_user_email = api_client.get_current_user().get("email")
    except GalileoException:
        print(
            "\n🚨 Invalid token. Make sure to get the latest token from the "
            "console and call dq.login\n"
        )
        return

    if not current_user_email:
        raise GalileoException(
            "\n🚨 User not found for this token. Make sure to get the latest token "
            "from the console. \n"
        )
    config.current_user = current_user_email
    config.update_file_config()
    print(f"\n🚀 You're logged in to Galileo as {current_user_email}!")


def login(token: Optional[str] = None) -> None:
    """Login to the Galileo environment with your JWT token

    :param token: The JWT token retrieved from the Galileo UI
        If not provided, you will be prompted for a token
    """
    if token:
        config.token = token
        config.update_file_config()
    verify_jwt_token()
