import os
from typing import Optional

from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.exceptions import GalileoException

api_client = ApiClient()


def use_code_for_access_token() -> None:
    """
    Submit the code for access token to the Galileo console
    """
    auth_code = os.getenv("GALILEO_AUTH_CODE", None)
    refresh_token = config.refresh_token or os.getenv("GALILEO_REFRESH_TOKEN")
    response = {}

    # if the refresh token is present, use it
    if refresh_token:
        print("🔐 Using refresh token")
        response = api_client.use_refresh_token(refresh_token)

    # otherwise, use the provided auth code
    elif auth_code is None or auth_code == "":
        auth_code = input(
            "🔐 Authentication code not found in environment. "
            "To skip this prompt in the future "
            "set the GALILEO_AUTH_CODE environment variable.\n\n You can get your auth "
            "code from the console.\n\n Please enter your auth code: \n"
        )
        print("🔭 Submitting code for access token...\n")
        response = api_client.get_refresh_token(code=auth_code)

    # otherwise use the provided auth code from the env
    else:
        print("🔭 Submitting code for access token...\n")
        response = api_client.get_refresh_token(code=auth_code)

    config.token = response.get("access_token")
    config.refresh_token = response.get("refresh_token")


def verify_jwt_token() -> None:
    """
    Set and verify the Auth0 token for the current user.
    """
    use_code_for_access_token()

    print(f"📡 {config.api_url.replace('api.','console.')}")
    print("🔭 Validating Auth0 token...\n")

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
