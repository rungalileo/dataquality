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
            "🔐 Missing GALILEO_JWT_TOKEN. This can be found in the \n"
            f"Galileo console. Please enter token: \n"
        )

    try:
        current_user_email = api_client.get_current_user().get("email")
    except GalileoException:
        print(
            "\n🚨 Invalid JWT token. Make sure to get the latest token from the console. \n"
        )
        return

    if not current_user_email:
        print(
            "\n🚨 User not found for this token. Make sure to get the latest token from the console. \n"
        )
        return
    config.current_user = current_user_email
    config.update_file_config()
    print(f"\n🚀 You're logged in to Galileo as {current_user_email}!")
