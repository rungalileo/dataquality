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
            "🔐 Authentication token not found. To skip this prompt in the future \n"
            "set the GALILEO_JWT_TOKEN environment variable. You can get your JWT \n"
            "token from the console. \n"
            "Please enter your token: \n"
        )

    try:
        current_user_email = api_client.get_current_user().get("email")
    except GalileoException:
        print(
            "\n🚨 Invalid token. Make sure to get the latest token from the "
            "console. \n"
        )
        return

    if not current_user_email:
        print(
            "\n🚨 User not found for this token. Make sure to get the latest token "
            "from the console. \n"
        )
        return
    config.current_user = current_user_email
    config.update_file_config()
    print(f"\n🚀 You're logged in to Galileo as {current_user_email}!")
