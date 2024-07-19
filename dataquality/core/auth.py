import getpass
import os
import webbrowser

from dataquality.clients.api import ApiClient
from dataquality.core._config import config, reset_config, url_is_localhost
from dataquality.schemas.route import Route
from dataquality.utils.helpers import check_noop

GALILEO_AUTH_METHOD = "GALILEO_AUTH_METHOD"
api_client = ApiClient()


class _Auth:
    def login_with_env_vars(self) -> None:
        """Automatically log in with environment variables."""
        api_client.get_token()

    def login_with_token(self) -> None:
        """Prompts a user to copy and paste a token from the console."""
        if url_is_localhost(url=config.api_url):
            token_url = f"{os.environ['GALILEO_CONSOLE_URL']}/{Route.token}"
        else:
            token_url = f"{config.api_url.replace('api.', 'console.')}/{Route.token}"
        try:
            webbrowser.open(token_url)
        except Exception:
            pass
        print(f"Go to {token_url} to generate a new Galileo token.")
        access_token = getpass.getpass("🔐 Enter your Galileo token: ")
        config.token = access_token
        config.update_file_config()


@check_noop
def login() -> None:
    """Log into your Galileo environment.

    The function will prompt your for an Authorization Token (api key) that you can
    access from the console.

    To skip the prompt for automated workflows, you can set `GALILEO_USERNAME`
    (your email) and GALILEO_PASSWORD if you signed up with an email and password.
    You can set `GALILEO_API_KEY` to your API key if you have one.
    """
    if not config.api_url:
        updated_config = reset_config()
        for k, v in updated_config.dict().items():
            config.__setattr__(k, v)
        config.token = None
        config.update_file_config()

    valid_current_user = api_client.valid_current_user()
    print(f"📡 {config.api_url.replace('api','console')}")
    print("🔭 Logging you into Galileo\n")

    _auth = _Auth()
    has_api_key = os.getenv("GALILEO_API_KEY")
    has_username_password = os.getenv("GALILEO_USERNAME") and os.getenv(
        "GALILEO_PASSWORD"
    )
    if has_api_key or has_username_password:
        _auth.login_with_env_vars()
    if not valid_current_user:
        _auth.login_with_token()

    current_user_email = api_client.get_current_user().get("email")
    if not current_user_email:
        return

    config.current_user = current_user_email
    config.update_file_config()

    print(f"🚀 You're logged in to Galileo as {current_user_email}!")
    print("Use logout() if you want to change users")


@check_noop
def logout() -> None:
    config.current_user = None
    config.token = None
    config.update_file_config()
    print("👋 You have logged out of Galileo")
