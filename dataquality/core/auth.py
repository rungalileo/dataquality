import getpass
import os
import webbrowser

import requests

from dataquality.clients.api import ApiClient
from dataquality.core._config import config, url_is_localhost
from dataquality.exceptions import GalileoException
from dataquality.schemas.route import Route
from dataquality.utils.helpers import check_noop

GALILEO_AUTH_METHOD = "GALILEO_AUTH_METHOD"
api_client = ApiClient()


class _Auth:
    def email_login(self) -> None:
        username = os.getenv("GALILEO_USERNAME")
        password = os.getenv("GALILEO_PASSWORD")
        res = requests.post(
            f"{config.api_url}/login",
            data={
                "username": username,
                "password": password,
                "auth_method": "email",
            },
            headers={"X-Galileo-Request-Source": "dataquality_python_client"},
        )
        if res.status_code != 200:
            raise GalileoException(
                (
                    f"Issue authenticating: {res.json()['detail']} "
                    "If you need to reset your password, "
                    f"go to {config.api_url.replace('api', 'console')}/forgot-password"
                )
            )

        access_token = res.json().get("access_token")
        config.token = access_token
        config.update_file_config()

    def token_login(self) -> None:
        if url_is_localhost(url=config.api_url):
            token_url = f"{os.environ['GALILEO_CONSOLE_URL']}/{Route.token}"
        else:
            token_url = f"{config.api_url.replace('api.', 'console.')}/{Route.token}"
        try:
            webbrowser.open(token_url)
        except Exception:
            pass
        print(f"Go to {token_url} to generate a new API Key")
        access_token = getpass.getpass("🔐 Enter your API Key:")
        config.token = access_token
        config.update_file_config()


@check_noop
def login() -> None:
    """Log into your Galileo environment.

    The function will prompt your for an Authorization Token (api key) that you can
    access from the console.

    To skip the prompt for automated workflows, you can set `GALILEO_USERNAME`
    (your email) and GALILEO_PASSWORD if you signed up with an email and password
    """
    if api_client.valid_current_user():
        print(f"✅ Already logged in as {config.current_user}!")
        print("Use logout() if you want to change users")

        return

    print(f"📡 {config.api_url.replace('api','console')}")
    print("🔭 Logging you into Galileo\n")

    _auth = _Auth()
    if os.getenv("GALILEO_USERNAME") and os.getenv("GALILEO_PASSWORD"):
        _auth.email_login()
    else:
        _auth.token_login()

    current_user_email = api_client.get_current_user().get("email")
    if not current_user_email:
        return
    config.current_user = current_user_email
    config.update_file_config()
    print(f"🚀 You're logged in to Galileo as {current_user_email}!")


@check_noop
def logout() -> None:
    config.current_user = None
    config.token = None
    config.update_file_config()
    print("👋 You have logged out of Galileo")
