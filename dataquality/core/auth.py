import getpass
import os
from typing import Callable, Dict

import requests

from dataquality.clients.api import ApiClient
from dataquality.core._config import AuthMethod, Config, GalileoConfigVars, config
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType, Route

GALILEO_AUTH_METHOD = "GALILEO_AUTH_METHOD"
api_client = ApiClient()


class _Auth:
    def __init__(self, config: Config, auth_method: AuthMethod):
        self.auth_method = auth_method
        self.config = config

    def auth_methods(self) -> Dict[AuthMethod, Callable]:
        return {AuthMethod.email: self.email_login}

    def email_login(self) -> None:
        if self.email_token_present_and_valid(self.config):
            return

        username = os.getenv("GALILEO_USERNAME")
        password = os.getenv("GALILEO_PASSWORD")
        if not username or not password:
            username = input("📧 Enter your email:")
            password = getpass.getpass("🤫 Enter your password:")
        res = requests.post(
            f"{self.config.api_url}/login",
            data={
                "username": username,
                "password": password,
                "auth_method": self.auth_method,
            },
        )
        if res.status_code != 200:
            print(res.json())
            return

        access_token = res.json().get("access_token")
        config.token = access_token
        os.environ[GalileoConfigVars.MINIO_ACCESS_KEY] = username
        os.environ[GalileoConfigVars.MINIO_SECRET_KEY] = access_token
        config.update_file_config()

    def email_token_present_and_valid(self, config: Config) -> bool:
        return config.auth_method == "email" and self.valid_current_user(config)

    def valid_current_user(self, config: Config) -> bool:
        if config.token:
            try:
                api_client.make_request(
                    RequestType.GET, url=f"{config.api_url}/{Route.current_user}"
                )
                return True
            except GalileoException:
                return False
        else:
            return False


def login() -> None:
    print("🔭 Logging you into Galileo\n")
    auth_methods = ",".join([am.value for am in AuthMethod])
    # Try auto auth config
    auth_method = os.getenv(GALILEO_AUTH_METHOD) or config.auth_method
    if not auth_method or auth_method.lower() not in list(AuthMethod):
        auth_method = input(
            "🔐 How would you like to login? \n"
            f"Enter one of the following: {auth_methods}\n"
        )
        if auth_method.lower() not in list(AuthMethod):
            print(
                "Invalid login request. You must input one of "
                f"the following authentication methods: {auth_methods}."
            )
            return
        else:  # Save it as an environment variable for the next login
            print("🤝 Saving preferred login method")
            os.environ[GALILEO_AUTH_METHOD] = auth_method
    else:
        print(f"👀 Found auth method {auth_method} set via env, skipping prompt.")
    config.auth_method = AuthMethod(auth_method)
    _auth = _Auth(config=config, auth_method=config.auth_method)
    _auth.auth_methods()[config.auth_method]()
    current_user_email = api_client.get_current_user().get("email")
    if not current_user_email:
        return
    config.current_user = current_user_email
    config.update_file_config()
    print(f"🚀 You're logged in to Galileo as {current_user_email}!")
