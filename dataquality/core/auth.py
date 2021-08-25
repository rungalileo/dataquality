import getpass
from typing import Callable, Dict

import requests

from dataquality.core.config import AuthMethod, Config, _Config, config


class _Auth:
    def __init__(self, config: Config, auth_method: AuthMethod):
        self.auth_method = auth_method
        self.config = config

    def auth_methods(self) -> Dict[AuthMethod, Callable]:
        return {AuthMethod.email: self.email_login}

    def email_login(self) -> None:
        if self.email_token_present_and_valid(self.config):
            return

        username = input("📧 Enter your email:")
        password = getpass.getpass("🤫 Enter your password:")
        req = requests.post(
            f"{self.config.api_url}/login",
            data={
                "username": username,
                "password": password,
                "auth_method": self.auth_method,
            },
        )
        access_token = req.json().get("access_token")
        self.config.token = access_token
        _config = _Config()
        _config.write_config(self.config.json())

    def email_token_present_and_valid(self, config: Config) -> bool:
        return config.auth_method == "email" and self.valid_current_user(config)

    def valid_current_user(self, config: Config) -> bool:
        return (
            requests.get(
                f"{self.config.api_url}/current_user",
                headers={"Authorization": f"Bearer {config.token}"},
            ).status_code
            == 200
        )

    def get_current_user(self, config: Config) -> Dict:
        return requests.get(
            f"{self.config.api_url}/current_user",
            headers={"Authorization": f"Bearer {config.token}"},
        ).json()


def login() -> None:
    print("🔭 Logging you into Galileo\n")
    auth_methods = ",".join([am.value for am in AuthMethod])
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
    config.auth_method = AuthMethod(auth_method)
    _auth = _Auth(config=config, auth_method=config.auth_method)
    _auth.auth_methods()[config.auth_method]()
    current_user_email = _auth.get_current_user(config).get("email")
    config.current_user = current_user_email
    config.update_file_config()
    print(f"🚀 You're logged in to Galileo as {current_user_email}!")
