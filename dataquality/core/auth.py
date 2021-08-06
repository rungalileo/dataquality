from typing import Callable, Dict

import inquirer
import requests

from dataquality.core.config import _Config, config
from dataquality.schemas.config import AuthMethod, Config


class _Auth:
    def __init__(self, config: Config, auth_method: AuthMethod):
        self.auth_method = auth_method
        self.config = config

    def auth_methods(self) -> Dict[AuthMethod, Callable]:
        return {AuthMethod.email: self.email_login}

    def email_login(self) -> None:
        if self.email_token_present_and_valid(self.config):
            return

        email_credentials = [
            inquirer.Text("username", message="😎 Enter your email"),
            inquirer.Password("password", message="🤫 Enter your password"),
        ]
        answers = inquirer.prompt(email_credentials)
        req = requests.post(
            f"{self.config.api_url}/login",
            data={
                "username": answers.get("username"),
                "password": answers.get("password"),
                "auth_method": self.auth_method,
            },
        )
        access_token = req.json().get("access_token")
        self.config.token = access_token
        _config = _Config()
        _config.write_config(self.config.dict())

    def email_token_present_and_valid(self, config: Config) -> bool:
        return config.auth_method == "email" and self.current_user_by_email(config)

    def current_user_by_email(self, config: Config) -> bool:
        return (
            requests.get(
                f"{self.config.api_url}/current_user",
                headers={"Authorization": f"Bearer {config.token}"},
            ).status_code
            == 200
        )


def login() -> None:
    print("🔭 Logging you into Galileo")
    auth_method_question = [
        inquirer.List(
            "auth_method",
            message="🔐 How would you like to login?",
            choices=[str(am.value).capitalize() for am in AuthMethod],
        )
    ]
    auth_method = inquirer.prompt(auth_method_question).get("auth_method").lower()
    config.auth_method = auth_method
    _auth = _Auth(config=config, auth_method=config.auth_method)
    _auth.auth_methods()[config.auth_method]()
    print("🚀 You're logged in!")
