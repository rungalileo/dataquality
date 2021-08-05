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


def login() -> None:
    _config = config()
    print("🔭 Logging you into Galileo")
    auth_method_question = [
        inquirer.List(
            "auth_method",
            message="🔐 How would you like to login?",
            choices=[str(am.value).capitalize() for am in AuthMethod],
        )
    ]
    auth_method = inquirer.prompt(auth_method_question).get("auth_method").lower()
    _config.auth_method = auth_method
    _auth = _Auth(config=_config, auth_method=_config.auth_method)
    _auth.auth_methods()[_config.auth_method]()
