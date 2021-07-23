"galileo_python"

__version__ = "0.0.1"


from galileo_python.auth import login
from galileo_python.config import config
from galileo_python.finish import finish
from galileo_python.init import init
from galileo_python.log import log

__all__ = ["__version__", "login", "init", "log", "config", "finish"]
