from typing import Any, Dict

import dataquality as dq
from dataquality.exceptions import GalileoException

NOTEBOOK_ENABLED = True
try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    NOTEBOOK_ENABLED = False

TRAIN_FILE = "train.csv"


def auto_notebook() -> None:
    """"""
    print("Welcome to notebook! Upload a file to begin")
    if not NOTEBOOK_ENABLED:
        raise GalileoException(
            "To use `auto_notebook`, you must have IPython and ipywidgets installed."
            "Run `pip install ipython ipywidgets` to continue"
        )

    def auto(*args: Any) -> None:
        """On button click, run auto"""
        print("Running Galileo! This may take a minute to begin")
        dq.auto(train_data=TRAIN_FILE)

    run_button = widgets.Button(description="Run Galileo 🔭", disabled=True)
    run_button.on_click(auto)

    def save_file(inputs: Dict) -> None:
        """On upload, save file to disk, enable run-auto button"""
        keys = list(inputs["new"].keys())
        if not keys:
            return
        key = keys[0]

        key_obj = inputs["new"][key]
        if isinstance(key_obj, list):
            content = key_obj[0]["content"].tobytes()
        else:
            content = key_obj["content"]
        with open(TRAIN_FILE, "w") as f:
            f.write(content.decode("utf-8"))
        run_button.disabled = False
        print("Now click the 'Run Galileo' button!")

    upload = widgets.FileUpload(accept=".csv")

    upload.observe(save_file)
    display(widgets.HBox([upload, run_button]))
