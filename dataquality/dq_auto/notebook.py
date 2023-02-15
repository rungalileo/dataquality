"""A notebook friendly `dq.auto` interface for an upload button to auto"""
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
    if not NOTEBOOK_ENABLED:
        raise GalileoException(
            "To use `auto_notebook`, you must have IPython and ipywidgets installed."
            "Run `pip install ipython ipywidgets` to continue"
        )

    def auto(*args: Any) -> None:
        """On button click, run auto"""
        dq.auto(train_data=TRAIN_FILE)

    run_button = widgets.Button(description="Run Galileo 🔭", disabled=True)
    run_button.on_click(auto)

    def save_file(inputs: Dict) -> None:
        """On upload, save file to disk, enable run-auto button"""
        key = list(inputs["new"].keys())[0]
        content = inputs["new"][key]["content"]
        with open(TRAIN_FILE, "w") as f:
            f.write(content.decode("utf-8"))
        run_button.disabled = False

    upload = widgets.FileUpload(accept=".csv")

    upload.observe(save_file, names="value")
    display(widgets.HBox([upload, run_button]))
