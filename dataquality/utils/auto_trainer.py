import webbrowser
from typing import Optional

from datasets import DatasetDict
from transformers import Trainer

import dataquality as dq
from dataquality.integrations.transformers_trainer import watch
from dataquality.schemas.split import Split


def open_console_url(link: Optional[str] = "") -> None:
    """Tries to open the console url in the browser, if possible.

    This will work in local environments like jupyter or a python script, but won't
    work in colab (because colab is running on a server, so there's no "browser" to
    interact with). This also prints out the link for users to click so even in those
    environments they still have something to interact with.
    """
    if not link:
        return
    try:
        webbrowser.open(link)
    # In some environments, webbrowser will raise. Other times it fails silently (colab)
    except Exception:
        pass
    finally:
        print(f"Click here to see your run! {link}")


def do_train(trainer: Trainer, encoded_data: DatasetDict, wait: bool) -> None:
    watch(trainer)
    trainer.train()
    if Split.test in encoded_data:
        # We pass in a huggingface dataset but typing wise they expect a torch dataset
        trainer.predict(test_dataset=encoded_data[Split.test])  # type: ignore
    res = dq.finish(wait=wait) or {}
    open_console_url(res.get("link"))
