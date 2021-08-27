import os

from dataquality import config
from dataquality.clients import object_store


def finish() -> None:
    # TODO: build final frame and upload!
    # TODO: will need to upload embeddings and joined input and output
    assert config.current_project_id
    assert config.current_run_id
    object_store.create_project_run_object(
        config.current_project_id,
        config.current_run_id,
        object_name="input_data.jsonl",
        file_path=f"{os.getcwd()}/.galileo/logs/{config.current_project_id}"
        f"/{config.current_run_id}/input_data.jsonl",
    )
