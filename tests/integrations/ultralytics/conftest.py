from typing import Generator

from pytest import MonkeyPatch, fixture


@fixture(autouse=True, scope="function")
def set_ultralytics_config_dir(
    tmpdir: str, monkeypatch: MonkeyPatch
) -> Generator[None, None, None]:
    # Ensure we use different config dirs for each time `ultralytics` is imported.
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/__init__.py
    # #L550-L582
    monkeypatch.setenv("YOLO_CONFIG_DIR", tmpdir)
    yield
    monkeypatch.delenv("YOLO_CONFIG_DIR")
