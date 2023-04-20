import hashlib
import multiprocessing
import os
import queue
import tempfile
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union

import vaex
from pydantic import UUID4
from tqdm import tqdm

from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException

api_client = ApiClient()


class UploadDfWorker(Thread):
    def __init__(
        self,
        request_queue: queue.Queue,
        project_id: UUID4,
        file_list: List[str],
        stop_val: str,
        export_format: str,
        export_cols: List[str],
        temp_dir: str,
        show_progress: bool = True,
        pbar: Optional[Any] = None,
        step: Optional[int] = None,
    ) -> None:
        Thread.__init__(self)
        self.queue = request_queue
        self.results: list = []
        self.stop_val = stop_val
        self.project_id = project_id
        self.file_list = file_list
        self.export_format = export_format
        self.export_cols = export_cols
        self.show_progress = show_progress
        self.pbar = pbar
        self.step = step
        self.temp_dir = temp_dir

    def _upload_file_for_project(
        self,
        file_path: str,
        project_id: Optional[UUID4] = None,
    ) -> Any:
        project_id = project_id or config.current_project_id
        if project_id is None:
            raise GalileoException(
                "project_id is not set in your config. Have you run dq.init()?"
            )
        return api_client.upload_file_for_project(
            project_id=str(project_id),
            file_path=file_path,
            export_format=self.export_format,
            export_cols=self.export_cols,
        )

    def run(self) -> None:
        while True:
            content = self.queue.get()
            if content == self.stop_val:
                break
            i, j = content
            with tempfile.NamedTemporaryFile(
                suffix=f".{self.export_format}"
            ) as temp_file:
                temp_file_name = temp_file.name
                ext_split = os.path.splitext(temp_file_name)
                chunk_file_path = f"{ext_split[0]}_{i}_{j}{ext_split[1]}"

                def load_bytes_from_file(
                    file_path: str,
                ) -> Dict[str, Union[str, bytes]]:
                    with open(file_path, "rb") as f:
                        img = f.read()
                        hash = hashlib.md5(img).hexdigest()
                        ext = os.path.splitext(file_path)[1]
                        return {
                            "file_path": file_path,
                            "data": img,
                            "object_path": f"{self.project_id}/{hash}{ext}",
                        }

                df = vaex.from_records(
                    list(
                        map(
                            load_bytes_from_file,
                            [f for f in self.file_list][i:j],
                        )
                    )
                )

                df[self.export_cols].export(chunk_file_path)
                res = self._upload_file_for_project(
                    file_path=chunk_file_path,
                    project_id=self.project_id,
                )
                os.remove(chunk_file_path)
                if res.ok:
                    df[["file_path", "object_path"]].export(
                        Path(self.temp_dir) / f"{i}_{j}.{self.export_format}"
                    )
                    if self.show_progress:
                        assert self.pbar is not None
                        self.pbar.update(self.step)
                else:
                    self.queue.put(content)


def chunk_load_then_upload_df(
    file_list: List[str],
    export_cols: List[str],
    temp_dir: str,
    project_id: Optional[UUID4] = None,
    parallel: bool = False,
    step: int = 50,
    num_workers: int = 1,
    stop_val: str = "END",
    export_format: str = "arrow",
    show_progress: bool = True,
) -> None:
    if parallel:
        num_workers = multiprocessing.cpu_count()

    if project_id is None:
        raise ValueError("project_id must be provided")

    # Create queue and add the ends of the chunks
    total = len(file_list)
    pbar = None
    if show_progress:
        pbar = tqdm(total=total, desc="Uploading content...")
    q: queue.Queue = queue.Queue()
    for i in range(0, total, step):
        q.put((i, i + step))

    for _ in range(num_workers):
        q.put(stop_val)

    # Create workers
    workers = []
    for _ in range(num_workers):
        worker = UploadDfWorker(
            request_queue=q,
            project_id=project_id,
            file_list=file_list,
            stop_val=stop_val,
            export_format=export_format,
            export_cols=export_cols,
            show_progress=show_progress,
            pbar=pbar,
            step=step,
            temp_dir=temp_dir,
        )
        worker.start()
        workers.append(worker)

    # Join workers to the main thread and
    # wait until all threads complete
    for worker in workers:
        worker.join()
