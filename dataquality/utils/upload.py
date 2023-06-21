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

from dataquality.clients.api import ApiClient

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
        bucket: str,
        object_path: str,
        show_progress: bool = True,
        pbar: Optional[Any] = None,
        step: Optional[int] = None,
        use_data_md5_hash: bool = True,
    ) -> None:
        Thread.__init__(self)
        self.project_id = project_id
        self.queue = request_queue
        self.results: list = []
        self.stop_val = stop_val
        self.file_list = file_list
        self.export_format = export_format
        self.export_cols = export_cols
        self.show_progress = show_progress
        self.pbar = pbar
        self.step = step
        self.temp_dir = temp_dir
        self.bucket = bucket
        self.object_path = object_path
        self.use_data_md5_hash = use_data_md5_hash

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

                        if self.use_data_md5_hash:
                            # Use a md5 hash of the data as the file name
                            # This can be used to de-dup data (e.g. images)
                            file_name = hashlib.md5(img).hexdigest()
                        else:
                            # Get the file name without the extension
                            file_name = file_path.split("/")[-1].split(".")[0]
                        ext = os.path.splitext(file_path)[1]
                        return {
                            "file_path": file_path,
                            "data": img,
                            "object_path": os.path.join(self.object_path, file_name)
                            + ext,
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
                res = api_client.upload_file_for_project(
                    project_id=str(self.project_id),
                    file_path=chunk_file_path,
                    export_format=self.export_format,
                    export_cols=self.export_cols,
                    bucket=self.bucket,
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
    bucket: str,
    project_id: Optional[UUID4] = None,
    object_path: Optional[str] = None,
    parallel: bool = False,
    step: int = 50,
    num_workers: int = 1,
    stop_val: str = "END",
    export_format: str = "arrow",
    show_progress: bool = True,
    use_data_md5_hash: bool = True,
) -> None:
    if parallel:
        num_workers = multiprocessing.cpu_count()

    if project_id is None:
        raise ValueError("project_id must be provided")

    object_path = object_path or str(project_id)

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
            bucket=bucket,
            use_data_md5_hash=use_data_md5_hash,
            object_path=object_path,
        )
        worker.start()
        workers.append(worker)

    # Join workers to the main thread and
    # wait until all threads complete
    for worker in workers:
        worker.join()
