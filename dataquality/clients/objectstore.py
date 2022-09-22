import os
import sys
from functools import partial
from tempfile import NamedTemporaryFile

import requests
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper
from vaex.dataframe import DataFrame

from dataquality.core.auth import api_client
from dataquality.utils.file import get_file_extension


class ObjectStore:
    ROOT_BUCKET_NAME = "galileo-project-runs"
    DOWNLOAD_CHUNK_SIZE_MB = 256

    def create_project_run_object(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
        progress: bool = True,
    ) -> None:
        url = api_client.get_presigned_url(
            project_id=object_name.split("/")[0],
            method="put",
            bucket_name=self.ROOT_BUCKET_NAME,
            object_name=object_name,
        )
        self._upload_file_from_local(
            url=url, file_path=file_path, content_type=content_type, progress=progress
        )

    def _upload_file_from_local(
        self,
        url: str,
        file_path: str,
        content_type: str = "application/octet-stream",
        progress: bool = True,
    ) -> None:
        """_upload_file_from_local

        Args:
            url (str): The url to request.
            file_path (str): Where data is stored on the local file system.
            content_type (str): The content type of the upload request.

        Returns:
            None
        """
        # https://gist.github.com/tyhoff/b757e6af83c1fd2b7b83057adf02c139
        open_type = "r" if content_type.startswith("text") else "rb"

        put_req = partial(requests.put, url, headers={"content-type": content_type})
        with open(file_path, open_type) as f:
            if progress:
                file_size = os.stat(file_path).st_size
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    file=sys.stdout,
                    desc="Uploading data to Galileo",
                    leave=False,
                ) as t:
                    wrapped_file = CallbackIOWrapper(t.update, f, "read")
                    put_req(data=wrapped_file)
            else:
                put_req(data=f)

    def create_project_run_object_from_df(
        self, df: DataFrame, object_name: str
    ) -> None:
        """Uploads a Vaex dataframe at the specified object_name location"""
        ext = get_file_extension(object_name)
        with NamedTemporaryFile(suffix=ext) as f:
            df.export(f.name)
            self.create_project_run_object(
                object_name=object_name,
                file_path=f.name,
            )

    def download_file(self, object_name: str, file_path: str) -> str:
        """download_file

        Args:
            object_name (str): The object name.
            file_path (str): Where to write the object data locally.

        Returns:
            str: The local file where the object name was written.
        """
        url = api_client.get_presigned_url(
            project_id=object_name.split("/")[0],
            method="get",
            bucket_name=self.ROOT_BUCKET_NAME,
            object_name=object_name,
        )
        return self._local_download_from_url(url=url, file_path=file_path)

    def _local_download_from_url(self, url: str, file_path: str) -> str:
        """_local_download_from_url

        Args:
            url (str): The url to request.
            file_path (str): The path to where data was streamed on the requester's
             local filesystem.

        Returns:
            str: The path to where data was streamed on the requester's local
             filesystem.
        """
        with requests.get(url, stream=True) as remote_file:
            remote_file.raise_for_status()
            with open(file_path, "wb") as local_file:
                for chunk in remote_file.iter_content(
                    chunk_size=1024 * self.DOWNLOAD_CHUNK_SIZE_MB
                ):
                    local_file.write(chunk)
        return file_path
