import os
import sys
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import requests
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.core.auth import api_client
from dataquality.utils.file import get_file_extension


class ObjectStore:
    DOWNLOAD_CHUNK_SIZE_MB = 256

    def create_minio_client_for_exoscale_cluster(self) -> Any:
        try:
            from minio import Minio
        except ImportError:
            raise ImportError(
                "🚨 The minio package is required to use the Exoscale cluster. "
                "Please run `pip install dataquality[minio]` to install minio "
                "with dataquality."
            )

        """Creates a Minio client for the Exoscale cluster.

        Exoscale does not support presigned urls, so we need to
        use the Minio client to upload files to the object store.

        To instantiate this, the user simply sets the EXOSCALE_API_KEY_ACCESS_KEY
        and EXOSCALE_API_KEY_ACCESS_SECRET environment variables with the values
        from their Exoscale API Key.

        Returns:
            Minio: A Minio client.
        """
        access_key = os.environ.get("EXOSCALE_API_KEY_ACCESS_KEY")
        secret_key = os.environ.get("EXOSCALE_API_KEY_ACCESS_SECRET")
        assert access_key is not None and secret_key is not None, (
            "EXOSCALE_API_KEY_ACCESS_KEY and EXOSCALE_API_KEY_ACCESS_SECRET "
            "environment variables must be set. "
            "Please set these variables with the values from your Exoscale "
            "API Key."
        )
        return Minio(
            endpoint=config.minio_fqdn,
            access_key=access_key,
            secret_key=secret_key,
            secure=True,
        )

    def __init__(self) -> None:
        self._minio_client = None
        if config.is_exoscale_cluster:
            self._minio_client = self.create_minio_client_for_exoscale_cluster()

    def create_object(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
        progress: bool = True,
        bucket_name: Optional[str] = None,
    ) -> None:
        _bucket_name = bucket_name or config.root_bucket_name
        assert _bucket_name is not None, (
            "No bucket name provided to create_object. Please provide "
            "a bucket_name by setting the root_bucket_name in your config with "
            "`dq.config.root_bucket_name = 'my-bucket-name'` or by passing a "
            "bucket_name to this function."
        )
        if config.is_exoscale_cluster:
            self._create_object_exoscale(
                object_name=object_name,
                file_path=file_path,
                content_type=content_type,
                bucket_name=_bucket_name,
            )
        else:
            url = api_client.get_presigned_url(
                project_id=object_name.split("/")[0],
                method="put",
                bucket_name=_bucket_name,
                object_name=object_name,
            )
            self._upload_file_from_local(
                url=url,
                file_path=file_path,
                content_type=content_type,
                progress=progress,
            )

    def _create_object_exoscale(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
        bucket_name: Optional[str] = None,
    ) -> None:
        """_create_object_exoscale

        This is a helper function for the Exoscale cluster. It uses the Minio
        client to upload files to the object store.

        This is necessary because Exoscale does not support presigned urls.

        Args:
            object_name (str): The name of the object to create.
            file_path (str): The path to the file to upload.
            content_type (str): The content type of the upload request.
            bucket_name (str): The name of the bucket to upload to.

        Returns:
            None
        """
        assert self._minio_client is not None
        self._minio_client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type,
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
            self.create_object(
                object_name=object_name,
                file_path=f.name,
            )

    def download_file(
        self, object_name: str, file_path: str, bucket: Optional[str] = None
    ) -> str:
        """download_file

        Args:
            object_name (str): The object name.
            file_path (str): Where to write the object data locally.
            bucket (Optional[str]): The bucket name. If None,
                the root bucket name is used.

        Returns:
            str: The local file where the object name was written.
        """
        if bucket is None:
            bucket = config.root_bucket_name
        assert bucket is not None, (
            "No bucket name provided to create_object. Please provide "
            "a bucket_name by setting the root_bucket_name in your config with "
            "`dq.config.root_bucket_name = 'my-bucket-name'` or by passing a "
            "bucket_name to this function."
        )
        url = api_client.get_presigned_url(
            project_id=object_name.split("/")[0],
            method="get",
            bucket_name=bucket,
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
