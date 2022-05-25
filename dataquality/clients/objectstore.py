from tempfile import NamedTemporaryFile

import requests
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
    ) -> None:
        url = api_client.get_presigned_url(
            project_id=object_name.split("/")[0],
            method="put",
            bucket_name=self.ROOT_BUCKET_NAME,
            object_name=object_name,
        )
        self._upload_file_from_local(
            url=url, file_path=file_path, content_type=content_type
        )

    def _upload_file_from_local(
        self, url: str, file_path: str, content_type: str = "application/octet-stream"
    ) -> None:
        """_upload_file_from_local

        Args:
            url (str): The url to request.
            file_path (str): Where data is stored on the local file system.
            content_type (str): The content type of the upload request.

        Returns:
            None
        """
        requests.put(
            url=url, data=open(file_path, "rb"), headers={"content-type": content_type}
        )

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
