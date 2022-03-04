from tempfile import NamedTemporaryFile

import vaex
from minio import Minio
from vaex.dataframe import DataFrame

from dataquality.core._config import config
from dataquality.utils.file import get_file_extension


class ObjectStore:
    ROOT_BUCKET_NAME = "galileo-project-runs"

    def __init__(self) -> None:
        self.minio_client = self._minio_client()

    def _minio_client(self) -> Minio:
        try:
            local_urls = ["127.0.0.1:9000", "localhost:9000"]
            return Minio(
                config.minio_url,
                access_key=config.current_user,
                secret_key=config._minio_secret_key,
                secure=False if config.minio_url in local_urls else True,
            )
        except Exception as e:
            raise Exception(f"Error initializing minio session: {e}")

    def create_project_run_object(
        self,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.minio_client.fput_object(
            self.ROOT_BUCKET_NAME,
            object_name,
            file_path=file_path,
            content_type=content_type,
        )

    def create_project_run_object_from_df(
        self, df: DataFrame, object_name: str
    ) -> None:
        """Uploads a Vaex dataframe to Minio at the specified object_name location"""
        ext = get_file_extension(object_name)
        with NamedTemporaryFile(suffix=ext) as f:
            with vaex.progress.tree("vaex", title="Writing data for upload"):
                df.export(f.name)
            self.create_project_run_object(
                object_name=object_name,
                file_path=f.name,
            )
