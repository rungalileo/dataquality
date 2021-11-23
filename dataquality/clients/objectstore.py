import os
from typing import Any, Dict

from minio import Minio
from pydantic.types import UUID4
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.loggers import JsonlLogger


class ObjectStore:
    ROOT_BUCKET_NAME = "galileo-project-runs"

    def __init__(self) -> None:
        self.minio_client = self._minio_client()

    def _minio_client(self) -> Minio:
        try:
            local_urls = ["127.0.0.1:9000", "localhost:9000"]
            return Minio(
                config.minio_url,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                secure=False if config.minio_url in local_urls else True,
            )
        except Exception as e:
            raise Exception(f"Error initializing minio session: {e}")

    def create_project_run_object(
        self,
        project_id: UUID4,
        run_id: UUID4,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.minio_client.fput_object(
            self.ROOT_BUCKET_NAME,
            f"{project_id}/{run_id}/{object_name}",
            file_path=file_path,
            content_type=content_type,
        )

    def create_project_run_object_from_df(
        self, df: DataFrame, object_name: str
    ) -> None:
        """Uploads a Vaex dataframe to Minio at the specified object_name location"""
        tmp_filename = f"{JsonlLogger.LOG_FILE_DIR}/{object_name}.tmp"
        df.export_arrow(tmp_filename)
        assert config.current_project_id is not None
        assert config.current_run_id is not None
        self.create_project_run_object(
            project_id=config.current_project_id,
            run_id=config.current_run_id,
            object_name=object_name,
            file_path=tmp_filename,
        )
        os.remove(tmp_filename)

    def get_fs_options(self) -> Dict[str, Any]:
        return dict(
            endpoint_override=config.minio_url,
            scheme="https" if config.minio_url.startswith("https") else "http",
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            region=config.minio_region,
        )


object_store = ObjectStore()
