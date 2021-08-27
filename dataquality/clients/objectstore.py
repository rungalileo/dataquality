from minio import Minio
from pydantic.types import UUID4

from dataquality import config


class ObjectStore:
    ROOT_BUCKET_NAME = "galileo-project-runs"

    def __init__(self) -> None:
        self.minio_client = self._minio_client()

    def _minio_client(self) -> Minio:
        try:
            return Minio(
                config.minio_url,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key.get_secret_value(),
                secure=False if config.minio_url == "127.0.0.1:9000" else True,
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
        """https://docs.min.io/docs/python-client-api-reference.html#fput_object"""
        self.minio_client.fput_object(
            self.ROOT_BUCKET_NAME,
            f"{project_id}/{run_id}/{object_name}",
            file_path=file_path,
            content_type=content_type,
        )


object_store = ObjectStore()
