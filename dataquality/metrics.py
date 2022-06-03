from typing import Optional, Dict, List

import vaex

from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.schemas.dataframe import FileType
from dataquality.schemas.split import Split

from vaex.dataframe import DataFrame

api_client = ApiClient()
object_store = ObjectStore()


def get_run_summary(
        project_name: str, run_name: str, split: Split, task: Optional[str] = None, inference_name: Optional[str] = None
) -> Dict:
    """Gets the summary for a run/split

    Calculates metrics (f1, recall, precision) overall (weighted) and per label.
    Also returns the top 50 rows of the dataframe (sorted by data_error_potential)

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    """
    pass


def get_metrics(project_name: str, run_name: str, split: Split, category: str = "gold", task: Optional[str] = None, inference_name: Optional[str] = None) -> Dict[str, List]:
    """Calculates available metrics for a run/split, grouped by a particular category

    The category/column provided (can be gold, pred, or any categorical metadata column)
    will result in metrics per "group" or unique value of that category/column

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param category: The category/column to calculate metrics for. Default "gold"
        Can be "gold" for ground truth, "pred" for predicted values, or any metadata
        column logged (or smart feature).
    :param task: (If multi-label only) the task name in question
    :param inference_name: (If inference split only) The inference split name
    """
    pass


def get_dataframe(project_name: str, run_name: str, split: Split, file_type: FileType = FileType.arrow) -> DataFrame:
    """Gets the dataframe for a run/split

    Downloads an arrow (or specified type) file to your machine and returns a loaded
    Vaex dataframe

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param file_type: The file type to download the data as. Default arrow
    """
    file_name = f"data.{file_type}"
    api_client.export_run(project_name, run_name, split, file_name=file_name)
    return vaex.open(file_name)


def get_num_epochs(project_name: str, run_name: str, split: Split) -> int:
    """Returns the number of epochs of data logged for a run/split

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    """


def get_embeddings(project_name: str, run_name: str, split: Split, epoch: int = None) -> DataFrame:
    """Downloads the embeddings for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the embeddings from the final epoch. Note that only the
    n and n-1 epoch embeddings are available for download

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param epoch: The epoch to get embeddings for. Default final epoch
    """


def get_probabilities(project_name: str, run_name: str, split: Split,
                             epoch: int = None) -> DataFrame:
    """Downloads the probabilities for a run/split at an epoch as a Vaex dataframe.

    If not provided, will take the probabilities from the final epoch.

    An hdf5 file will be downloaded to local and a Vaex dataframe will be returned

    :param project_name: The project name
    :param run_name: The run name
    :param split: The split (training/test/validation/inference)
    :param epoch: The epoch to get embeddings for. Default final epoch
    """
