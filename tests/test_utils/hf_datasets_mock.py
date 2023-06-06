from datasets import ClassLabel, Dataset, Features, Value

from tests.test_utils.mock_data import mock_dict, mock_dict_repeat

features = Features(
    {
        "text": Value(dtype="string", id=None),
        "label": ClassLabel(num_classes=2, names=["neg", "pos"]),
    }
)

mock_hf_dataset = Dataset.from_dict(
    mock_dict,
    features,
)


mock_hf_dataset_repeat = Dataset.from_dict(
    mock_dict_repeat,
    features,
)


mock_dataset_numbered = Dataset.from_dict(
    {
        "text": [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ],
        "label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    },
    features=Features(
        {
            "text": Value(dtype="string", id=None),
            "label": ClassLabel(
                num_classes=20,
                names=[
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                    "13",
                    "14",
                    "15",
                    "16",
                    "17",
                    "18",
                    "19",
                ],
            ),
        }
    ),
)
