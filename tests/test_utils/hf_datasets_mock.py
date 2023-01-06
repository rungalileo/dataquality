from datasets import ClassLabel, Dataset, Features, Value

features = Features(
    {
        "text": Value(dtype="string", id=None),
        "label": ClassLabel(num_classes=2, names=["neg", "pos"]),
    }
)

mock_dataset = Dataset.from_dict(
    {
        "text": [
            "i didnt feel humiliated",
            "i can go from feeling so hopeless to so damned ",
            "im grabbing a minute to post i feel greedy wrong",
            "i am ever feeling nostalgic about the fireplace ",
            "i am feeling grouchy",
            "ive been feeling a little burdened lately wasnt sure why that was",
            "ive been taking or milligrams or times recommended",
            "i feel as confused about life as a teenager",
            "i have been with petronas for years i feel that petronas",
            "i feel romantic too",
            "i feel like i have to make the suffering i m seeing mean something",
            "i do feel that running is a divine experience",
            "i think it s the easiest time of year to feel dissatisfied",
            "i feel low energy i m just thirsty",
            "i have immense sympathy with the general point but as a possible",
            "i do not feel reassured anxiety is on each side",
            "i didnt really feel that embarrassed",
            "i feel pretty pathetic most of the time",
            "i started feeling sentimental about dolls",
            "i now feel compromised and skeptical",
        ],
        "label": [0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    },
    features,
)


mock_dataset_repeat = Dataset.from_dict(
    {
        "text": [
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
            "i didnt feel humiliated",
        ],
        "label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
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
