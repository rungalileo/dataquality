import pandas as pd
from torch.utils.data import Dataset

mock_dict = {
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
}

mock_dict_repeat = {
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
}


# Step 1: Create your mock dataframe
data = {
    "label": ["Sports", "Tech", "Politics"] * 2,
    "text": [
        "This is a sample text about {}.".format(i)
        for i in ["Sports", "Tech", "Politics"] * 2
    ],
}
mock_df = pd.DataFrame(data)

# Step 2: Split dataframe into training and testing data
train_df, test_df = mock_df, mock_df
labels = pd.concat([train_df["label"], test_df["label"]]).unique()
labels.sort()
label_map = {label: i for i, label in enumerate(labels)}


# Define dataset class
class MockDataset(Dataset):
    labels = labels

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self.labels = labels

    def __getitem__(self, index: int) -> tuple:
        label, text = self.dataframe[["label", "text"]].iloc[index]
        return label_map[label], text

    def __len__(self) -> int:
        return len(self.dataframe)
