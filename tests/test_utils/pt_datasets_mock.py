from torch.utils.data import Dataset as TorchDataset

from tests.test_utils.mock_data import mock_dict


class CustomDatasetWithTokenizer(TorchDataset):
    def __init__(self, tokenizer, with_index=True):
        self.data = mock_dict
        self.tokenizer = tokenizer
        self.with_index = with_index

    def __getitem__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        tokenized = self.tokenizer(
            text, truncation=True, max_length=20, return_tensors="pt"
        )
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)
        tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
        tokenized["label"] = label
        if self.with_index:
            tokenized["id"] = idx
        return tokenized

    def __len__(self):
        return len(self.data["text"])
