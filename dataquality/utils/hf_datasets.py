import pandas as pd
from datasets import Dataset

def load_pandas_df(data:Dataset) -> pd.DataFrame:
  # Load the labels in a dictionary
 
  labels = data.features['label'].names
  labels = {v:k for v, k in enumerate(labels)}

  # Load the train data into a frame
  data_df = pd.DataFrame.from_dict(data)
  data_df['label'] = data_df['label'].map(labels)
  data_df['id'] = data_df.index
  return data_df
