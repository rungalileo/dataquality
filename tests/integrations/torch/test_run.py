# %%
import numpy as np
import dataquality as dq
from dataquality.schemas.split import Split

dq.set_console_url("https://console.dev.rungalileo.io")
dq.config.token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmcmFuekBydW5nYWxpbGVvLmlvIiwiaWQiOiJiMmY0MWNkOS0xNDVhLTRhODUtOWQ2OC01ZDIxNTE2ZGJjNzEiLCJleHAiOjE2ODUyMTIzNjJ9.HyPX2DBw0lfrtpGUqTkn-SqiGW2n4by7cj_3ykXAJMA"
dq.init(task_type="image_classification")
labels = ["a", "b"]
dq.set_labels_for_run(labels)
dq.config.allow_missing_in_df_ids = True
dq.log_data_samples(
    texts=["a", "b", "a"],
    labels=["a", "b", "a"],
    ids=[0, 1, 2],
    split=Split.training,
)
dq.log_model_outputs(
    embs=np.array([[0, 0], [1, 1]]),
    ids=[0, 1],
    probs=[[0, 1], [1, 0]],
    split=Split.training,
    epoch=0,
)
dq.finish()

# %%
