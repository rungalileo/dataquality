# %%
from typing import Union

from torch import Tensor
from torch.nn import Module
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput

from dataquality.utils.torch import ModelHookManager

# %%
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
hm = ModelHookManager()
classifier_layer = hm.get_layer_by_name(model, "classifier")


def forward_hook(
    model: Module,
    model_input: Tensor,
    model_output: Union[TokenClassifierOutput, Tensor],
):
    print("model_output", model_output.shape)
    print("model_input", model_input[0].shape)


classifier_layer.register_forward_hook(forward_hook)

# %%

# %%
prediction = model(
    **tokenizer(["hello world", "beer"], return_tensors="pt", padding=True)
)


# %%

# %%
