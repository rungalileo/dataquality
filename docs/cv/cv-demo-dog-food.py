from datasets import load_dataset

food = load_dataset("sasha/dog-food")

print("about to segment")
food["train"] = food["train"].select(range(10))
food["test"] = food["test"].select(range(10))
print("done segmenting")


labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

from transformers import ViTFeatureExtractor

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ToTensor

normalize = Normalize(
    mean=feature_extractor.image_mean, std=feature_extractor.image_std
)
_transforms = Compose(
    [RandomResizedCrop(feature_extractor.size), ToTensor(), normalize]
)


def transforms(examples):
    examples["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in examples["image"]
    ]
    del examples["image"]
    return examples


food = food.with_transform(transforms)

from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

from transformers import Trainer, TrainingArguments, ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)


def _test(model, x, y) -> None:
    print(x, y)


next(model.children()).register_forward_hook(_test)

print("setting train args")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)

print("making trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    tokenizer=feature_extractor,
)


print("about to train")
trainer.train()
