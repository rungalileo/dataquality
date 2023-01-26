# Model Monitoring and Data Drift with Production or Unlabeled Data

Once your model is in production, it is essential to monitor its health:&#x20;

{% embed url="https://www.loom.com/share/60282fceade44882b0611b5d4c3216c1" %}
Production data monitoring with Galileo
{% endembed %}

> _Is there training<>production data drift? What unlabeled data should I select for my next training run? Is the model confidence dropping on an existing class in production? ..._&#x20;

To answer the above questions and more with Galileo, you will need:

1. Your unlabeled production data
2. Your model

### ⚡️Simply run an [inference job](inference-run-with-galileo-a-quick-guide.md) on production data to view, inspect and select samples directly in the Galileo UI.&#x20;

Here is what to expect:

• Get the list of [**drifted data samples**](../../glossary/galileo-product-features/data-drift-detection.md) **out of the box**

• Get the list of [**on the class boundary**](../../glossary/galileo-product-features/class-boundary-detection.md) **samples out of the box**

• Quickly **compare model confidence and class distributions** between production and training runs

• Find **similar samples to low confidence production data** within less than a second

... and a lot more

How to run an inference job with Galileo? [Start here](inference-run-with-galileo-a-quick-guide.md)
