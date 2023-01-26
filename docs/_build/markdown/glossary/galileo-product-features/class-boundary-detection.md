---
description: Detecting samples on the decision boundary
---

# Class Boundary Detection

{% hint style="info" %}
**\[**:medal: **Enterprise-Only] Class Boundary Detection is only available for Enterprise users for now.**

Stay tuned for future announcements.
{% endhint %}

Understanding a model's decision boundaries and the samples that exist near or on these decision boundaries is critical when evaluating a model's robustness and performance. A model with poorly defined decision boundaries is prone to making low confidence and erroneous predictions.

Galileo's **On the Boundary** feature highlights data cohorts that exist near or on these decision boundaries - i.e. data that the model struggles to discern between distinct classes. Identifying these samples reveals high ROI data that are not well distinguished by the model (i.e. confidently predicted as a certain class) and are likely to be poorly classified. Moreover, tracking these samples in production can reveal overlapping class definitions and signal a need for model and data tuning to better differentiate select classes.&#x20;

Within the Galileo Console, selecting the **On the Boundary** tab filters exactly the samples existing between the model's learned definition of classes:&#x20;

![Full Dataset View - Samples Colored by Class Label](<../../.gitbook/assets/image (3).png>) ![On the boundary - samples on the model's decision boundary](<../../.gitbook/assets/image (20).png>)

#### **On the Boundary Calculation**

On the boundary samples are identified by analyzing the model's output probability distribution. Given the model's output probabilities, we analyze the model's class confusion through computing per-sample certainty ratios - a metric computed as the ratio between a model's most confident predictions. Certainty ratios provide intuitive measures of class confusion not captured by traditional methods such as confidence. Through smart thresholding, we then identify samples that are particularly confused between two or more prediction classes.
