# Multi Label Text Classification

[Multi-label text classification](https://en.wikipedia.org/wiki/Multi-label\_classification) (MLTC), also known as multi-output text classification is a variant of the text classification problem, where multiple labels are assigned to each sample. It is a generalization of [multiclass text classification](broken-reference), where a single label is assigned to each sample.&#x20;

Samples are assigned a subset of the available label classes, where there are no constraints on how many classes a sample can be assigned. We refer to the set of available label classes as tasks and behind the scenes, Galileo treats assigning each class (a task) as a binary prediction problem - 1 if the given class is assigned, 0 otherwise. Here's an example:

```
Input: Now I'm wondering on what I've been missing out. Again thank you for this.
Output: Curosity, Gratitude

Input: That is odd.
Output: Disappointment, Disgust
```

### Supported frameworks:

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-pytorch.md" %}
[text-classification-pytorch.md](../../bring-your-own-model/text-classification/text-classification-pytorch.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-tensorflow.md" %}
[text-classification-tensorflow.md](../../bring-your-own-model/text-classification/text-classification-tensorflow.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-huggingface.md" %}
[text-classification-huggingface.md](../../bring-your-own-model/text-classification/text-classification-huggingface.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-keras.md" %}
[text-classification-keras.md](../../bring-your-own-model/text-classification/text-classification-keras.md)
{% endcontent-ref %}
