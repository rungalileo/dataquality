# Natural Language Inference

[Natural Language Inference (NLI)](http://nlpprogress.com/english/natural\_language\_inference.html), also known as Recognizing Textual Entailment (RTE), is a sequence classification problem, where given two (short, ordered) documents -- `premise` and `hypothesis`, the task is to determine the inference relation between them.&#x20;

Samples are classified into one of the three labels depending on whether a `hypothesis` is true (entailment), false (contradiction), or undetermined (neutral) given a `premise`. Here's an example:

```
Premise: A man inspects the uniform of a figure in some East Asian country.
Hypothesis: The man is sleeping.
Label: contradiction


Premise: An older and younger man smiling.
Hypothesis: Two men are smiling and laughing at the cats playing on the floor.
Label: neutral


Premise: A soccer game with multiple males playing.
Hypothesis: Some men are playing a sport.
Label: entailment
```

{% hint style="info" %}
**Note**: For NLI you must combine the `premise` and `hypothesis` documents for logging. We recommend joining the document text with a separator such as `<>` to help visualization in the Galileo console.  &#x20;
{% endhint %}

### Supported frameworks:

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-pytorch.md" %}
[text-classification-pytorch.md](../../bring-your-own-model/text-classification/text-classification-pytorch.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-tensorflow.md" %}
[text-classification-tensorflow.md](../../bring-your-own-model/text-classification/text-classification-tensorflow.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-keras.md" %}
[text-classification-keras.md](../../bring-your-own-model/text-classification/text-classification-keras.md)
{% endcontent-ref %}

{% content-ref url="../../bring-your-own-model/text-classification/text-classification-huggingface.md" %}
[text-classification-huggingface.md](../../bring-your-own-model/text-classification/text-classification-huggingface.md)
{% endcontent-ref %}
