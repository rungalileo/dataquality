---
description: Overview of how users can use Galileo to improve NER models
---

# Types of NER Errors

### A_nnotation mistakes of overlooked spans_

As shown in Figure 1, observing the samples that have a high DEP score (i.e. they are hard for the model), and a non-zero count for ghost spans, can help identify samples where the annotators overlooked actual spans. Such annotation errors can cause inconsistencies in the dataset, which can affect model generalization.&#x20;

![Figure 1](<../../.gitbook/assets/ezgif.com-gif-maker (14).gif>)

### _Annotation mistakes of incorrectly labelled spans_

As shown in Figure 2, observing the subset of data with span labels in pairs with high confusion matrix and having high DEP, can help identify samples where the annotators incorrectly labelled the spans with a different class tag. Example: An annotator confused "ACTOR" spans with "DIRECTOR" spans, thereby contributing to the model biases.&#x20;

![Figure 2](<../../.gitbook/assets/ezgif.com-gif-maker (15).gif>)

### _Most frequent erroneous words across spans_

As shown in Figure 3, the insights panel provides top erroneous words across all spans in the dataset. These words have the highest average DEP across spans, and should be further inspected for error patterns. Example: "rated" had high DEP because it was inconsistently labelled as "RATING\_AVERAGE" or "RATING" by the annotators.

![Figure 3](<../../.gitbook/assets/ezgif.com-gif-maker (16).gif>)

### _Error patterns for least performing class_&#x20;

As shown in Figure 4, the model performance charts can be used to identify and filter on the least performing class. The erroneously annotated spans surface to the top.&#x20;

![](<../../.gitbook/assets/ezgif.com-gif-maker (21).gif>)

### H_ard spans for the model_

As shown in the Figure 5, the "color-by" feature can be used to observe predicted embeddings, and see the spans that are present in ground truth data, but were not predicted by the model. These spans are hard for the model to predict on&#x20;

![Figure 5](<../../.gitbook/assets/ezgif.com-gif-maker (19).gif>)

### _Confusing spans_&#x20;

As shown in Figure 6, the error distribution chart can be used to identify which classes have highly confused spans, where the span class was predicted incorrectly. Sorting by DEP and wrong tag error can help surface such confusing spans.&#x20;

![Figure 6](<../../.gitbook/assets/ezgif.com-gif-maker (18).gif>)

### _Smart features: to find malformed samples_&#x20;

As shown in Figure 7, the smart features from Galileo allow one to quickly find ill-formed samples. Example: Adding text length as a column and sorting based on it will surface malformed samples.&#x20;

![Figure 7](<../../.gitbook/assets/ezgif.com-gif-maker (20).gif>)



### &#x20;
