---
description: >-
  Let's dig deeper with a sentiment classification run on a movie reviews
  dataset
---

# Using Galileo effectively for Multi Class use cases

### _Summarizing model performance_

As shown in Figure 1, the overall model performance is 0.83 (F1), with the least performing classes being `positive` with 0.78 (F1), and `negative` with 0.81 (F1). The aggregate test data performance is 0.69 (F1), with the highest class generalization error matching the per class training set performances.&#x20;

![Figure 1](../../.gitbook/assets/Final1.gif)

Figure 2 shows that the dataset is quite balanced. Scrolling down to the the bottom of the insights panel, we see the most commonly misclassified pairs: `positive` with `very positive`, and `negative` with `very negative`.&#x20;

![Figure 2](../../.gitbook/assets/Final2.gif)

### _**Inspecting decision boundaries**_

In Figure 3, we show the embeddings visualization. This view visualizes your dataset samples as the model sees them (i.e. based on the model's embeddings of the data), separated and clustered in the embedding space.

We use the "Color By" functionality to visually inspect the _latent decision boundary_ by toggling between _ground truth labels_ and _predicted labels_.&#x20;

The visualization differences suggests the model is ineffective at separating the overlapping regions between classes - e.g. `positive` with `very positive`, and `negative` with `very negative`.  The resulting decision boundary corroborates the most confused classes shown in the insights panel. We note that this same affect can be observed on test data.&#x20;

![Figure 3](<../../.gitbook/assets/Final3 (1).gif>)

### _**Summarizing data quality**_

As shown in Figure 4, the training dataset has 25,000 samples. Of these 25,000 samples we identify 17% as "Hard for the model" and 16% as "Misclassified". Thus, 1% of the data is "Hard for the model" and "Correctly classified," requiring an even closer inspection.&#x20;

The majority of samples in the "Hard for the model" subset are `negative` or `positive`, with an overall performance of 0.31 (F1). We contrast that with the samples that are  "Easy for the model," primarily coming from the  `very negative` or `very positive` classes, and have an overall performance of 0.99 (F1). For this dataset, the extreme classes are always relatively easy to differentiate, and hence achieve maximal performance.

![Figure 4](../../.gitbook/assets/Final4.gif)

Figure 5 demonstrates how adjusting the [Data Error Potential (DEP)](../../glossary/galileo-product-features/galileo-data-error-potential-dep.md) score range to higher thresholds result in greater F1 drop; moreover, we observe an  increased class imbalance (confirming that easier samples from the `very positive` and `very negative` classes have been filtered out).&#x20;

![Figure 5](../../.gitbook/assets/Final5.gif)

### _Identifying annotation errors_

The samples with high DEP score generally belong to low performing regions of the model. As shown in Figure 6, samples with the highest DEP score are most likely annotation errors. Through the Galileo Console you can mark these samples individually, both for training and test data. The marked samples can for example be exported as _csv files_ for further investigation.&#x20;

![Figure 6](../../.gitbook/assets/Final6.gif)

### _**Identifying high ROI samples**_

Correctly classified hard samples can be either **very detrimental** to the model (as it indicates overfitting to samples with annotation errors) or **very useful** for the model (as collecting more samples similar to these can help the model perform better over low performing regions).&#x20;

As shown in Figure 7, within the "Hard for the model" data subset, the `very positive` class has the lowest performance 0.45 (F1). If we select samples that belong to this class and are correctly classified, further inspection reveals these samples either start with satire, start by refuting a previous negative comment, or use negative words to describe the plot of the movie but do not rate the movie itself negative. Collecting additional data with similar attributes and samples can help improve the model performance.&#x20;

![Figure 7](../../.gitbook/assets/Final7.gif)

### _**Inspecting low-performing model regions**_

As shown in Figure 8, Coloring By DEP score can provide insights on the overall low-performing model regions. Here we see that the highlighted samples (high DEP score), which fall near overlapping class boundaries, are causing issues for our model. This suggests that getting more data points similar to these will improve class distinction. The highlighted samples away from decision boundaries suggest likely annotation errors which need to be inspected manually. Fixing these will likely also improve model performance.&#x20;

To inspect the areas of interest described above, you can use the _lasso-select / box-select_ features. In Figure 8 we highlight critical annotation errors near overlapping classes. Fixing these annotation errors is crucial as these samples act as support samples (for defining the decision boundary) and will improve model generalization.&#x20;

![Figure 8](../../.gitbook/assets/Final8.gif)

### _**Inspecting generalization gap**_

As demonstrated in Figure 9, you can toggle between the training and test data embedding representation to find test samples that are out of distribution and lead to generalization gap. The selected test samples have a performance of 0.31 (F1). Collecting more training samples similar to them will likely improve performance.&#x20;

![Figure 9](../../.gitbook/assets/Final9.gif)

### _**Inspecting outliers**_

The embedding view can help you quickly inspect outlier samples. With Galileo, we quickly found that the outlier region, highlighted in Figure 10, did not contain movie reviews at all. Rather, the highlighted data samples are wrestling tournament reviews, thus confusing the model and resulting in poor model performance on this subset 0.57 (F1).&#x20;

![Figure 10](../../.gitbook/assets/Figure11.gif)

### _**Inspecting slices of data**_&#x20;

The Galileo Dataset View allows you to search for specific keywords in the input text, and track the performance for those samples. In Figure 11 we demonstrate searching for genre keywords. We see that "thriller" is the easiest genre with 0.73 (F1), "romance" has middle range performance of 0.68 (F1), and "sci fi" has the lowest performance of 0.54 (F1).&#x20;

Galileo then enables you to save these metadata filters as custom slices that can be compared across various runs in a project. Reach out to us for learning more about our full enterprise software!

![Figure 11](<../../.gitbook/assets/Final10 (1).gif>)

### _**Find similar samples**_

As shown in Figure 12, we used the [Quick Similarity search ](../../glossary/galileo-product-features/similarity-search.md)feature on the console, to get similar samples to one that was incorrectly annotated. The new subset also had many annotation errors and very low F1 (0.25). So we saved the subset as a [custom slice](../../glossary/galileo-product-features/dataset-slices.md) of `potential errors` for further inspection.

![](<../../.gitbook/assets/ezgif.com-gif-maker (12).gif>)

### _**Inspecting long samples**_

**Coming soon**
