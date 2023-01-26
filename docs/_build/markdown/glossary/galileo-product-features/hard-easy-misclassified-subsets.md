# Hard/Easy/Misclassified Subsets

Throughout this section we use an example dataset of 25000 samples to showcase term definitions. For future reference, we provide a snapshot of the Galileo Console on the _full dataset._ In later examples we will draw specific connections within the insights pane on the right.

![Fig. Full dataset view](../../.gitbook/assets/Full\_Data.png)

### **Hard Subset**

The subset of data samples with highest DEP scores, thereby "hard" for the model to learn from during training or "hard" for the model to make predictions on at test time. These samples can be hard due to one or more of the following reasons: _boundary samples_, _noisy / corrupt samples_, _mislabelled samples_, _misclassified samples_, _out-of-distribution samples_ etc.&#x20;

The figure below highlights the "Hard for the model" data subset (4371 samples or 17% of the dataset). Using the insights pane we confirm that the _hard_ data subset has significantly higher per class DEP scores than the full dataset from above (average DEP 0.67 vs. 0.33). Additionally, we see that the _hard_ **** samples are in fact hard for the model by observing significantly lower per class F1 scores (see "Model performance" chart) and average F1 score (0.31 << 0.83).&#x20;

![Fig. Hard for the model (insights on the right show poor model performance)](../../.gitbook/assets/Hard\_Samples.png)

### **Easy Subset**

The subset of data samples with lowest DEP scores, thereby "easy" for the model to learn from during training, or "easy" for the model to make predictions on at test time. Typically these "easy" samples are clean, noise free data samples that the model had no issues training on.&#x20;

The figure below showcases the "Easy for the model" data subset (14119 samples or 56% of the dataset). In contrast to the _hard_ data, the insights pane shows that the _easy_ data have lower per class DEP scores than the full data (average DEP 0.6 vs. 0.33). Moreover, the _easy_ **** subset is significantly easier than _full_ dataset and _hard_ subset, as showcased by the higher per class F1 scores (see "Model performance" chart) and average F1 score (0.99 >> 0.83 >> 0.31).&#x20;

![Fig. Easy for the model (insights on the right show strong model performance)](../../.gitbook/assets/Easy\_Samples.png)



### **Misclassified Subset**

The subset of data samples where the ground truth class does not match the predicted class. In other words, the prediction errors of the model. Galileo makes it very easy to surface samples where the model prediction differed from the ground truth. In a click of a button, Galileo shows all misclassified samples.

![Fig. Misclassified data samples](../../.gitbook/assets/Missclassified.png)

### Smart Thresholding

We leverage smart thresholding based on DEP score values and distributions to segregate datasets into corresponding "easy" and "hard" subsets. The Galileo Console enables you to surface these subsets at the click of a button.
