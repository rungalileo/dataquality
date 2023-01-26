# Insights Panel

Galileo provides a dynamic _Insights Panel_ that provides a bird's eye view of your model's performance on the data currently in scope. Specifically, the Insights Panel displays:

* Overall model and dataset metrics
* Class level model performance
* Class level DEP scores
* Class distributions&#x20;
* Top most misclassified pairs

The Insights Panel allows you to keep a constant check on model performance as you continue the inspection process (through the [Dataset View](dataset-view.md) and [Embeddings View](embeddings-view.md)).

&#x20;                 ![](<../../.gitbook/assets/image (25).png>)![](<../../.gitbook/assets/image (11).png>)

### Model and Dataset Metrics

The top of the Insights Panel displays aggregate model performance (default F1) and allow you to select between Precision, Recall, and F1. Additionally, the Insights Panel shows the number of current data samples in scope along with what % of the total data is represented.

![Fig. Top level data insights](<../../.gitbook/assets/image (39).png>)

### Class Level Model Performance

Based on the model metric selected (F1, Precision, Recall), the "Model performance" bar chart displays class level model performance.

![Fig. Class level model performance based on model metric (F1 by default)](<../../.gitbook/assets/image (1) (1).png>)

### Class Level DEP Scores

The Data Error Potential (DEP) chart shows the per class average DEP score distribution for the data currently in scope. The aggregate average DEP score is shown with a dotted black line.&#x20;

![Fig. Data Error Potential (DEP) chart showing per class DEP score distribution](<../../.gitbook/assets/image (33).png>)

### Class Distribution

The Class Distribution chart shows the breakdown of samples within each class. This insights chart is critical for quickly drawing insights about the class makeup of the data in scope and for detecting issues with class imbalance.&#x20;

![Fig. Class Distribution plot](<../../.gitbook/assets/image (27).png>)

### Top most misclassified pairs

At the bottom of the Insights Panel we show the "Top five 5 most misclassified data label pairs", where each pair shows a gold label, the incorrect prediction label, and the number of samples falling into this misclassified pairing. This insights chart provides a snapshot into the most common mistakes made by the model (i.e. mistaking ground truth label X for prediction label Y).

![Fig. Top 5 misclassified label pairs - surfaces the most common mistakes made by the model](<../../.gitbook/assets/image (24).png>)

### Interacting with Insights Charts

In addition to providing visual insights, each insights chart can also be interacted with. Within the "Model performance", "Data Error Potential (DEP)", and "Class distribution" charts selecting one of the bars restricts the data in scope to data with `Gold Label` equal to the selected `bar label`.&#x20;

An even more powerful interaction exists in the "Top 5 most misclassified label pairs" panel. Clicking on a row within this insights chart filters for _misclassified data_ matching the `gold label` and `prediction label` of the misclassified label pair.&#x20;

![Fig. Interaction with "Most misclassified label pairs" chart allows for quick dataset filtering](../../.gitbook/assets/misclassified\_pairs.gif)
