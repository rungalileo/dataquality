---
description: Here are some Galileo features to quickly help you find errors in your data
---

# NER Console Features

### 1. Rows sorted by span-level DEP scores

![](<../../.gitbook/assets/Screen Shot 2022-03-15 at 4.27.03 PM.png>)

For NER, the Data Error Potential ([DEP](../../glossary/galileo-product-features/galileo-data-error-potential-dep.md)) score is calculated at a span level. This allows rows with spans that the model had a particularly hard time with to bubble up at the top.

You can always adjust the DEP slider to filter this view and update the Insights.

### 2. Sort by 4 out-of-the-box Error types

Galileo automatically identifies whether any of the following errors are present per row:

a. **Span Shift:** A count of the misaligned spans that have overlapping predicted and gold spans

b. **Wrong Tag:** A count of aligned predicted and gold spans that primarily have mismatched labels

c. **Missed Span:** A count of the spans that have gold spans, but no corresponding predicted spans

d. **Ghost Span:** A count of the spans that have predicted spans, but no corresponding gold spans

### 3. Explore the most frequent words with the highest DEP Score

![](<../../.gitbook/assets/Screen Shot 2022-03-15 at 4.40.16 PM.png>)

Often it is critical to get a high level view of what specific words the model is struggling with most. This NER specific insight lists out the words that are most frequently contained within spans with high DEP scores.

Click on any word to get a filtered view of the high DEP spans containing that word.

### 4. Explore span-level embedding clusters

![](<../../.gitbook/assets/Screen Shot 2022-03-15 at 4.34.51 PM.png>)

For NER, [embeddings](../../glossary/galileo-product-features/embeddings-view.md) are at a span level as well (that is, each dot is a span).

Hover over any region to get a list of spans and the corresponding DEP scores in a list.

Click the region to get a detailed view for a particular span that has been clicked.

### 5. Find similar spans

![](<../../.gitbook/assets/Screen Shot 2022-03-15 at 4.36.15 PM.png>)

We leverage the Galileo [similarity clustering](../../glossary/galileo-product-features/similarity-search.md) to find all similar samples to a particular span quickly -- select a span and click the 'Similar to' button.

### 6. Remove and re-label rows/spans by adding to the Edits Cart

![](<../../.gitbook/assets/Screen Shot 2022-03-15 at 4.37.56 PM.png>)

After every run, you might want to prune your dataset to either

a. Prep it for the next training job

b. Send the dataset for re-labeling

You can think of the 'Edits Cart' as a means to capture all the dataset changes done during the discovery phase (removing/re-labeling rows and spans) to collectively take action upon a curated dataset.

### 7. Export your filtered dataset to CSV

At any point you can export the dataset to a CSV file in a easy to view format.

