# Similarity Search

Similarity search provides out of the box ability to discover **similar samples** within your datasets. Given a data sample, similarity search leverages the power of embeddings and similarity search clustering algorithms to surface the most contextually similar samples.

The similarity search feature can be accessed through the "Show similar" action button in both the **Dataset View** and the **Embeddings View .**

### **2 WAYS TO USE SIMILARITY SEARCH**

#### 1. Find similar labeled data across splits

This is useful when you find low quality data (mislabeled, garbage, empty, etc) and you want to find other samples similar to it, so that you can take bulk action (remove, relabel, etc). Galileo automatically assigns a smart threshold to give you the most similar data samples.

{% embed url="https://www.loom.com/share/b0761ce0c2d84c699c2146cfde2186ed" %}

While surfacing similar samples, you can easily change the number of similar samples shown within the dataset view and embeddings visualization.

{% embed url="https://www.loom.com/share/7b0853483dca4a8e865c1e4b0d7838ec" %}

#### 2. Find similar unlabeled data to train with next

This is useful when you want to search for the right unlabeled data (production data) to train with next. Examples:

a. Find unlabeled data most similar to the highest DEP (hard for the model) samples

b. Find unlabeled data most similar to an under-represented class or data split (eg: a certain gender, zip-code, etc from your meta-data)

{% embed url="https://www.loom.com/share/9637f316949f4ec399efad105272c9cd" %}

