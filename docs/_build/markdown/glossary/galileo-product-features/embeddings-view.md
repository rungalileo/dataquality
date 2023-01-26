# Embeddings View

The Embeddings View provides a visual playground for you to interact with your datasets. To visualize your datasets, we leverage your model's embeddings logged during training, validation, testing or inference. Given these embeddings, we plot the data points on the 2D plane using the techniques explained below.

## Scalable Visualization

After experimenting with a host of different dimensionality reduction techniques, we have adopted the principles of UMAP \[[1](https://arxiv.org/abs/1802.03426)]. Given a high dimensional dataset, UMAP seeks to preserve the positional information of each data sample while projecting the data into a lower dimensional space (the 2D plane in our case). We additionally use a parameterized version of UMAP along with custom compression techniques to efficiently scale our data visualization to O(million) samples.  &#x20;

## Embedding View Interaction

The Embedding View allows you to visually detect patterns in the data, interactively select dataset sub populations for further exploration, and visualize different dataset features and insights to identify model decision boundaries and better gauge overall model performance. Visualizing data embeddings provides a key component in going beyond traditional dataset level metrics for analyzing model performance and understanding data quality.&#x20;

### General Navigation

Navigating the embedding view is made easy with interactive plotting. While exploring your dataset you can easily adjust and drag the embedding plane with the P_an_ tool, zoom in and out on specific data regions with S_croll to Zoom,_ and reset the visualization with the _Reset Axes_ tool_._ To interact with individual data samples, simply hover the cursor over a data sample of interest to display information and insights.

![Fig. General embeddings view navigation](../../.gitbook/assets/General\_Nav.gif)

### **Color By**

One powerful feature is the ability to color data points by different data fields e.g. `ground truth labels`, `data error potential (DEP)`, etc. Different data coloring schemes reveal different dataset insights (i.e. using color by `predicted labels` reveals the model's perceived decision boundaries) and altogether provide a more holistic view of the data. &#x20;

![Fig. Coloring by different data fields opens the door to a range of insights](../../.gitbook/assets/Color\_By.gif)

### Subset Selection

Once you have identified a data subset of interest, you can explicitly select this subset to further analyze and view insights on. We offer two different selection tools: _lasso selection_ and _box_ _select_.

After selecting a data subset, the embeddings view, insights charts, and the general data table are all updated to reflect _just_ the selected data. As shown below, given a cluster of miss-classified data points, you can make a lasso selection to easily inspect subset specific insights. For example, you can view model performance on the selected sub population, as well as develop insights into which classes are most significantly underperforming.&#x20;

![Fig. Lasso Selection](<../../.gitbook/assets/ezgif.com-gif-maker (4).gif>)

### Similarity Search

In the Embeddings View, you can easily interact with Galileo's _similarity search_ feature. Hovering over a data point reveals the "Show similar" button. When selected, your inspection dataset is restricted to the data samples with most similar embeddings to the selected data sample, allowing you to quickly inspect model performance over a highly focused data sub-population. See the[ _similarity search_](similarity-search.md) __ documentation for more details. &#x20;

![Fig. Similarity search enables quick surfacing of similar data samples](../../.gitbook/assets/Similar\_To.gif)
