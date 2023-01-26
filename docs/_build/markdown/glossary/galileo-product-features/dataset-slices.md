# Dataset Slices

"Slices" is a powerful Galileo feature that allows you to monitor, across training runs, a sub-population of the dataset based on metadata filters.&#x20;

### Creating Your First Simple Slice

Imagine you want to monitor model performance on samples containing the keyword "star wars." To do so, you can simply type "star wars" into the search panel and save the resulting data as a new custom **Slice** (see Figure below).

![Fig. Slice for reviews with "star wars" in it](<../../.gitbook/assets/Screen Shot 2021-12-14 at 7.57.06 AM.png>)

When creating a new slice you are presented a pop up that allows you to give a **custom name** to your slice and displays slice level details: 1) Slice project scope, 2) Slice Recipe (filter rules to create the slice). Your newly created slice will be available across all training runs within the selected project.&#x20;

&#x20;                                                ![](<../../.gitbook/assets/image (16).png>)

### Complex Slices

You can create a custom slice in many different ways e.g. using [similarity search](similarity-search.md), using [subsets](hard-easy-misclassified-subsets.md) etc. Moreover, you can create complex slices based on multiple filtering criteria. For example, the figure below walks through creating a slice by first using similarity search and then filtering for samples that contain the keyword "worst."

![Fig. Creation of complex slice (Recipe: Similar to (with 880 samples) + Search keyword(s) = "worst")](../../.gitbook/assets/Complex\_Slice.gif)

The final "Slice Recipe" is as follows:

&#x20;                                               ![](<../../.gitbook/assets/image (9).png>)
