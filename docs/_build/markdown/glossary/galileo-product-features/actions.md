# Actions

Actions help close the inspection loop and error discovery process. Currently, we support 2 major actions

1. Create a Dataset Slice
2. Export Data

### **Create a Dataset Slice**

Once you identify a data subset of interest within the Galileo Console, you can select "Save as new [slice](dataset-slices.md)**"** to specifically represent this data subset. Each Slice is accessible across different training runs, making it easy to track performance on specific data subsets over time.

&#x20;             ![](https://lh5.googleusercontent.com/5S8jFx3QINPyX-pxM61VRJkEQLHEygGmUzU9RyVh5JyiV495dmHc\_URZ4KyJJsSiKq6kLqd44TfOKFFZ\_x0jrhRgjrq5NaOWn684EVIsb5K0x59BccjMRZTfBbm1E3HsfjhtTv6lIBjm) ![](https://lh4.googleusercontent.com/hBDXWWAs4PKt48uxinkv5RVPWqbEGtFXqCm4se-OCEBlta0G2GyUSdcrQnKcF2vO3gountmZ4hGR8qG519NBHTmdeAdH-GUWx0I0UjowKNqIGOqrmOoHhghLYLIGrsG-y4lQwTACM6UB)

### **Export Data**

At any point in the inspection process you can export the data currently in scope to a CSV. We provide two 2 modes:

#### **Export selected dataset (csv)**

Exports a CSV file with the data currently in scope (i.e. displayed in the dataset view and shown in the embeddings view).

&#x20;                                               ![](<../../.gitbook/assets/Screen Shot 2021-12-14 at 8.18.14 AM.png>)&#x20;

#### **Export dataset excluding selected (csv)**

Exports a CSV file with **ALL BUT** __ the data currently in scope (i.e. the samples in scope are **removed** from the original DataFrame)

&#x20;                                               ![](<../../.gitbook/assets/Screen Shot 2021-12-14 at 8.18.45 AM.png>)****

#### _Coming soon: Exporting data to S3_&#x20;
