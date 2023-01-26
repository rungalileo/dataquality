# Galileo Data Error Potential (DEP)

It is crucial to quickly identify errors in ML training data and fix them. This is incredibly hard to do at scale when working with millions of data points.

Today teams typically leverage model confidence scores to separate well trained from poorly trained data. This has two major problems:

* **Confidence scores** are highly model centric. There is high bias towards training performance and very little use of inherent data quality to segregate the good data from the bad (results below)&#x20;
* Even with powerful pre-trained models, confidence scores are unable to capture nuanced sub-categories of data errors (details below)

The **Galileo Data Error Potential (DEP)** score has been built to provide a per sample holistic data quality score to identify samples in the dataset contributing to low or high model performance i.e. ‘pulling’ the model up or down respectively. In other words, the DEP score measures the potential for "misfit" of an observation to the given model.

Categorization of "misfit" data samples includes:&#x20;

* Mislabelled samples (annotation mistakes)&#x20;
* Boundary samples or overlapping classes&#x20;
* Outlier samples or Anomalies&#x20;
* Noisy Input&#x20;
* Misclassified samples&#x20;
* Other errors

This sub-categorization is crucial as different dataset actions are required for each category of errors. For example, one can augment the dataset with samples similar to boundary samples to improve classification.&#x20;

As shown in below, we assign a DEP score to every sample in the data. The _Data Error Potential (DEP) Slider_ can be used to filter samples based on DEP score, allowing you to filter for samples with DEP greater than x, less than y, or within a specific range \[x, y].&#x20;

![Galileo Platform surfaces mislabeled, garbage samples by ordering in desc order of DEP score](<../../.gitbook/assets/Screen Shot 2021-12-20 at 11.25.44 PM.png>)

#### **DEP score calculation**

The base DEP score calibration is done by calculating a hybrid ‘**Area Under Margin’ (AUM)** mechanism. AUM is the cross-epoch average of the model uncertainty for each data sample (calculated as the difference between the ground truth confidence and the maximum confidence on a non ground truth label).

We then dynamically leverage techniques like K-Distinct Neighbors, IH Metrics (multiple weak learners) and Energy Functions on Logits to clearly separate out annotator mistakes from samples that are confusing to the model or are outliers and noise. This is dynamic because DEP takes into account the level of class imbalance, variability etc to cater to the nuances of each dataset.

#### **DEP score efficacy**&#x20;

To measure the efficacy of the DEP score, we performed experiments on a public dataset and induced varying degrees of noise. We observed that unlike Confidence scores, the DEP score was successfully able to separate bad data (red) from the good (green). This demonstrates true data-centricity (model independence) of Galileo’s DEP score. Below are results from experiments on the public Banking Intent dataset. The dotted lines indicate a dynamic thresholding value (adapting to each dataset) that segments noisy (red) and clean (green) samples of the dataset.

| Galileo DEP score                                                                                                                                                                                                  | Model confidence score                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="https://lh6.googleusercontent.com/A2KgtvDq8g33KJ54gmXDxYLzNcprvRR6S-fu6hBvPr4lgkp6kWt6Bp105sYUq_RMLw4VOBIc7LR6KuX6LdHYgDmGa3JIIu0TykmD6udohMOqJJ432D5aNZCeR9ay2aw7cE7aakhy" alt="" data-size="original"> | ![](https://lh4.googleusercontent.com/G0q5322ZSpH7rQjCvch57ooOEWCNZZ\_3iyHWts-C9OxP6rf5o73zLAjPiDCuoCrFKq-qQK5TWAl6dFSd2EFRMHUfkDpGcdbvX3K1gKh4s9DpzguUUKRfiNT6K\_xwmRsupjCmPeIE) |
| Noise recall: 99.2%                                                                                                                                                                                                | Noise recall: 14.4%                                                                                                                                                               |

&#x20;                                                          10% Noise added to the dataset

| Galileo DEP score                                                                                                                                                                 | Model confidence score                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ![](https://lh6.googleusercontent.com/S2LPSsMojuZM7CXD7gbbKaS1Atw9yI7-t445TgRzyNOg98OzOtHPMmbg\_gqOJaIjh5p2rRrfZoKWm5y8M0Eswaums\_fymlN62YDY951ULnuvYhoHd-1EU25KwVvkzDBWvrsQUECl) | ![](https://lh5.googleusercontent.com/QUTn5Bpn\_PbjK1JdV3w3RryX-fvtkXh1xUM-T4dvMK5UhGyubyGuMnmj\_YVCxR4y6lKNCKZTnM3NpxShXpb5c0sxdJl6RPP\_IWLEFY3W3lFNGq0RyF-0Lw968jqMom9GoK4LVoMt) |
| Noise recall: 89.0%                                                                                                                                                               | Noise recall: 09.3%                                                                                                                                                                |

&#x20;                                                           20% Noise added to the dataset
