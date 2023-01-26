# XRay Insights

{% embed url="https://www.loom.com/share/68bee18d72cb4dfa80bdb6c389fdea69" %}

## What are Galileo XRay Insights?

Galileo XRay is your personal sidekick while you inspect your datasets. After navigating to a run, XRay will provide you with immeiately valuable insights, and suggested actions you can take to address critical data errors.&#x20;

You can think of XRay as a partner Data Scientist working with you to find and fix your data.

## XRay Insights that we support today

We support a growing list of insights, and are open to feature requests!

#### Low Performing Classes

This insight will inform you of any classes in your data with an F1 score more than 0.05 lower than the average.

#### Low Performing High Imbalanced Classes

This insight will inform you of any classes in your data with an F1 score more than 0.05 lower than the average and more than 5X smaller than the largest class in your dataset.

#### Low Performing Metadata Categories

For all logged metadata categories (including Galileo Smart Features), this insight will illuminate particular values of those categories with an F1 score more than 0.05 lower than the average for that category.

#### Worse than Random Guessing

If your model is performing statistically worse than random guessing, we will let you know!

#### PII Data

If more than 1% of your data has PII, this insight will keep you informed.

#### Non-Primary Language Data

If more than 1% of your data is not of that of your primary language, this insight will alert you.

#### Non-Primary Low Performing Language Data

For all of your non-primary languages, this insight will check if any of them have an F1 score of more than 0.05 less than your average F1.

#### Low Performing Empty Samples

This insight checks to see if you have empty samples in your data which are performing poorly

#### NER Missed-Annotation Spans

(NER Only) This insight will look for spans that are likely misannotations (predicted correctly by the model, but labeled incorrectly).

## How to request a new insight?

Have a great idea for an insight? We'd love to hear about it! Simply click "Request New Insight" from the XRay navigation, and we'll immediately get your request :telescope:

![](<../../.gitbook/assets/image (13).png>)

![](<../../.gitbook/assets/image (12).png>)
