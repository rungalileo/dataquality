---
description: '"Garbage in, Garbage out"'
---

# Likely Mislabeled

Training ML models with noisy, mislabeled data can dramatically affect model performance. Dataset errors easily permeate the training process, leading to issues in convergence, inaccurate decision boundaries, and poor model generalization.&#x20;

On the evaluation side, mislabeled data in a test set will also hurt the ML model's performance, often resulting in lower benchmark scores. Since this is one the biggest factor in deciding whether a model is ready to deploy, we cannot overstate the importance of also having clean test sets.

Therefore, identifying and fixing labeling errors is extremely crucial for both training effective and reliable ML models, and evaluating them accordingly. However, accurately identifying labeling errors is challenging and deploying ineffective algorithms can lead to large, manual efforts with little realized return on investment.

Galileo's mislabel detection algorithm addresses these challenges by employing state of the art statistical methods for identifying data that are highly likely to be _mislabeled_. In the Galileo Console, these samples can be accessed through the _Likely Mislabeled_ data tab.

In addition, we surface a tunable parameter which allows the user to fine-tune the method for their use case. The slider balances between precision (minimize number of mistakes) and recall (maximize number of mislabeled samples detected). Hovering over the slider will display a short description, while hovering over the thumb button displays the number of likely mislabeled samples to expect in that position.

For illustration, we highlight a few data samples from the [**Conversational Intent**](https://www.kaggle.com/datasets/joydeb28/nlp-benchmarking-data-for-intent-and-entity) dataset that are correctly identified as mislabeled.&#x20;

<figure><img src="../../.gitbook/assets/Screenshot 2022-11-28 at 2.34.07 PM.png" alt=""><figcaption></figcaption></figure>

### Adjusting the slider for your use-case

The _Likely Mislabeled_ slider allows the user to fine-tune both the qualitative and quantitive output of the algorithm, depending on your use-case.&#x20;

<figure><img src="../../.gitbook/assets/Screenshot 2022-11-28 at 3.11.48 PM.png" alt=""><figcaption></figcaption></figure>

On one extreme it will optimize for maximum Recall: this maximizes the number of mislabeled samples caught by the algorithm and in most cases ensures 90% of mislabeled points caught (see results below).

On the other extreme it will optimize for maximum Precision: this minimizes the number of errors made by the algorithm, i.e., it minimizes the number of datapoints which are not mislabeled but are marked as likely mislabeled.

#### Setting the threshold for a common use-case: fixed re-labelling budget

Suppose that we have a relabelling budget of only 200 samples. Start with the slider on the Recall side where the algorithm returns all the samples that are likely to be mislabeled. As you move the thumb of the slider towards the Precision side, a hovering box will appear and you should notice the number of samples decreasing, allowing you to fine-tune the algorithm for returning the 200 samples that are most likely to be mislabeled.

### Likely Mislabeled Computation

Galileo's _Likely Mislabeled_ _Algorithm_ is adapted from the well known '**Confident Learning**' algorithm. The working hypothesis of confident learning is that counting and comparing a model's "confident" predictions to the ground truth can reveal class pairs that are most likely to have class confusion. We then leverage and combine this global information with per-sample level scores, such as [DEP](galileo-data-error-potential-dep.md) (which summarizes individual data sample training dynamics), to identify samples most likely to be mislabeled.

This technique particularly shines in multi-class settings with potentially overlapping class definitions, where labelers are more likely to confuse specific scenarios.

### DEP vs. Likely Mislabeled

Although related, [Galileo's DEP score](galileo-data-error-potential-dep.md) is distinctly different from the _Likely Mislabeled_ algorithm: samples with a higher DEP score are not necessarily more likely to be mislabeled (even though the opposite is true). While _Likely Mislabeled_ focuses solely on the potential for being mislabeled, DEP more generally measures the potential for "misfit" of an observation to the given model. As described in our documentation, the categorization of "misfit" data samples includes:

* _Mislabeled_ _samples_ (annotation mistakes)&#x20;
* Boundary samples or overlapping classes&#x20;
* Outlier samples or Anomalies&#x20;
* Noisy Input&#x20;
* Misclassified samples&#x20;
* Other errors

Through summarizing per-sample training dynamics, DEP captures and categorizes _many_ different sample level errors without specifically differentiating / pinpointing a specific one.&#x20;

### **Likely Mislabeled evaluation**&#x20;

To measure the effectiveness of the _Likely Mislabeled_ algorithm, we performed experiments on 10+ datasets covering various scenarios such as binary/multi-class text classification, balanced/unbalanced distribution of classes, etc. We then added various degrees of noise to these datasets and trained different models on them. Finally, we evaluated the algorithm on how well it is able to identify the noise manually added.

Below are plots indicating the Precision and Recall of the algorithm.&#x20;

{% tabs %}
{% tab title="10-20% Noise" %}
<figure><img src="https://lh3.googleusercontent.com/MBtOzk6XX471fmI7pz_S_78GHBAIWsrmtPlQJAmQE68GQ2eqQo01qRqPKksazJBqE2xQ2NMvNFTW5UsbW6_DCaMTtZwPBs-os4sHr6VoeFShdKyH3_Xw2sQLsYgRvGeDO-9BWq2pbkJ9VAgrWgItUecrbhUPXEwt3O09Gttvpxda7vAQH-3-wJ5KIE62bQ" alt=""><figcaption><p>The horizontal alignment of the bars matches the position of the slider: the bars to the left are for better Precision and the bars to the right for better Recall.</p></figcaption></figure>
{% endtab %}

{% tab title="5-10% Noise" %}
<figure><img src="https://lh6.googleusercontent.com/L5DXc2Tgiyb4lUW57FoFbqx74LwwKHgu785gEE4gNFYqg0uDbWTSMw4onVmxub9b2SI0xJdHiKXyXd5NvKF6ka8dWvHRG7fK9DP5ODS4xwHTPAenr7IngvCprflQn-Zrw66EkiaUaTK80BV475ApFLMTiPU8RF3uxiULUJbJAP_W0_7AVEyOwhUuj7QasA" alt=""><figcaption><p>The horizontal alignment of the bars matches the position of the slider: the bars to the left are for better Precision and the bars to the right for better Recall.</p></figcaption></figure>
{% endtab %}

{% tab title="2-5% Noise" %}
<figure><img src="https://lh6.googleusercontent.com/gLj-OJbTFiAkD46zwQjDWVZKvCGnYQvKMRmpsy2TqitDJx8vyCILu8-TJkKG0Rt9Piw6kVMQOk-K2IAQ6yRL7qsxTIcojX4oAjxGJfUgUTNJOotGmk-tfbI5w-Rs2rSsVKZhCjzwrIwhuoj7oN4z62smZDvYiQXsF8NJGWoIDzaw0zo_xMqYScoprV4g3Q" alt=""><figcaption><p>The horizontal alignment of the bars matches the position of the slider: the bars to the left are for better Precision and the bars to the right for better Recall.</p></figcaption></figure>
{% endtab %}

{% tab title="0-2% Noise" %}
<figure><img src="https://lh3.googleusercontent.com/RB9r0osGlb9B6Gnn92aPB5__6HAfjGDrKv8shmMwuSjf0tIOp6ly7CHCYinUgWL5-uwcqf0FMKLqegEfmKrc3hBStp2VzZx_uyuCb82PBGq75y0_TXquSrYZq9S5QEVt6Z1mPIFP8i_nFfQ-1l1Pm99zHhM7snTw9M55WNoYGreDl5pphW-CYmKaCOIuyw" alt=""><figcaption><p>The horizontal alignment of the bars matches the position of the slider: the bars to the left are for better Precision and the bars to the right for better Recall.</p></figcaption></figure>
{% endtab %}
{% endtabs %}
