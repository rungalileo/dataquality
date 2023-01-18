Glossary
========

**DEP Score**

A per sample holistic data quality score to identify samples in the
dataset contributing to low or high model performance i.e. ‘pulling’ the
model up or down respectively. In other words, the DEP score measures
the potential for “misfit” of an observation to the given model. `Read
More <galileo-product-features/galileo-data-error-potential-dep.md#dep-score-calculation>`__

**Hard for the model**

The subset of data samples with highest DEP scores, thereby “hard” for
the model to learn from during training or “hard” for the model to make
predictions on at test time. These samples can be hard due to one or
more of the following reasons: *boundary samples*, *noisy / corrupt
samples*, *mislabelled samples*, *misclassified samples*,
*out-of-distribution samples* etc. `Read
More <galileo-product-features/hard-easy-misclassified-subsets.md#hard-subset>`__

**Easy for the model**

The subset of data samples with lowest DEP scores, thereby “easy” for
the model to learn from during training, or “easy” for the model to make
predictions on at test time. Typically these “easy” samples are clean,
noise free data samples that the model had no issues training/predicting
on. `Read
More <galileo-product-features/hard-easy-misclassified-subsets.md#easy-subset>`__ 

**Gold**

The ground truth label of a sample as specified by the user. In other
words, the target label used by the model for a sample. 

**Pred**

The predicted label of a sample as specified by the model. In other
words, the label with highest prediction probability by the model. 

**Label**

Use interchangeably with ``Gold``, and is the ground truth of a sample
as specified by the user. In other words, the target used by the model
for a sample. 

**Task**

In a ``text_multi_label`` run, the task corresponds to the task your
model is making a prediction for. For example:

::

   input: I can't believe today is the day!
   outputs: task happiness: true, task nervousness: true, task anger: false 

In this case, your tasks are ``happiness``, ``nervousness``, and
``anger``

**Task Type**

The particular modeling exercise your model is working towards. Some
task types in Galileo:

-  ``text_classification``
-  ``text_multi_label``
-  ``text_ner`` 
