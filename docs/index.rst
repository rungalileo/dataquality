.. include:: <isonum.txt>

.. _introduction:

Introduction
============

What is Galileo's dataquality?
-----------------
**dataquality** is a Python library that enables data scientists to build high-performing models quickly, with better quality training data. With a few lines of code added during training or during an inference run, dataquality instantly finds data errors.

.. raw:: html

   <iframe width="750" height="420" src="https://youtube.com/embed/vo6MeVaTacU"
     frameborder="0"
     allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
     allowfullscreen>
   </iframe>

Galileo inspects data, discovers errors and helps you improve labels. It offers a per-sample holistic data quality score, called the DEP Score, to identify samples in the dataset that are contributing to low or high model performance. The DEP score measures the potential for "misfit" of an observation to the given model. The samples with the highest DEP scores are considered "hard for the model" to learn from during training or "hard" for the model to make predictions on at test time. On the other hand, samples with the lowest DEP scores are considered "easy for the model" to learn from during training or "easy" for the model to make predictions on at test time.

What dataquality does:

* Compute a data quality score
* Find likely mislabeled samples
* Provide insights into the dataset


Overview of this documentation
------------------------------

This documentation space consists of the following main sections:

* **Introduction** |rarr| *You are here now*
* :doc:`/install/index` |rarr| *How to install dataquality servers,
  nodes and clients*
.. * :doc:`/use/index` |rarr| *How to use dataquality servers,
..   nodes and clients*
.. * :doc:`/technical-documentation/index` (Under construction) |rarr|
..   *Implementation details of the dataquality platform*
.. * :doc:`/devops/index` |rarr| *How to collaborate on the development of the
..   dataquality infrastructure*
.. * :doc:`/algorithms/index` |rarr| *Develop algorithms that are compatible with
..   dataquality*
* :doc:`/glossary` |rarr| *A dictionary of common terms used in these docs*
* :doc:`/api` |rarr| *API Docs*


dataquality resources
------------------

This is a - non-exhaustive - list of dataquality resources.

**Documentation**

* `docs.dataquality.ai <https://docs.dataquality.ai>`_ |rarr| *This documentation.*
* `dataquality.ai <https://dataquality.ai>`_ |rarr| *dataquality project website*
* `Academic papers <https://dataquality.ai/dataquality/>`_ |rarr|
  *Technical insights into dataquality*

**Source code**

* `dataquality <https://github.com/dataquality/dataquality>`_ |rarr| *Contains all*
  *components (and the python-client).*
* `Planning <https://github.com/orgs/dataquality/projects>`_ |rarr| *Contains all
  features, bugfixes and feature requests we are working on. To submit one
  yourself, you can create a*
  `new issue <https://github.com/dataquality/dataquality/issues>`_.

**Community**

* `Discord <https://discord.gg/yAyFf6Y>`_  |rarr| *Chat with the dataquality
  community*



-------------------------------------------------------------------------------

Index
=====

.. toctree::
   :maxdepth: 4

   self

.. toctree::
   :numbered: 3
   :maxdepth: 4

   install/index
   api
   use/index
   glossary

.. toctree::
   :maxdepth: 2
   release_notes
   partners
