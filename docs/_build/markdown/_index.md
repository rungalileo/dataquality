<!-- This data file has been placed in the public domain. -->
<!-- Derived from the Unicode character mappings available from
<http://www.w3.org/2003/entities/xml/>.
Processed by unicode2rstsubs.py, part of Docutils:
<https://docutils.sourceforge.io>. -->
# Introduction

## What is Galileo's dataquality?

**dataquality** is a Python library that enables data scientists to build high-performing models quickly, with better quality training data. With a few lines of code added during training or during an inference run, dataquality instantly finds data errors.

<iframe width="750" height="420" src="https://youtube.com/embed/vo6MeVaTacU"
  frameborder="0"
  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen>
</iframe>Galileo inspects data, discovers errors and helps you improve labels. It offers a per-sample holistic data quality score, called the DEP Score, to identify samples in the dataset that are contributing to low or high model performance. The DEP score measures the potential for "misfit" of an observation to the given model. The samples with the highest DEP scores are considered "hard for the model" to learn from during training or "hard" for the model to make predictions on at test time. On the other hand, samples with the lowest DEP scores are considered "easy for the model" to learn from during training or "easy" for the model to make predictions on at test time.

What dataquality does:


* Compute a data quality score


* Find likely mislabeled samples


* Provide insights into the dataset

## Overview of this documentation

This documentation space consists of the following main sections:


* **Introduction** → *You are here now*


* [How to install](install/index.md) → *How to install dataquality servers,
nodes and clients*

<!-- * :doc:`/use/index` |rarr| *How to use dataquality servers, -->
<!-- nodes and clients* -->
<!-- * :doc:`/technical-documentation/index` (Under construction) |rarr| -->
<!-- *Implementation details of the dataquality platform* -->
<!-- * :doc:`/devops/index` |rarr| *How to collaborate on the development of the -->
<!-- dataquality infrastructure* -->
<!-- * :doc:`/algorithms/index` |rarr| *Develop algorithms that are compatible with -->
<!-- dataquality* -->

* [Glossary](glossary.md) → *A dictionary of common terms used in these docs*


* [Reference](api.md) → *API Docs*

## dataquality resources

This is a - non-exhaustive - list of dataquality resources.

**Documentation**


* [docs.dataquality.ai](https://docs.dataquality.ai) → *This documentation.*


* [dataquality.ai](https://dataquality.ai) → *dataquality project website*


* [Academic papers](https://dataquality.ai/dataquality/) →
*Technical insights into dataquality*

**Source code**


* [dataquality](https://github.com/dataquality/dataquality) → *Contains all*
*components (and the python-client).*


* [Planning](https://github.com/orgs/dataquality/projects) → *Contains all
features, bugfixes and feature requests we are working on. To submit one
yourself, you can create a*
[new issue](https://github.com/dataquality/dataquality/issues).

**Community**


* [Discord](https://discord.gg/yAyFf6Y)  → *Chat with the dataquality
community*


---

# Index


* [Introduction]()



* [How to install](install/index.md)


* [Reference](api.md)


    * [watch (PyTorch)](api.md#watch-pytorch)


        * [`watch()`](api.md#dataquality.integrations.torch.watch)


        * [`watch()`](api.md#dataquality.integrations.transformers_trainer.watch)


        * [`watch()`](api.md#dataquality.integrations.spacy.watch)


        * [`DataQualityCallback`](api.md#dataquality.integrations.keras.DataQualityCallback)


            * [`DataQualityCallback.on_epoch_begin()`](api.md#dataquality.integrations.keras.DataQualityCallback.on_epoch_begin)


            * [`DataQualityCallback.on_test_batch_begin()`](api.md#dataquality.integrations.keras.DataQualityCallback.on_test_batch_begin)


            * [`DataQualityCallback.on_test_batch_end()`](api.md#dataquality.integrations.keras.DataQualityCallback.on_test_batch_end)


            * [`DataQualityCallback.on_train_batch_begin()`](api.md#dataquality.integrations.keras.DataQualityCallback.on_train_batch_begin)


            * [`DataQualityCallback.on_train_batch_end()`](api.md#dataquality.integrations.keras.DataQualityCallback.on_train_batch_end)


        * [`DataQualityLoggingLayer`](api.md#dataquality.integrations.keras.DataQualityLoggingLayer)


            * [`DataQualityLoggingLayer.call()`](api.md#dataquality.integrations.keras.DataQualityLoggingLayer.call)


        * [`add_ids_to_numpy_arr()`](api.md#dataquality.integrations.keras.add_ids_to_numpy_arr)


        * [`add_sample_ids()`](api.md#dataquality.integrations.keras.add_sample_ids)


        * [`auto()`](api.md#dataquality.auto)


        * [`finish()`](api.md#dataquality.finish)


        * [`init()`](api.md#dataquality.init)


        * [`log_data_sample()`](api.md#dataquality.log_data_sample)


        * [`log_dataset()`](api.md#dataquality.log_dataset)


        * [`log_model_outputs()`](api.md#dataquality.log_model_outputs)


        * [`login()`](api.md#dataquality.login)


        * [`set_epoch()`](api.md#dataquality.set_epoch)


        * [`set_labels_for_run()`](api.md#dataquality.set_labels_for_run)


        * [`set_split()`](api.md#dataquality.set_split)


* [Indices and tables](api.md#indices-and-tables)


* [How to use](use/index.md)


    * [Client](use/client.md)


        * [Introduction](use/client.md#introduction)


        * [User Interface](use/client.md#user-interface)


        * [Python client](use/client.md#python-client)


            * [Authentication X](use/client.md#authentication-x)


            * [Creating an organization](use/client.md#creating-an-organization)


            * [Creating a collaboration](use/client.md#creating-a-collaboration)


            * [Registering a node](use/client.md#registering-a-node)


            * [Creating a task](use/client.md#creating-a-task)


        * [R Client](use/client.md#r-client)


            * [Example](use/client.md#example)


        * [Server API](use/client.md#server-api)


    * [Node](use/node.md)


        * [Introduction](use/node.md#introduction)


            * [Quick start](use/node.md#quick-start)


            * [Available commands](use/node.md#available-commands)


        * [Configure](use/node.md#configure)


            * [Configuration file structure](use/node.md#configuration-file-structure)


            * [Configure using the Wizard](use/node.md#configure-using-the-wizard)


            * [Update configuration](use/node.md#update-configuration)


            * [Local test setup](use/node.md#local-test-setup)


        * [Security](use/node.md#security)


        * [Logging](use/node.md#logging)


    * [Server](use/server.md)


        * [Introduction](use/server.md#introduction)


            * [Quick start](use/server.md#quick-start)


            * [Available commands](use/server.md#available-commands)


        * [Configure](use/server.md#configure)


            * [Configuration file structure](use/server.md#configuration-file-structure)


            * [Configuration wizard](use/server.md#configuration-wizard)


            * [Update configuration](use/server.md#update-configuration)


            * [Local test setup](use/server.md#local-test-setup)


            * [Batch import](use/server.md#batch-import)


        * [Logging](use/server.md#logging)


        * [Shell](use/server.md#shell)


            * [Organizations](use/server.md#organizations)


            * [Roles and Rules](use/server.md#roles-and-rules)


            * [Users](use/server.md#users)


            * [Collaborations](use/server.md#collaborations)


            * [Nodes](use/server.md#nodes)


            * [Tasks and Results](use/server.md#tasks-and-results)


* [Glossary](glossary.md)
