<!-- This data file has been placed in the public domain. -->
<!-- Derived from the Unicode character mappings available from
<http://www.w3.org/2003/entities/xml/>.
Processed by unicode2rstsubs.py, part of Docutils:
<https://docutils.sourceforge.io>. -->
# Introduction

## What is Galileo’s dataquality?

**dataquality** is a Python library that enables data scientists to build high-performing models quickly, with better quality training data. With a few lines of code added during training or during an inference run, dataquality instantly finds data errors.

<iframe width="750" height="420" src="https://youtube.com/embed/vo6MeVaTacU"
  frameborder="0"
  allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen>
</iframe>Galileo inspects data, discovers errors and helps you improve labels. It offers a per-sample holistic data quality score, called the DEP Score, to identify samples in the dataset that are contributing to low or high model performance. The DEP score measures the potential for “misfit” of an observation to the given model. The samples with the highest DEP scores are considered “hard for the model” to learn from during training or “hard” for the model to make predictions on at test time. On the other hand, samples with the lowest DEP scores are considered “easy for the model” to learn from during training or “easy” for the model to make predictions on at test time.

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


* [API](api.md) → *API Docs*

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


* [API](api.md)


    * [`AggregateFunction`](api.md#dataquality.AggregateFunction)


    * [`Condition`](api.md#dataquality.Condition)


    * [`ConditionFilter`](api.md#dataquality.ConditionFilter)


    * [`Operator`](api.md#dataquality.Operator)


    * [`auto()`](api.md#dataquality.auto)


    * [`build_run_report()`](api.md#dataquality.build_run_report)


    * [`configure()`](api.md#dataquality.configure)


    * [`docs()`](api.md#dataquality.docs)


    * [`finish()`](api.md#dataquality.finish)


    * [`get_run_status()`](api.md#dataquality.get_run_status)


    * [`init()`](api.md#dataquality.init)


    * [`log_data_sample()`](api.md#dataquality.log_data_sample)


    * [`log_data_samples()`](api.md#dataquality.log_data_samples)


    * [`log_dataset()`](api.md#dataquality.log_dataset)


    * [`log_model_outputs()`](api.md#dataquality.log_model_outputs)


    * [`login()`](api.md#dataquality.login)


    * [`register_run_report()`](api.md#dataquality.register_run_report)


    * [`set_console_url()`](api.md#dataquality.set_console_url)


    * [`set_epoch()`](api.md#dataquality.set_epoch)


    * [`set_epoch_and_split()`](api.md#dataquality.set_epoch_and_split)


    * [`set_labels_for_run()`](api.md#dataquality.set_labels_for_run)


    * [`set_split()`](api.md#dataquality.set_split)


    * [`set_tagging_schema()`](api.md#dataquality.set_tagging_schema)


    * [`set_tasks_for_run()`](api.md#dataquality.set_tasks_for_run)


    * [`wait_for_run()`](api.md#dataquality.wait_for_run)


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
