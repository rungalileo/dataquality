# 🏁 Setup: Installing Galileo

### Python Environment Prerequisite

Galileo works with **python 3.7+ environments**.&#x20;

You can install the Galileo python client via a simple pip install.

### Installing Galileo

```
pip install --upgrade pip dataquality
```

{% hint style="info" %}
**\[Enterprise Only] Set your Galileo Console URL**

To point to your custom Enterprise environment, simply call

```python
import dataquality as dq

dq.set_console_url()
```
{% endhint %}

#### Login()&#x20;

Simply run the following lines in your notebook/python runtime.

```
import dataquality as dq

dq.login()
```

#### For automated login and configuration, see [Environment Variables](../python-library-api/environment-variables.md)

### Projects and Runs

In Galileo, your experimentation is organized into **projects** and **runs**.

* **Project**: A top level namespace holding one or more associated Runs.
* **Run**: A single instance of a model training/evaluation loop where data is logged

### Start a Project and a Run

Initialize a new run in Galileo by calling `dq.init()`.

```
dq.init(
    task_type="text_classification", 
    project_name="my_awesome_project", 
    run_name="my_awesome_run"
)
```

`project_name` and `run_name` are optional. If not provided, a generated name for each will be created and used.

[Get Started now with your data!](1-get-started-add-your-data-to-galileo.md)

Or see below to integrate your own model.

#### Task Types supported

Choose one of the following task types supported by Galileo:

1. [Text Classification](broken-reference) - Binary or multi-class models
2. [Multi Label Classification](broken-reference) - [Multi-label/Multi-output](https://en.wikipedia.org/wiki/Multi-label\_classification) models
3. [Named Entity Recognition ](broken-reference)
4. [Natural Language Inference (NLI)](broken-reference)

