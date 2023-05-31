# dataquality

The Official Python Client for [Galileo](https://rungalileo.io).

Galileo is a tool for understanding and improving the quality of your NLP and CV data.

Galileo gives you access to all of the information you need, at a UI and API level, to continuously build better and more robust datasets and models.

`dataquality` is your entrypoint to Galileo. It helps you start and complete the loop of data quality improvements.

# ToC
* [Getting Started](#getting-started)
* [Custom Integrations](#can-i-analyze-data-using-a-custom-model)
* [No labels? No problem](#what-if-i-dont-have-labels-to-train-with-can-you-help-with-labeling)
* [Programmatic Access](#is-there-a-python-api-for-programmatically-interacting-with-the-console)
* [Contributing](#contributing)


<details>
<summary><h2>Getting Started</h2></summary>

Install the package.
```sh
pip install dataquality
```

Create an account at [Galileo](https://console.cloud.rungalileo.io/sign-up)

Grab your [token](https://console.cloud.rungalileo.io/get-token)

Get your dataset and analyze it with `dq.auto`
(You will be prompted for your token here)
```python
import dataquality as dq

dq.auto(
    train_data="/path/to/train.csv",
    val_data="/path/to/val.csv",
    test_data="/path/to/test.csv",
    project_name="my_first_project",
    run_name="my_first_run",
)
```

☕️ Wait for Galileo to train your model and analyze the results.  
✨ A link to your run will be provided automatically

#### Pro tip: Set your token programmatically for automated workflows
By setting the token, you'll never be prompted to log in
```python
import dataquality as dq

dq.config.token = 'MY-TOKEN'
```
For long-lived flows like CI/CD, see our docs on [environment variables](https://rungalileo.gitbook.io/galileo/python-library-api/environment-variables)

<details>
<summary><h3>What kinds of datasets can I analyze?</h3></summary>

Currently, you can analyze **Text Classification** and **NER**

If you want support for other kinds, [reach out!](https://github.com/rungalileo/dataquality/issues/new?assignees=ben-epstein&labels=enhancement&template=feature.md&title=%5BFEATURE%5D)
</details>

<details>
<summary><h3>Can I use auto with other data forms?</h3></summary>

`auto` params `train_data`, `val_data`, and `test_data` can also take as input pandas dataframes and huggingface dataframes!
</details>

<details>
<summary><h3>What if all my data is in huggingface?</h3></summary>

Use the `hf_data` param to point to a dataset in huggingface
```python
import dataquality as dq

dq.auto(hf_data="rungalileo/emotion")
```
</details>

<details>
<summary><h3>Anything else? Can I learn more?</h3></summary>

Run `help(dq.auto)` for more information on usage<br>
Check out our [docs](https://rungalileo.gitbook.io/galileo/getting-started/add-your-data-to-galileo/dq-auto) for the inspiration behind this methodology.
</details>
</details>


<details>
<summary><h2>Can I analyze data using a custom model?</h2></summary>

Yes! Check out our [full documentation](https://rungalileo.gitbook.io/galileo/getting-started/byom-bring-your-own-model) and [example notebooks](https://rungalileo.gitbook.io/galileo/example-notebooks) on how to integrate your own model with Galileo
</details>


<details>
<summary><h2>What if I don't have labels to train with? Can you help with labeling?</h2></summary>

We have an [app for that](https://github.com/rungalileo/bulk-labeling/)! Currently text classification only, but [reach out](https://github.com/rungalileo/bulk-labeling/issues/new?assignee=ben-epstein) if you want a new modality!<br>

This is currently in development, and not an official part of the Galileo product, but rather an open source tool for the community.

We've built a bulk-labeling tool (and hosted it on streamlit) to help you generate labels quickly using semantic embeddings and text search.

For more info on how it works and how to use it, check out the [open source repo](https://github.com/rungalileo/bulk-labeling/).
</details>


<details>
<summary><h2>Is there a Python API for programmatically interacting with the console?</h2></summary>

Yes! See our docs on [`dq.metrics`](https://rungalileo.gitbook.io/galileo/python-library-api/dq.metrics) to access things like overall metrics, your analyzed dataframe, and even your embeddings.
</details>

<details>
<summary><h2>Contributing</h2></summary>

Read our [contributing doc](./CONTRIBUTING.md)!

