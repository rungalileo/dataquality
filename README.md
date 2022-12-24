# dataquality

The Official Python Client for [Galileo](https://rungalileo.io).

Galileo is a tool for understanding and improving the quality of your NLP (and soon CV!) data.

Galileo gives you access to all of the information you need, at a UI and API level, to continuously build better and more robust datasets and models.

`dataquality` is your entrypoint to Galileo. It helps you start and complete the loop of data quality improvements.

## Getting Started

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

### What kinds of datasets can I analyze?
Currently, you can analyze **Text Classification** and **NER**

If you want support for other kinds, [reach out!](https://github.com/rungalileo/dataquality/issues/new?assignees=ben-epstein&labels=enhancement&template=feature.md&title=%5BFEATURE%5D)

### Can I use auto with other data forms?
`auto` params `train_data`, `val_data`, and `test_data` can also take as input pandas dataframes and huggingface dataframes!

### What if all my data is in huggingface?
Use the `hf_data` param to point to a dataset in huggingface
```python
import dataquality as dq

dq.auto("rungalileo/emotion")
```

### Anything else? Can I learn more?
Run `help(dq.auto)` for more information on usage<br>
Check out our [docs](https://rungalileo.gitbook.io/galileo/getting-started/add-your-data-to-galileo/dq-auto) for the inspiration behind this methodology.


## Can I analyze data using a custom model?
Yes! Check out our [full documentation](https://rungalileo.gitbook.io/galileo/getting-started/byom-bring-your-own-model) and [example notebooks](https://rungalileo.gitbook.io/galileo/example-notebooks) on how to integrate your own model with Galileo

## Contibuting

Read our [contributing doc](./CONTRIBUTING.md)!

