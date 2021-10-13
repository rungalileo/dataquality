# dataquality

The Official Python Client for [Galileo](https://rungalileo.io).

## Getting Started

Install the package.
```sh
pip install dataquality
```

Import the package, login, and initialize a new project and run.
```python
import dataquality

dataquality.login()

dataquality.init()
```

Log your input datasets and log outputs as your model trains.

```python
dataquality.log(...)
```

Upload data to view in the Galileo Cloud Console.

```python
dataquality.finish()
```

## Contibuting

Read our [contributing doc](./CONTRIBUTING.md)!

