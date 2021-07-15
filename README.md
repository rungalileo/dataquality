# galileo_python

The Python client for [Galileo](https://rungalileo.io).

## Usage

Replace `pyproject.toml` with relevant values for your project.

Use [pyenv](https://github.com/pyenv/pyenv) for managing your python environment.

Install [flit](https://flit.readthedocs.io/en/latest/).

```sh
$ which python
$HOME/.pyenv/shims/python
$ python -m pip install flit
```

Set up your virtual environment.

```sh
$ python -m venv .venv
$ source .venv/bin/activate
```

Install dependencies and run tests.

```sh
# the following command installs to your virtualenv
$ flit install --python $(which python)
# the following symlink the module/package into site packages in .venv instead of copying it
$ ./scripts/install.sh
# use flit install --python $(which python) locally and ./scripts/install.sh in ci-things
$ pyenv rehash
$ ./scripts/test.sh
$ ./scripts/test-cov-html.sh
```
