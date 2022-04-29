# Contributing

Assuming you have cloned this repository to your local machine, you can follow these guidelines to make contributions.

## Use a virtual environment

```sh
$ python -m venv .venv
```

This will create a directory `.venv` with python binaries and then you will be able to install packages for that isolated environment.

Next, activate the environment.

```sh
$ source .venv/bin/activate
```

To check that it worked correctly;

```sh
$ which python pip
/path/to/dataquality/.venv/bin/python
/path/to/dataquality/.venv/bin/pip
```

[`pyenv`](https://github.com/pyenv/pyenv) is suggested for local python development.

## Flit

This project uses `flit` to manage our project's dependencies.

After activating the environment as described above, install flit:

```sh
$ pip install flit
```

Install dependencies

```sh
./scripts/install.sh
```

## Development
Developing with Flit is simple. Symlink this repo to your venv so you can make changes
and test them without reinstalling the package.

Run the following from the root of dataquality:
```sh
flit install -s
```

You can specify which python environment to install into using the `--python` flag
(useful for developing/testing from external venvs)

### Debugging
If you're looking to debug some code in dataquality, for example with `pdb` in jupyter,
you can do that easily:
1. Install and symlink `dataquality` as shown above
2. Use **the same** python env to start your jupyter session
   1. Now, your jupyter session will be symlinked to dataquality
   2. **Note:** You still need to restart the kernel after code changes
3. Set your `pdb` trace in your code, restart your kernel, and run. You'll see the `ipdb` session

## Formatting

```sh
./scripts/format.sh
```

## Tests

You will need to have a local cluster running, read our [API documentation](https://github.com/rungalileo/api/blob/main/CONTRIBUTING.md) to get set up.

Set some local env variables

```
. ./scripts/set-local-env.sh
```

Now run your tests!

```sh
./scripts/test-cov-html.sh
```


## Test Notebooks

Run this from this project's root directory to boot up tests.

```
. ./scripts/set-local-env.sh; jupyter notebook docs/
```


## Deployment

Everything is done through github actions. Make sure to bump the version of the package

```
./scripts/bump-version.sh
```

commit the change and publish a new version.
