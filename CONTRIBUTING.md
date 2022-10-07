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


## Downloading Console Data

You know how sometimes you just want to try that new distribution plot over some real datasets? Well now you can ML experiment to your heart content using these handy functions to download the real data from the Galileo console (including probs and embeddings).

Refer to [dataquality.metrics](https://github.com/rungalileo/dataquality/blob/main/dataquality/metrics.py) for helpful utility functions like `dq.metrics.get_dataframe`!


## How does this work?
The dataquality client is a tool whose core purpose is to **get the users data into Galileo**

It tries to do this as quickly as possible in as lightweight a way as possible. Without any streaming solutions, however, this is the current state of the architecture.

First, note that all logged data is written to disk until the user calls `dq.finish` at which point the input data is joined with the output data, validated, and uploaded to Galileo's data store

The input and output data are logged separately, and stored differently.

### Input data
The input data is much more straightforward, because it's logged upfront and synchronously, before the model training begins.
The user has 3 ways of logging input data
1. `dq.log_dataset` - For logging data thats in a pandas/vaex/hf/iterable dataset
2. `dq.log_data_samples` - For logging a dictionary of lists/arrays
3. `dq.log_data_sample` - For logging a single row

The input data varies depending on what you're doing, but generally contains the following
* ids
* text
* labels (if not doing inference)
* token indices (if NER)
* input spans (if NER)

But there will **always be text and ids**

See `help(dq.func)` for more details on any one of those. `log_data_samples` will likely be the fastest,
although `log_dataset` will work with dataframes/iterables that do not fit in memory, so that may be preferable.
We do not suggest using `log_data_sample` unless you have to, as it will be significantly slower (the I/O on writing 1 row at a time isn't ideal)

The input data is stored as follows:
```
~/.galileo/logs/{project_id}/{run_id}
└── input_data
    ├── training
    │   └── data_0.arrow
    |   └── data_1.arrow
    |   └── data_2.arrow
    |   └── data_3.arrow
    ├── test
    │   └── data_0.arrow
    │   └── data_1.arrow
    └── validation
    │   └── data_0.arrow
    │   └── data_1.arrow
        └── data_2.arrow
```

Every time you call `log_data_samples` or `log_dataset` etc, a new file is created for the split you are logging for, and the number of logs you've made is recorded and incremented.
This separation makes uploading at the end straightforward because we know where to look.

When we upload the data, we upload one split at a time, and simply `glob` for all `arrow` files in the split directory for the input data.

### Output data
Output data is more complex. There is only one way to log data
* `dq.log_model_outputs`

Which alwyas has the same parameters
* id
* embeddings
* logits (or probabilities)

No matter the task, these are required. We can only access this data during the model training process,
getting the embeddings within the layers of the model, and the logits from the final layer. It is paramount
that we do not slow down the training process, which makes this part tricky. It's relatively computational to
validate the logits and embeddings data, and write it to disk, all without blocking the model, so we use threading.

Within the threaded process, we validate the data, and write it to disk in a small hdf5 file named something random (uuid).
We end up with a huge number of small hdf5 files. We are typically logging one per batch (depending on where users inject the dataquality logging code),
so the number of files really adds up.

The end structure after training will look something like this
```
~/.galileo/logs/4a8a50d8-9a50-48c0-9c5b-d3f015b775d3/8fdb936b-8719-495c-a35b-143e1c2d17a5
├── input_data
│   ├── ...
├── test
│   ├── 0
│   │   ├── 0061c72d52ee.hdf5
│   │   ├── 00c25e6fa062.hdf5
│   │   ├── 00d203017c24.hdf5
│   │   ...
│   ├── 1
│   │   ├── 00e4571f73e8.hdf5
│   │   ├── 010e170e8d54.hdf5
│   ├── ...
--
├── training
│   ├── 0
│   │   ├── 001124581f52.hdf5
│   │   ├── 003159f57997.hdf5
│   │   ...
│   ├── 1
│   │   ├── 006eb8f391d7.hdf5
│   │   ├── 008006c44a7a.hdf5
│   │   ...
--
└── validation
    ├── 0
    │   ├── 001aa5b0d70f.hdf5
    │   ├── 0034c0f6f98f.hdf5
    ├── ...
    ├── 1
    │   ├── 0046edaf69cf.hdf5
    │   ├── 007deba76648.hdf5
    │   ...
```

After training, the user will call finish, which does a few things
* Waits for all of the threads to finish (in case there were more running/writing)
* Combines all of the small hdf5 files per split into a single hdf5 file
* Joins the output data with the input data for the split
* Uploads the split to Galileo data store

### Why do we need IDs???
I know, it's not ideal. During model training particularly, its uncommon to have IDs in forward passes,
and it makes out integrations a bit harder.

Currently, we need the IDs in order to join the input data logged before model training with the output data
during model training. Fundamentally in the model training process there's nothing "naturally" there
that is also there during the input training process. If *you* have an idea to eliminate the need for
these IDs, please open a ticket, message us on slack, or open a PR! We would love to move past the IDs
