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

## Troubleshooting
In specific Python environments like Google Colaboratory the kernel needs to be restarted before dataquality works.

```python
#@title Install `dataquality`
try:
    import dataquality as dq
except ImportError:
    # Install the dependencies
    !pip install -U pip &> /dev/null

    # Install HF datasets for downloading the example datasets
    !pip install -U dataquality torch datasets transformers &> /dev/null
    
    print('👋 Installed necessary libraries and restarting runtime! This should only need to happen once.')
    print('🙏 Continue with the rest of the notebook or hit "Run All" again!')

    # Restart the runtime
    import os, time
    time.sleep(1) # gives the print statements time to flush
    os._exit(0) # exits without allowing the next cell to run
```

## Contibuting

Read our [contributing doc](./CONTRIBUTING.md)!

