"""A script to run all python integration tests in the tests/integrations folder
Meant to be run from 1 level behind this directory (python scripts/run_integrations.py)
"""

import os
from glob import glob


if __name__ == "__main__":
    print(os.popen("pwd").read())
    for file in glob("tests/integration/*.py"):
        print(f"Running test {file}")
        os.system(f"python {file}")
