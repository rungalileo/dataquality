
[build-system]
requires = [
    "setuptools>=65.0",
    "wheel>=0.37",
]
build-backend = "setuptools.build_meta"

[project]
name = "dataquality"
dynamic = ["version"]
readme = "README.md"
license = {text = 'See LICENSE'}
requires-python = ">=3.8"
dependencies = [
    "dataquality",
]
[[project.authors]]
name = "Galileo Technologies, Inc."
email = "devs+dq@rungalileo.io"

[project.scripts]
dqyolo = "dataquality.dqyolo:main"

[project.urls]
Homepage = "https://github.com/rungalileo/dataquality"
Documentation = "https://rungalileo.gitbook.io/galileo"

[project.optional-dependencies]
# Install this by adding the --extra-index-url option as shown below
# `pip install 'dataquality[cuda]' --extra-index-url=https://pypi.nvidia.com/ `
cuda = [
   "ucx-py-cu11<=0.30",
   "rmm-cu11==23.2.0",
   "raft-dask-cu11==23.2.0",
   "pylibraft-cu11==23.2.0",
   "dask-cudf-cu11==23.2.0",
   "cudf-cu11==23.2.0",
   "cuml-cu11==23.2.0"
]
minio = ["minio>=7.1.0,<7.2.0"]
setfit = ["setfit==0.7.0"]

[tool.setuptools.dynamic]
version = {attr = "dataquality.__version__"}


[tool.coverage.run]
parallel = true
source = [
    "dataquality/",
    "tests/"
]
omit = [
    "*__init__.py",
    "*metrics.py",
]
