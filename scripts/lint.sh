#!/bin/sh -ex

mypy galileo_python
flake8 galileo_python tests
black galileo_python tests --check
isort galileo_python tests scripts --check-only
