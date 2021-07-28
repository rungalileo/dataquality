#!/bin/sh -ex

mypy dataquality
flake8 dataquality tests docs_src
black dataquality tests docs_src --check
isort dataquality tests docs_src --check-only
