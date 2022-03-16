#!/bin/sh -ex

mypy dataquality
black dataquality tests docs_src --check
flake8 dataquality tests docs_src
isort dataquality tests docs_src --check-only
