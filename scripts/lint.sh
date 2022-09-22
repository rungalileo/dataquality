#!/bin/sh -ex

mypy dataquality
flake8 dataquality tests
black dataquality tests --check
isort dataquality tests --check-only
