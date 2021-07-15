#!/bin/sh -ex

# Sort imports one per line, so autoflake can remove unused imports
isort --force-single-line-imports galileo_python tests scripts

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place galileo_python tests scripts --exclude=__init__.py
black galileo_python tests scripts
isort galileo_python tests scripts
