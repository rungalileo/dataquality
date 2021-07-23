#!/bin/sh -ex

# Sort imports one per line, so autoflake can remove unused imports
isort --force-single-line-imports dataquality tests scripts

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place dataquality tests scripts --exclude=__init__.py
black dataquality tests scripts
isort dataquality tests scripts
