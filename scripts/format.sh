#!/bin/sh -ex

# Sort imports one per line, so autoflake can remove unused imports
isort --force-single-line-imports dataquality tests docs_src

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place dataquality tests docs_src --exclude=__init__.py
black dataquality tests docs_src
isort dataquality tests docs_src
