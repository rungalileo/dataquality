#!/bin/sh -e

mkdocs build

cp ./docs/index.md ./README.md
