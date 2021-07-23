#!/bin/sh -ex

mkdocs build

cp ./docs/index.md ./README.md
