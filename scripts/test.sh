#!/bin/sh -ex

. ./scripts/set-local-env.sh

# ./scripts/lint.sh

pytest tests/inference/test_text_classification.py # --cov=dataquality --cov=tests --cov=docs_src --cov-report=term-missing --cov-report=xml -o console_output_style=progress --disable-warnings ${@}
