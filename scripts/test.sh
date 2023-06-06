#!/bin/sh -ex

. ./scripts/set-local-env.sh

./scripts/lint.sh

pytest --cov=dataquality --cov=tests --cov-report=term-missing --cov-report=xml --durations=10 -o console_output_style=progress --disable-warnings ${@}
