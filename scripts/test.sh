#!/bin/sh -ex

. ./scripts/set-local-env.sh

./scripts/lint.sh

pytest -s tests/integrations/ultralytics