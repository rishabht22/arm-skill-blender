#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${WEAVE_PYTHON:-.venv-weave/bin/python}" -m weave.cli download-data "$@"
