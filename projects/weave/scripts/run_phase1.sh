#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

weave_python="${WEAVE_PYTHON:-.venv-weave/bin/python}"
if [[ ! -x "$weave_python" ]]; then
    echo "Run scripts/remote_gpu_setup.sh first. See docs/REMOTE_RUN.md." >&2
    exit 1
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to one available GPU before starting Phase 1." >&2
    exit 1
fi
export PYTHONPATH="$PWD/src:$PWD/../robofactory_lift/src${PYTHONPATH:+:$PYTHONPATH}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-$HOME/.maniskill/data}"
if [[ ! -f "$MS_ASSET_DIR/assets/mani_skill2_ycb/info_pick_v0.json" ]]; then
    "$weave_python" -m mani_skill.utils.download_asset ycb -y
fi
exec "$weave_python" -m weave.cli phase1-smoke \
    --results-dir "${WEAVE_PHASE1_RESULTS:-results/phase1}"
