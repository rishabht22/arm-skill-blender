#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]] || ! command -v nvidia-smi >/dev/null; then
    echo "The pinned SAPIEN audit needs a Linux x86-64 host with NVIDIA drivers. See docs/REMOTE_RUN.md." >&2
    exit 1
fi
nvidia-smi -L
if ! command -v git >/dev/null || ! command -v "${WEAVE_SETUP_PYTHON:-python3.11}" >/dev/null; then
    echo "Install Git and Python 3.11 with venv support, or set WEAVE_SETUP_PYTHON." >&2
    exit 1
fi
"${WEAVE_SETUP_PYTHON:-python3.11}" -m venv .venv-weave
weave_python="$PWD/.venv-weave/bin/python"
"$weave_python" -m pip install 'hydra-core==1.3.2'
export PYTHONPATH="$PWD/src:$PWD/../robofactory_lift/src${PYTHONPATH:+:$PYTHONPATH}"
read -r weave_rf_rev weave_ms_version weave_sapien_version < <(
    "$weave_python" -c 'from weave.cli import load_config; c=load_config(); print(
        c["robofactory_revision"], c["maniskill_version"], c["sapien_version"])'
)

weave_upstream="$PWD/.venv-weave/robofactory"
if [[ ! -d "$weave_upstream/.git" ]]; then
    git init "$weave_upstream"
    git -C "$weave_upstream" remote add origin https://github.com/MARS-EAI/RoboFactory.git
fi
if [[ "$(git -C "$weave_upstream" rev-parse --verify HEAD 2>/dev/null || true)" != "$weave_rf_rev" ]]; then
    git -C "$weave_upstream" fetch --depth=1 origin "$weave_rf_rev"
    git -C "$weave_upstream" checkout --detach FETCH_HEAD
fi
git -C "$weave_upstream" diff --quiet HEAD
"$weave_python" -m pip install     "mani_skill==$weave_ms_version" "sapien==$weave_sapien_version"     'torch==2.6.0' 'h5py>=3.11,<4' lxml 'six==1.17.0'
"$weave_python" -m pip install --no-deps -e "$weave_upstream"
"$weave_python" -m pip check

"$weave_python" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
import robofactory
import torch
from weave.cli import load_config

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see an NVIDIA CUDA device.")
asset_dir = Path(robofactory.ASSET_DIR)
revision = load_config()["asset_revision"]
snapshot_download(
    repo_id="sparklexfantasy/RoboFactory_asset", repo_type="dataset", revision=revision,
    local_dir=asset_dir,
    allow_patterns=[
        "scenes/table/table.glb", "objects/meat_annotated/*",
        "objects/single_shoe_annotated/*", "objects/pot_annotated/*",
    ],
)
(asset_dir / ".weave-asset-revision").write_text(revision + "\n")
PY
echo "WEAVE audit runtime ready. Run scripts/run_phase0.sh."
