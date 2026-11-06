#!/usr/bin/env bash
# Run on Mac or Linux in a separate Python 3.10 environment.
# Usage: bash experiments/setup_calvin.sh sources|install|model|metadata|language [asset-root]
# Large datasets may be placed under /ICONAS/users/<username>/calvin.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
phase="${1:-}"
asset_root="${2:-$project_root/data/calvin}"
mkdir -p "$asset_root"
asset_root="$(cd "$asset_root" && pwd)"
export CALVIN_ROOT="$asset_root/vendor/calvin"
export HULC_ROOT="$asset_root/vendor/hulc"

download_zip() {
    local url="$1" archive="$2" destination="$3" overwrite="${4:--n}"
    mkdir -p "$destination"
    curl --fail --location --retry 3 --continue-at - "$url" --output "$archive"
    unzip "$overwrite" "$archive" -d "$destination"
}

case "$phase" in
sources)
    mkdir -p "$asset_root/vendor"
    # An rsync that excludes .git can leave unversioned vendor directories.
    # Preserve those copies, then fetch complete managed source checkouts.
    for source_root in "$CALVIN_ROOT" "$HULC_ROOT"; do
        if [ -d "$source_root" ] && [ ! -e "$source_root/.git" ]; then
            mv "$source_root" "$source_root.unversioned.$(date +%s)"
        fi
    done
    if [ ! -d "$CALVIN_ROOT" ]; then
        git clone --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_ROOT"
        git -C "$CALVIN_ROOT" checkout fa03f01f19c65920e18cf37398a9ce859274af76
        git -C "$CALVIN_ROOT" submodule update --init --recursive
    fi
    git -C "$CALVIN_ROOT" submodule update --init --recursive
    if [ ! -d "$HULC_ROOT" ]; then
        git clone https://github.com/mees/hulc.git "$HULC_ROOT"
        git -C "$HULC_ROOT" checkout 7fdb09f5f1ff831ec8df29cfc61a41db573e9a06
    fi
    ;;
install)
    python -c 'import sys; assert sys.platform in ("linux", "darwin") and sys.version_info[:2] == (3,10), "Use a separate Python 3.10 environment"'
    python -m pip install 'pip<24.1' 'setuptools==57.5.0' wheel
    python -c 'import pybullet' || { echo 'Install prebuilt pybullet into this conda environment first.' >&2; exit 1; }
    python -m pip install -r "$project_root/requirements.txt"
    # The RGB workflow does not need upstream tactile/TSNE/data-training packages.
    # Expose the pinned source trees without installing those unused dependencies.
    python - <<'PY'
import os, pathlib, sysconfig
roots = [os.environ['CALVIN_ROOT'] + '/calvin_env', os.environ['CALVIN_ROOT'] + '/calvin_models', os.environ['HULC_ROOT']]
pathlib.Path(sysconfig.get_paths()['purelib'], 'calvin_sources.pth').write_text('\n'.join(roots) + '\n')
PY
    python -m pip check
    ;;
model)
    if [ ! -f "$asset_root/models/HULC_D_D/saved_models/HULC_D_D.ckpt" ]; then
        download_zip http://hulc.cs.uni-freiburg.de/model_weights/HULC_D_D.zip "$asset_root/HULC_D_D.zip" "$asset_root/models"
    fi
    ;;
metadata)
    cd "$project_root"
    python -m src.envs.calvin_assets "$asset_root"
    ;;
language)
    test -d "$asset_root/datasets/task_D_D/validation"
    if [ ! -f "$asset_root/datasets/task_D_D/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy" ]; then
        download_zip http://hulc.cs.uni-freiburg.de/language_embeddings/D_D_lang_embs_val.zip "$asset_root/D_D_lang_embs_val.zip" "$asset_root/datasets/task_D_D" -o
    fi
    ;;
*)
    echo 'Usage: bash experiments/setup_calvin.sh sources|install|model|metadata|language [asset-root]' >&2
    exit 2
    ;;
esac
