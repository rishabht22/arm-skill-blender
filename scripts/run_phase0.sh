#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
weave_python="${WEAVE_PYTHON:-.venv-weave/bin/python}"
weave_results="${WEAVE_RESULTS:-results/phase0}"
weave_data="${WEAVE_DATA:-data/robofactory}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$weave_results"
shopt -s nullglob
weave_previous=("$weave_results"/*.json "$weave_results"/report.md)
if [[ -f "${weave_previous[0]}" ]]; then
    weave_archive=$(mktemp -d "$weave_results/previous.XXXXXX")
    for weave_file in "${weave_previous[@]}"; do
        if [[ -f "$weave_file" ]]; then mv "$weave_file" "$weave_archive/"; fi
    done
fi
scripts/download_data.sh --data-dir "$weave_data"
"$weave_python" -m weave.cli phase0-audit --mode inventory \
    --results-dir "$weave_results" --data-dir "$weave_data"

weave_status=0
"$weave_python" -m weave.cli phase0-audit --mode replay \
    --results-dir "$weave_results" --data-dir "$weave_data" || weave_status=1
read -r -a weave_tasks <<< "$("$weave_python" -c 'from weave.cli import load_config; print(*load_config()["benchmark"]["tasks"])')"
read -r -a weave_counts <<< "$("$weave_python" -c 'from weave.cli import load_config; print(*load_config()["benchmark"]["env_counts"])')"
for weave_task in "${weave_tasks[@]}"; do
    for weave_count in "${weave_counts[@]}"; do
        "$weave_python" -m weave.cli phase0-audit --mode benchmark \
            --task "$weave_task" --num-envs "$weave_count" --results-dir "$weave_results" || weave_status=1
    done
done

"$weave_python" -m weave.cli report --results-dir "$weave_results" --output "$weave_results/report.md"
exit "$weave_status"
