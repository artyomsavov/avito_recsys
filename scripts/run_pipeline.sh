#!/bin/bash
# Полный пайплайн одного прогона: bash scripts/run_pipeline.sh synth|leaky [FROM_STEP]
# FROM_STEP — продолжить с указанного шага (например, train_als)
set -euo pipefail
RUN=${1:?usage: run_pipeline.sh synth|leaky [FROM_STEP]}
FROM=${2:-}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1

steps=(aggregate build_matrix train_als build_covisitation build_item_pool
       generate_candidates_als generate_candidates_covisit merge_candidates build_features
       train_ranker evaluate)

for step in "${steps[@]}"; do
    if [ -n "$FROM" ]; then
        [ "$step" = "$FROM" ] && FROM="" || continue
    fi
    echo "===== [$RUN] $step $(date '+%H:%M:%S')"
    uv run src/$step.py --run "$RUN"
done
echo "===== [$RUN] done $(date '+%H:%M:%S')"
