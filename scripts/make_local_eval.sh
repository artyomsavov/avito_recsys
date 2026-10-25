#!/bin/bash
# Воспроизводит таргет synth-валидации (data/raw/local_eval.csv, local_eval_users.csv)
# скриптом организаторов и сверяет с зафиксированными контрольными суммами.
# Существующие файлы не перезаписываются — только проверяются; отсутствующие создаются.
#
# Как был получен оригинал:
#   * только партиция part_000 (все 7 131 synth-юзера имеют user_id % 100 == 0);
#   * TZ=Europe/Moscow: prepare_local_eval.py читает порог "2026-04-08T00:00:00" как naive
#     datetime -> фактический порог 2026-04-07T21:00Z (SYNTH_CUTOFF_MS в src/config.py).
# Порядок строк polars group_by не детерминирован, поэтому сверяем отсортированное содержимое.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RAW=data/raw
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

declare -A EXPECTED=(
    [local_eval.csv]=771f67af395b57a4fb6252fa5a84c056416d14c09f83ea1057f6b4fa25bbae32
    [local_eval_users.csv]=9c3f137df2284300b4f712b20a837e50e98358fc2a74ae159af0e7d27c3d85ad
)

check() {
    local got
    got=$(tail -n +2 "$1" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
    if [ "$got" = "$2" ]; then echo "OK       $1"; else echo "MISMATCH $1 ($got)"; exit 1; fi
}

TZ=Europe/Moscow uv run external/prepare_local_eval.py \
    --train "$RAW/train_data/part_000.parquet" \
    --item-features "$RAW/item_features.parquet" \
    --contact-eids "$RAW/contact_eids.csv" \
    --out "$OUT/local_eval.csv"

for f in "${!EXPECTED[@]}"; do
    check "$OUT/$f" "${EXPECTED[$f]}"
    if [ -f "$RAW/$f" ]; then
        check "$RAW/$f" "${EXPECTED[$f]}"
    else
        cp "$OUT/$f" "$RAW/$f" && echo "created  $RAW/$f"
    fi
done
