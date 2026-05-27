#!/bin/bash
# Скрипт для локальной разработки и отладки (train 20/100 part)
# Запуск: bash src/download_research.sh

# Определяем корень проекта
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR" || exit

BASE="https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4"

echo "[Research] Скачивание базовых файлов..."
curl -O -C - $BASE/item_features.parquet
curl -O -C - $BASE/contact_eids.csv
curl -O -C - $BASE/eval_users.csv
curl -O -C - $BASE/prepare_local_eval.py
curl -O -C - $BASE/popular.py
curl -O -C - $BASE/submission_popular.csv
curl -O -C - $BASE/eval_user_events.zip

echo "[Research] Скачивание 1/5 обучающей выборки..."
curl -O -C - $BASE/train_000-019.zip

echo "[Research] Распаковка..."
unzip -o eval_user_events.zip
unzip -o train_000-019.zip

echo "[Research] Очистка архивов..."
rm -f eval_user_events.zip train_000-019.zip

echo "[Research] Готово! Легкий срез данных для разработки собран."
