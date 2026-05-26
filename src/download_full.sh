#!/bin/bash
# Скрипт для полного скачивания (весь кликстрим 40+ ГБ)
# Запуск: bash src/download_full.sh

RAW_DIR="data/raw"
mkdir -p $RAW_DIR
cd $RAW_DIR

BASE="https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4"

echo "[Full] Скачивание базовых файлов..."
curl -O -C - $BASE/item_features.parquet
curl -O -C - $BASE/contact_eids.csv
curl -O -C - $BASE/eval_users.csv
curl -O -C - $BASE/prepare_local_eval.py
curl -O -C - $BASE/popular.py
curl -O -C - $BASE/eval_user_events.zip

echo "[Full] Распаковка истории пользователей..."
unzip -o eval_user_events.zip
rm -f eval_user_events.zip

echo "[Full] Скачивание и потоковая распаковка полного трейна..."
for i in 000-019 020-039 040-059 060-079 080-099; do
    echo "-> Обработка блока train_${i}..."
    curl -O -C - $BASE/train_${i}.zip
    unzip -o train_${i}.zip
    # Удаляем архив ДО скачивания следующего, чтобы не раздувать диск
    rm -f train_${i}.zip 
done

echo "[Full] Готово! Полный датасет развернут."
