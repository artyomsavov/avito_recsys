#!/bin/bash
# Скачивание данных для пайплайна (докачка при нестабильном интернете):
# item_features, contact_eids и полный train (100 партиций, ~40 GB).
# Таргет synth-валидации после скачивания: bash scripts/make_local_eval.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR" || exit

BASE="https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4"

# Функция для надежного скачивания с бесконечными попытками
download_reliable() {
    local url=$1
    local filename=$(basename "$url")
    
    echo "=================================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Начинаем загрузку/докачку: $filename"
    
    # Бесконечный цикл: крутится, пока файл не скачается на 100% (пока curl не вернет код 0)
    while true; do
        # --retry 5: curl сам попытается переподключиться при мелких разрывах
        # -C -: докачка с места обрыва
        curl -O -C - --retry 5 --retry-delay 3 "$url"
        
        # Если curl завершился успешно (код 0), выходим из цикла
        if [ $? -eq 0 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') | [Успех] $filename загружен полностью."
            break
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') | [Сбой сети] Обрыв на файле $filename. Ждем 10 секунд и возобновляем докачку..."
            sleep 10
        fi
    done
}

echo "[1/2] Скачивание базовых файлов..."
download_reliable "$BASE/item_features.parquet"
download_reliable "$BASE/contact_eids.csv"

echo "[2/2] Скачивание и потоковая распаковка полного трейна..."
for i in 000-019 020-039 040-059 060-079 080-099; do
    # Узнаем номер последнего файла в текущем блоке (например, 019)
    LAST_PART=$(echo $i | cut -d'-' -f2)
    
    # Если последний паркет из блока уже лежит в папке, значит блок полностью готов
    if [ -f "train_data/part_${LAST_PART}.parquet" ]; then
        echo "--------------------------------------------------"
        echo "$(date '+%Y-%m-%d %H:%M:%S') | -> Блок train_${i} уже распакован, пропускаем."
        continue
    fi

    echo "--------------------------------------------------"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Обработка блока train_${i}..."
    
    download_reliable "$BASE/train_${i}.zip"
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Начинаю распаковку train_${i}.zip..."
    
    # Оператор if выполнит rm -f ТОЛЬКО если unzip завершился без ошибок
    if unzip -o train_${i}.zip; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | [Успех] Блок train_${i} распакован. Удаляю ZIP-архив."
        rm -f train_${i}.zip
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') | [КРИТИЧЕСКАЯ ОШИБКА] Не удалось распаковать train_${i}.zip."
        echo "Скрипт экстренно остановлен, чтобы не удалить битый архив. Запустите скрипт заново для докачки."
        exit 1
    fi
done

echo "=================================================="
echo "$(date '+%Y-%m-%d %H:%M:%S') | Готово! Данные развернуты."
