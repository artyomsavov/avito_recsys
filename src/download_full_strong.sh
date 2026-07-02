#!/bin/bash
# Бронебойный скрипт для скачивания при слабом/нестабильном интернете

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

echo "[1/3] Скачивание базовых файлов..."
download_reliable "$BASE/item_features.parquet"
download_reliable "$BASE/contact_eids.csv"
download_reliable "$BASE/eval_users.csv"
download_reliable "$BASE/prepare_local_eval.py"
download_reliable "$BASE/popular.py"

echo "[2/3] Обработка истории пользователей..."
# Проверяем, есть ли уже распакованный файл (в твоем дереве он называется eval_user_events.pq)
if [ ! -f "eval_user_events.pq" ]; then
    download_reliable "$BASE/eval_user_events.zip"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Начинаю распаковку eval_user_events.zip..."
    
    if unzip -o eval_user_events.zip; then
        rm -f eval_user_events.zip
        echo "$(date '+%Y-%m-%d %H:%M:%S') | [Успех] eval_user_events распакован. Архив удален."
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') | [ОШИБКА] Не удалось распаковать eval_user_events.zip. Скрипт остановлен."
        exit 1
    fi
else
    echo "-> eval_user_events.pq уже существует, пропускаем скачивание."
fi

echo "[3/3] Скачивание и потоковая распаковка полного трейна..."
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
echo "$(date '+%Y-%m-%d %H:%M:%S') | [Full] Готово! Полный датасет развернут."
