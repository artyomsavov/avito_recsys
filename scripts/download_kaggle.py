"""Скачивает данные для final.ipynb с Kaggle (публичный датасет, токен не нужен).

    python download_kaggle.py            # всё нужное -> data/raw/
    python download_kaggle.py --test     # только contact_eids.csv (проверка доступа)
    python download_kaggle.py -j 32      # больше параллельных потоков (по умолчанию 16)
    python download_kaggle.py --parts 30 # только part_000..029 (для SAMPLE = 0.3)

Раскладка: data/raw/{contact_eids.csv, item_features.parquet, train_data/part_XXX.parquet}.
Повторный запуск докачивает только недостающее; размеры сверяются со списком Kaggle.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://www.kaggle.com/api/v1/datasets"
DS = "nikitakuznetsof/avito-ml-cup-2026"
OUT = "data/raw"


def listing():
    files, token = [], ""
    while True:
        url = f"{API}/list/{DS}" + (f"?pageToken={urllib.parse.quote(token)}" if token else "")
        page = json.load(urllib.request.urlopen(url, timeout=60))
        files += [(f["name"], f["totalBytes"]) for f in page["datasetFiles"]]
        token = page.get("nextPageTokenNullable") or ""
        if not token:
            return files


def target(name):  # путь на Kaggle -> путь, который ждёт ноутбук
    if name.endswith(".parquet") and "/train_data/" in name:
        return f"{OUT}/train_data/{os.path.basename(name)}"
    return f"{OUT}/{name}" if name in ("contact_eids.csv", "item_features.parquet") else None


def fetch(name, size, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return "есть"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{API}/download/{DS}/{urllib.parse.quote(name, safe='')}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dst + ".part", "wb") as f:
                while chunk := r.read(8 << 20):
                    f.write(chunk)
            if os.path.getsize(dst + ".part") == size:
                os.replace(dst + ".part", dst)
                return "ok"
        except Exception as ex:  # сеть: повторить
            print(f"  повтор {attempt + 1} {name}: {ex}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"не скачался {name}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    # по порядку: contact_eids, item_features, part_000, part_001, ...
    files = sorted((t, s, n) for n, s in listing() if (t := target(n)))
    files = [(n, s, t) for t, s, n in files]
    jobs = int(sys.argv[sys.argv.index("-j") + 1]) if "-j" in sys.argv else 16
    if "--parts" in sys.argv:
        n = int(sys.argv[sys.argv.index("--parts") + 1])
        files = [f for f in files if "/train_data/" not in f[2] or f[2][-11:-8] < f"{n:03d}"]
    if "--test" in sys.argv:
        files = [f for f in files if f[0] == "contact_eids.csv"]
    total, t0, done, got = sum(s for _, s, _ in files) / 2**30, time.time(), 0, 0
    print(f"{len(files)} файлов, {total:.1f} GB -> {OUT}/", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        for (name, size, _), res in zip(files, ex.map(lambda f: fetch(*f), files)):
            done += size / 2**30
            got += size / 2**20 if res == "ok" else 0  # скорость только по реально скачанному
            sec = time.time() - t0
            mbs = got / max(sec, 1)
            print(f"[{sec:5.0f} с] {res:4s} {name} ({done:.1f}/{total:.1f} GB, {mbs:.0f} MB/s)")
    td = f"{OUT}/train_data"
    parts = sum(f.endswith(".parquet") for f in os.listdir(td)) if os.path.isdir(td) else 0
    print(f"готово: {parts} партиций train_data", flush=True)


if __name__ == "__main__":
    main()
