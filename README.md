
# Avito RecSys — двухэтапный retrieval + ranking пайплайн

Решение задачи рекомендации объявлений Avito (DataFest 2026, соревнование завершено): для пользователя предсказать до 160 новых объявлений, с которыми он проконтактирует.

Репозиторий содержит пайплайн и честную локальную валидацию (synthetic split); инференс на официальных eval-юзерах не входит.

---

## Финальный пайплайн: `final.ipynb`

Весь пайплайн в одном ноутбуке на 200 строк кода. Кандидаты — блуждания по графу item → сущность → item (юзер × категория × регион, юзер × день, юзер), ранжирование — CatBoost YetiRank. Ранкер учится на неделе до порога, тест — неделя после, таргет собирается логикой организаторов.

| `SAMPLE` | Данные | Где | Recall@160 (тест W0) |
|---|---|---|---|
| `0.01` | 1 партиция | ноутбук, CPU, ~60 мин (v2) | **0.1327** [0.1252; 0.1400] v2; было 0.1227 в v1 |
| `0.3` | 30 партиций | A100 80 GB, 28 мин | 0.1175 [0.1154; 0.1195], 76 081 юзер (v1) |
| `0.75` | 75 партиций | A100 80 GB, 89 мин | **0.1376** [0.1356; 0.1396], 90 318 юзеров (v1) |
| `1.0` | 100 партиций | v1 не проходит (граф `s24` ~2.4 млрд nnz > int32); в v2 граф по блокам — на GPU ещё не запускался | — |

v2 (эксп. 9): NOUT 1000, co-visitation и BM25 на графах `s24`/`u0`, популярное в сегментах s0 × loc, граф по блокам. Оценка памяти `GB_PER_PART = 0.4` взята из замера на 30% (~0.33 GB на граф × партицию): при `SAMPLE ≥ 0.5` графы строятся по одному за проход. Метрики при разных `SAMPLE` считаются на разных юзерах, но на 75% выросли все бакеты.

Больше ничего менять не нужно. Ячейка зависимостей ставит недостающие пакеты с точными версиями (на GPU — `cupy-cuda12x`), устройство выбирается автоматически, первая ячейка после конфига — smoke-тест всего пути на 3% юзеров. Этапы сохраняют чекпоинты в `data/final/`, повторный запуск продолжает с последнего готового.

### Запуск

Данные (38 GB, токен Kaggle не нужен, докачивает недостающее):

```bash
python scripts/download_kaggle.py              # всё -> data/raw/
python scripts/download_kaggle.py --parts 30   # только part_000..029 (для SAMPLE = 0.3)
```

**Docker** (скачивает данные и запускает ноутбук; нужен NVIDIA Container Toolkit, без `--gpus all` — на CPU):

```bash
docker build -t avito .
docker run -d --name avito --gpus all --restart on-failure:3 -e SAMPLE=1.0 -v $PWD/data:/app/data avito
docker logs -f avito
```

**Без Docker** (Jupyter-сервер; нужен Python ≥ 3.11, процесс переживает закрытие вкладки):

```bash
jupyter nbconvert --to script final.ipynb && sed -i 's/^SAMPLE = 0.01/SAMPLE = 1.0/' final.py
setsid nohup python final.py > run.log 2>&1 < /dev/null &
```

Через 1–3 минуты в логе — Recall smoke-теста; если вместо него ошибка, дальше не ждать. Результат: `data/final/sample_<SAMPLE>/metrics.json`, предсказания — `top160.parquet` там же.

Старый пайплайн (ALS + covisit, ниже) в том же режиме одной партиции даёт 0.0040. Как выбирались компоненты: [docs/experiments.md](docs/experiments.md), эксперимент 4. Стенд экспериментов — `exp/` (`lib.py`, `run.py`, лог `results.jsonl`, исходник ноутбука `final_src.py`).

---

## Результаты старого пайплайна

**Метрика:** Recall@160, усреднённый по пользователям.

Локальная валидация на synthetic split: таргет — новые контакты после `2026-04-07 21:00 UTC + 12h`, 1 426 val-юзеров, 2 388 таргет-пар. Все шаги пайплайна строятся **строго на событиях до порога** (глобальный темпоральный сплит).

| Метод | Recall@160 | Retrieval recall (потолок) |
|---|---|---|
| Popular baseline | 0.0048 | — |
| ALS retrieval | 0.0137 | — |
| Co-visitation retrieval | 0.0216 | — |
| **CatBoost ranker на ALS ∪ covisit** | **0.0325** | 4.52% |

Итоговый пайплайн превосходит baseline популярности в ~6.7 раза. Ранкер добавляет +50% к лучшему источнику и выжимает ~72% доступного потолка (0.0325 / 0.0452).

Recall источника (ALS, covisit) считается по его собственным кандидатам. Оценка ранкера шумная (611 positive в train): одна только перестановка строк на входе CatBoost давала 0.0312 вместо 0.0325, поэтому порядок строк теперь зафиксирован.

**Retrieval recall** — доля таргет-пар, попавших в пул кандидатов. Это жёсткий потолок: ранкер физически не может вытащить в top-160 то, чего нет среди кандидатов.

### Исправленная утечка

Ранние версии строили ALS, co-visitation, пул item и фичи по данным до `2026-04-15`, перекрывая окно synth-таргета. Метрики были завышены в 1.6–6.3 раза, причём **по-разному для разных методов**, так что сравнение методов между собой тоже было некорректным:

| Метод | С утечкой | Честно | Завышение |
|---|---|---|---|
| Popular | 0.0078 | 0.0048 | ×1.6 |
| ALS | 0.0368 | 0.0137 | ×2.7 |
| Co-visitation | 0.1357 | 0.0216 | ×6.3 |
| CatBoost ranker | 0.1613 | 0.0325 | ×5.0 |
| Потолок retrieval | 19.60% | 4.52% | ×4.3 |

Разбор утечки: [docs/leakage_audit.md](docs/leakage_audit.md). Оба прогона воспроизводятся одним кодом (`--run leaky` и `--run synth` отличаются только cutoff). Исторические числа README до исправления: covisit 0.1364 и ranker 0.1611; разница с таблицей — из-за нефиксированного порядка строк (tie-break при равных скорах и вход CatBoost).

---

## Архитектура

```
Сырой кликстрим (5 млрд событий, 100 партиций по user_id % 100)
    │                     все шаги: только события из [cutoff - 28d, cutoff)
    │
    ├─> aggregate.py ──────> interactions_agg.parquet (~119М пар user-item)
    │       фильтр: 28 дней, eid!=7 (без показов)
    │       веса: контакт=5, клик/избранное=1
    │       лимит: top-100 свежих item на юзера
    │
    ├─> build_matrix.py ───> interactions.npz + mappings.npz
    │       CSR-матрица ~6.4М юзеров x 34.3М item
    │
    ├─> train_als.py ──────> item_factors.npy, user_factors.npy (factors=32, implicit CPU)
    │
    ├─> build_covisitation.py -> covisit_matrix.parquet (2.53М связей, 1.14М item)
    │       depth=3 соседних события, gap <= 24ч, симметрично
    │       top-50 соседей/item, min 3 юзера на пару
    │
    ├─> build_item_pool.py ─> allowed_item_pool.npz (16.53М item)
    │       фильтр: вертикали {0,2,3,4,5,7} + >= 2 юзера на item
    │
    ├─> generate_candidates_als.py -> top-400 ALS на юзера (по обученным user_factors)
    ├─> generate_candidates_covisit.py -> top-400 covisit на юзера
    │       15 последних item истории -> соседи -> сумма весов
    │
    ├─> merge_candidates.py > ALS ∪ covisit + popular-fallback для холодных юзеров
    │
    ├─> build_features.py ──> features.parquet (DuckDB) + проверка сплита
    │
    ├─> train_ranker.py ────> ranker.cbm (CatBoostRanker, YetiRank)
    │
    └─> evaluate.py ────────> metrics.json
```

### Прогоны

Все параметры прогона задаются в [`src/config.py`](src/config.py), артефакты пишутся в `data/runs/<run>/`:

| Прогон | cutoff | Назначение |
|---|---|---|
| `synth` | `2026-04-07 21:00 UTC` | честная локальная валидация |
| `leaky` | `2026-04-15 00:00 UTC` (конец train) | воспроизведение старых (протёкших) экспериментов, только для измерения утечки |

Оба прогона используют один таргет (`local_eval.csv`) и один val-сплит ранкера (`data/splits/synth_val_users.csv`, в git).

Порог `synth` — это фактический порог `local_eval.csv`. `prepare_local_eval.py` читает `"2026-04-08T00:00:00"` как naive datetime и на машине в MSK получает `2026-04-07 21:00 UTC`.

Защита от регрессии утечки (assert в коде):
- `aggregate.py`: ни одного события `>= cutoff` в агрегате;
- `generate_candidates_covisit.py`: ни одной затравки `>= cutoff`;
- `build_features.py`: пересечение таргета с историей до cutoff равно 0.

### Воспроизводимость

| Что | Как зафиксировано |
|---|---|
| Окружение | `uv.lock`, `.python-version` |
| Таргет | `scripts/make_local_eval.sh` + sha256 отсортированного содержимого |
| Val-сплит ранкера | `data/splits/synth_val_users.csv` в git |
| Случайность | `random_state=42` в ALS, `random_seed=42` в CatBoost |
| Порядок при равных скорах | явные tie-break в SQL (`ORDER BY score DESC, item_id`), stable-сортировка top-n |

Результат каждого прогона пишется в `data/runs/<run>/metrics.json`; ожидаемые значения — в таблицах выше.

---

## Ключевые инженерные решения

### Co-visitation сильнее ALS (+58%)

Таргет соревнования — **только новые** item ((user, item) $\notin$ train, novelty-фильтр). ALS через матричную факторизацию учится находить item, *похожие* на историю юзера, — это плохо совпадает с задачей «предсказать переключение на новое».

Co-visitation ловит именно переходы: «если юзер интересовался item A, на что он переключался дальше». Пара `(A, B)` по построению состоит из разных item, что естественно выталкивает уже виденное. Источники дополняют друг друга: ранкер на объединении даёт +50% к covisit.

### Feature importance

```
als_rank                  29.32
item_n_users              15.67   <- популярность item
covisit_rank              14.44
item_popularity            9.61
als_score                  8.95
user_top_vertical_match    8.14
region_id_y                7.34
vertical_id                4.55
in_covisit                 1.64
covisit_score              0.34   <- ранг информативнее сырого скора
in_als                     0.00
category_ext_y             0.00
loc_id_y                   0.00
```

### Работа под жёстким лимитом памяти (10 GB RAM)

Основная сложность проекта — обработка 5 млрд событий на ноутбуке. Наработанные практики:

| Проблема | Решение |
|---|---|
| `polars streaming` ненадёжен по памяти на `concat + join + group_by` при 100М+ строк (5 подтверждённых OOM) | Перешли на **DuckDB** — out-of-core SQL с жёстким `memory_limit`, спиллит на диск |
| `n_unique()` / `approx_n_unique()` в group_by по 34.4М групп держит sketch на группу | Заменили на `pl.len()` — данные уже дедуплицированы по `(user, item)` |
| `dict{int:int}` на 34.4М записей = 6-10 GB | `np.searchsorted` по отсортированным массивам |
| `item_factors.npy` (4.4 GB) не влезает | `np.load(mmap_mode="r+")` — lazy page loading |
| `model.recommend(items=pool)` заново собирает срез факторов 16.5М x 32 на каждый батч (~8 ч на 6k юзеров) | Срез по пулу один раз (2 GB) + `U @ F.T` и построчный `argpartition` (~30 мин) |
| Накопление 20М кортежей в Python-списке | Периодический `flush` в parquet-чанки, финальная склейка через `sink_parquet` |

---

## Setup

```bash
git clone https://github.com/artyomsavov/avito_recsys
cd avito_recsys
uv sync
```

Данные скачиваются отдельно (не в git, ~40 GB, ссылки от организаторов уже не валидны, можете обратиться, что-нибудь придумаем):

```bash
bash scripts/download_data.sh   # item_features, contact_eids, train_data (100 партиций)
bash scripts/make_local_eval.sh # таргет synth-валидации + сверка sha256
```

`make_local_eval.sh` запускает скрипт организаторов [`external/prepare_local_eval.py`](external/prepare_local_eval.py) ровно так, как был получен исходный таргет (только `part_000`, `TZ=Europe/Moscow`), и сверяет результат с зафиксированными контрольными суммами.

---

## Запуск пайплайна

```bash
bash scripts/run_pipeline.sh synth            # ~3 ч: все шаги + ранкер + metrics.json
bash scripts/run_pipeline.sh synth train_als  # продолжить с указанного шага
bash scripts/run_pipeline.sh leaky            # протёкший вариант для сравнения
```

Отдельный шаг: `uv run src/<step>.py --run synth|leaky`. Порядок и примерное время:

| Шаг | Время |
|---|---|
| `aggregate` | ~12 мин |
| `build_matrix` | ~7 мин |
| `train_als` | ~80 мин |
| `build_covisitation` | ~17 мин |
| `build_item_pool` | ~2 мин |
| `generate_candidates_als` | ~30 мин |
| `generate_candidates_covisit`, `merge_candidates`, `build_features` | ~5 мин |
| `train_ranker` | ~20 мин |
| `evaluate` | <1 мин |

---

## Структура

```
final.ipynb                       # финальный пайплайн (200 строк), собирается из exp/final_src.py

exp/                              # стенд экспериментов на 1 партиции
├── lib.py                        # таргет, блуждания, признаки, ранкер, метрика + бутстреп
├── run.py                        # группы экспериментов (W2 -> W1, финальный тест W1 -> W0)
├── baseline.py                   # старый пайплайн в той же схеме
├── report.py                     # results.jsonl -> таблицы для docs/experiments.md
├── build_notebook.py             # final_src.py -> final.ipynb
└── results.jsonl                 # лог всех прогонов

src/                              # старый пайплайн (ALS + covisit)
├── config.py                     # прогоны (cutoff, юзеры), пути, общие хелперы
├── aggregate.py                  # 5 млрд событий -> агрегат (user, item) до cutoff
├── build_matrix.py               # CSR + id-маппинги
├── train_als.py                  # implicit ALS, factors=32
├── build_covisitation.py         # item-item матрица переходов
├── build_item_pool.py            # фильтр вертикалей + min_users
├── generate_candidates_als.py    # ALS top-400 по user_factors
├── generate_candidates_covisit.py
├── merge_candidates.py           # ALS ∪ covisit + popular-fallback
├── build_features.py             # feature engineering (DuckDB) + проверка сплита
├── train_ranker.py               # CatBoostRanker, YetiRank
└── evaluate.py                   # Recall@160: popular / ALS / covisit / ranker

scripts/
├── download_data.sh              # скачивание данных
├── make_local_eval.sh            # воспроизведение таргета + сверка sha256
├── run_pipeline.sh               # весь пайплайн одного прогона
└── leak_check_covisit.py         # абляция утечки covisit (затравки vs матрица)

external/prepare_local_eval.py    # код организаторов (без изменений)

data/                             # не в git
├── raw/                          # исходные данные + local_eval.csv
├── splits/synth_val_users.csv    # замороженный val-сплит ранкера (1 426 юзеров, в git)
└── runs/<run>/                   # interim/ processed/ models/ features/ metrics.json
```

---

## Дальнейшие улучшения

Узкое место — **retrieval**: потолок 4.52%, covisit покрывает всего 1.14М item из 34.3М (3.3% каталога).

**Методология валидации (следующий шаг):**
- Отдельная валидация для выбора итерации CatBoost: сейчас `eval_set` совпадает с юзерами, на которых считается Recall (`use_best_model`)
- Метки ранкера из более раннего окна (`[T-7d, T)`), а не из тестового: сейчас ранкер учится на других юзерах той же недели
- Перегенерировать `local_eval` по всем 100 партициям с UTC-порогом (сейчас только `part_000`)

**Модель:**
- Novelty-фильтр кандидатов: уже виденные item не могут быть таргетом и тратят слоты
- `N_SEED_ITEMS` 15 $\to$ 30-50, `MIN_PAIR_USERS` 3 $\to$ 2
- Направленная co-visitation (A $\to$ B $\neq$ B $\to$ A) вместо симметричной
- Sid-based кандидаты через `sid_0..sid_3` (RQ-эмбеддинги контента) — для холодного старта (886 из 7 131 synth-юзеров без истории)
- Popularity внутри вертикали юзера вместо глобальной

---

## Железо

Разработка на ноутбуке: Ryzen 7 7840HS, 10 GB доступной RAM. Все компоненты работают на CPU.

Доступный сервер (0.25x A100 (20GB), 3 CPU, 16 GB RAM, 48 GB диска) не использовался: узкое место задачи — I/O и RAM на агрегации, а не матричные вычисления. ALS на 119М взаимодействий с `factors=32` обучается на CPU за приемлемое время.
