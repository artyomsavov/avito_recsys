
# Avito RecSys — двухэтапный retrieval + ranking пайплайн

Решение задачи рекомендации объявлений Avito (DataFest 2026, соревнование завершено): для каждого из 94 408 eval-пользователей предсказать до 160 объявлений, с которыми он проконтактирует.

---

## Результаты

**Метрика:** Recall@160, усреднённый по пользователям.

Локальная валидация на synthetic split (1 426 val-юзеров, 468 positive-примеров):

| Метод | Recall@160 | Retrieval recall (потолок) |
|---|---|---|
| Popular baseline | 0.0078 | — |
| ALS retrieval | 0.0368 | 7.62% |
| **Co-visitation retrieval** | **0.1364** | **17.46%** |
| Combined (ALS ∪ covisit) | — | 21.27% |
| **CatBoost ranker на merged** | **0.1611** | 21.27% |

Итоговый пайплайн превосходит baseline популярности примерно в 20 раз.

**Retrieval recall** — доля таргет-пар, попавших в пул кандидатов. Это жёсткий потолок: ранкер физически не может вытащить в top-160 то, чего нет среди кандидатов. Ранкер выжимает ~76% от доступного потолка (0.1611 / 0.2127).

### Известное ограничение: temporal leakage

ALS и co-visitation матрица построены на данных до `2026-04-15`, включая 7-дневное окно (`04-08 -> 04-15`), из которого собран synth-eval таргет. Retrieval-модели имели доступ к событиям, породившим таргет — классическая временная утечка.

Эффект не измерен точно (потребовался бы честный прогон с cutoff `04-08` без нахлёста). Метрики **валидны для сравнения методов между собой** (все затронуты одинаково, включая baseline), но **не сопоставимы напрямую с официальным лидербордом**.

---

## Архитектура

```
Сырой кликстрим (5 млрд событий, 100 партиций по user_id % 100)
    │
    ├─> aggregate.py ──────> interactions_agg.parquet (119М пар user-item)
    │       фильтр: 28 дней, eid!=7 (без показов)
    │       веса: контакт=5, клик/избранное=1
    │       лимит: top-100 свежих item на юзера
    │
    ├─> build_matrix.py ───> interactions.npz + mappings.npz
    │       CSR-матрица 6.4М юзеров x 34.4М item
    │
    ├─> train_als.py ──────> als_model.npz (factors=32, implicit CPU)
    │
    ├─> build_covisitation.py -> covisit_matrix.parquet (2.52М связей, 1.15М item)
    │       depth=3 соседних события, gap <= 24ч, симметрично
    │       top-50 соседей/item, min 3 юзера на пару
    │
    ├─> build_item_pool.py ─> allowed_item_pool.npz (16.58М item)
    │       фильтр: вертикали {0,2,3,4,5,7} + >= 2 юзера на item
    │
    ├─> generate_candidates*.py -> кандидаты (top-400 на юзера)
    │       ALS: folding-in через recalculate_user для eval-юзеров
    │       Covisit: 15 затравок из истории -> соседи  агрегация скора
    │
    ├─> merge_candidates.py > candidates_merged_synth.parquet
    │
    ├─> build_features.py ──> train_features_merged.parquet (DuckDB)
    │
    ├─> train_ranker.py ────> ranker.cbm (CatBoostRanker, YetiRank)
    │
    └─> eval_recall.py ─────> финальные метрики
```

---

## Ключевые инженерные решения

### Co-visitation оказался главным драйвером (+271% над ALS)

Таргет соревнования — **только новые** item ((user, item) $\notin$ train, novelty-фильтр). ALS через матричную факторизацию учится находить item, *похожие* на историю юзера — это плохо совпадает с задачей "предсказать переключение на новое".

Co-visitation ловит именно переходы: "если юзер интересовался item A, на что он переключался дальше". Пара `(A, B)` по построению состоит из разных item, что естественно выталкивает уже виденное.

Результат: covisit даёт ~22 кандидата на юзера (против 400 у ALS), но эти кандидаты **в 40 раз плотнее по сигналу** — 1 617 попаданий на 107k строк против ~900 на 2.8M.

### Feature importance

```
item_n_users              20.43   <- популярность item доминирует
item_popularity           19.40
covisit_rank              12.55   <- covisit сильнее ALS
als_rank                  11.46
user_top_vertical_match    8.11
als_score                  8.02
category_ext_y             7.45
region_id_y                5.44
vertical_id                4.66
covisit_score              1.46   <- ранг информативнее сырого скора
loc_id_y                   0.74
in_als                     0.24   <- избыточны: дублируют информацию
in_covisit                 0.03      из als_rank/covisit_rank (999 = не было)
```

### Работа под жёстким лимитом памяти (10 GB RAM)

Основная сложность проекта — обработка 5 млрд событий на ноутбуке. Наработанные практики:

| Проблема | Решение |
|---|---|
| `polars streaming` ненадёжен по памяти на `concat + join + group_by` при 100М+ строк (5 подтверждённых OOM) | Перешли на **DuckDB** — out-of-core SQL с жёстким `memory_limit`, спиллит на диск |
| `n_unique()` / `approx_n_unique()` в group_by по 34.4М групп держит sketch на группу | Заменили на `pl.len()` — данные уже дедуплицированы по `(user, item)` |
| `dict{int:int}` на 34.4М записей = 6-10 GB | `np.searchsorted` по отсортированным массивам |
| `item_factors.npy` (4.4 GB) не влезает | `np.load(mmap_mode="r+")` — lazy page loading; `r+` нужен из-за non-const memoryview в Cython-коде implicit |
| `model.recommend()` на 16.58М item × много юзеров разом | Батчинг по 15 юзеров + chunked-запись на диск каждые N батчей |
| Накопление 20М кортежей в Python-списке | Периодический `flush` в parquet-чанки, финальная склейка через `sink_parquet` |

### eval-юзеров нет в train

`train_data/part_*.parquet` не содержит 94 408 eval-пользователей — их история вынесена в `eval_user_events.pq`. Обнаружено проверкой `overlap = 0` между `eval_users.csv` и `mappings.npz`.

Решение: **folding-in** через `recalculate_user=True` в implicit — вектор нового юзера считается на лету из его истории и обученных `item_factors`, без переобучения модели.

---

## Setup

```bash
git clone https://github.com/artyomsavov/avito_recsys
cd avito_recsys
uv sync
```

Данные скачиваются отдельно (не в git, ~41 GB, ссылки от организаторов уже не валидны, можете обратиться, что-нибудь придумаем).

---

## Запуск пайплайна

```bash
# 1. Агрегация кликстрима (~30 мин)
uv run src/aggregate.py
uv run src/aggregate_eval_users.py

# 2. Матрица + ALS (~1 час)
uv run src/build_matrix.py
uv run src/train_als.py

# 3. Co-visitation (~5 мин)
uv run src/build_covisitation.py

# 4. Пул разрешённых item
uv run src/build_item_pool.py

# 5. Кандидаты
uv run src/generate_candidates_synth.py      # ~20 мин (7k synth-юзеров)
uv run src/generate_candidates_covisit.py    # секунды
uv run src/merge_candidates.py

# 6. Фичи + ранкер + метрики
uv run src/build_features.py
uv run src/train_ranker.py                   # ~25 мин
uv run src/eval_recall.py
```

Для реального инференса на 94 408 eval-юзерах: `generate_candidates.py` (folding-in, ~4 часа) $\to$ `add_popular_fallback.py` (15 218 холодных юзеров).

---

## Структура

```
src/
├── aggregate.py                  # 5 млрд событий -> 119М пар (streaming)
├── aggregate_eval_users.py       # то же для eval_user_events.pq
├── build_matrix.py               # CSR + id-маппинги
├── train_als.py                  # implicit ALS, factors=32
├── build_covisitation.py         # item-item матрица переходов
├── build_item_pool.py            # фильтр вертикалей + min_users
├── generate_candidates.py        # ALS folding-in для реальных eval-юзеров
├── generate_candidates_synth.py  # ALS для synth-валидации
├── generate_candidates_covisit.py
├── merge_candidates.py           # union ALS U covisit
├── add_popular_fallback.py       # холодные юзеры -> popularity
├── build_features.py             # feature engineering (DuckDB)
├── train_ranker.py               # CatBoostRanker, YetiRank
├── eval_recall.py                # Recall@160 + сравнение с baseline
└── eval_popular_baseline.py

data/
├── raw/         # исходные данные (gitignored)
├── interim/     # агрегаты, кандидаты
├── processed/   # матрицы, маппинги, covisit
├── features/    # таблицы для ранкера
└── models/      # als_model.npz, ranker.cbm
```

---

## Дальнейшие улучшения

Узкое место сместилось с retrieval на **полноту covisit-матрицы**: покрытие всего 1.15М item из 34.4М (3.3% каталога).

**Дёшево (минуты):**
- `N_SEED_ITEMS` 15 $\to$ 30-50: больше затравок → больше кандидатов
- `MIN_PAIR_USERS` 3 $\to$ 2: шире покрытие матрицы ценой шума
- Убрать избыточные `in_als` / `in_covisit`, нормализовать `covisit_score` на число затравок

**Средне:**
- Направленная co-visitation (A $\to$ B $\neq$ B $\to$ A) вместо симметричной
- Sid-based кандидаты через `sid_0..sid_3` (RQ-эмбеддинги контента) — для холодного старта
- Popularity внутри вертикали юзера вместо глобальной

**Дорого, но методологически необходимо:**
- Честный synth-прогон с cutoff `2026-04-08` (retrieval без нахлёста с eval-окном) — единственный способ измерить реальную величину temporal leakage

---

## Железо

Разработка на ноутбуке: Ryzen 7 7840HS, 10 GB доступной RAM. Все компоненты работают на CPU.

Доступный сервер (0.25x A100 (20GB), 3 CPU, 16 GB RAM, 48 GB диска) не использовался: узкое место задачи — I/O и RAM на агрегации, а не матричные вычисления. ALS на 119М взаимодействий с `factors=32` обучается на CPU за приемлемое время.