"""Единая конфигурация прогонов: временной cutoff и пути артефактов.

Глобальный темпоральный сплит: каждый шаг пайплайна (агрегаты, ALS, co-visitation,
пул item, фичи, популярность) строится строго на событиях с timestamp < cutoff_ms.
Таргет synth-валидации (local_eval.csv) — новые контакты с cutoff_ms + 12h
(см. scripts/make_local_eval.sh).

Прогоны отличаются только cutoff:
    synth — честная локальная валидация (cutoff = порог local_eval.csv);
    leaky — воспроизведение старых экспериментов (cutoff = конец train, перекрывает окно
            таргета) — только для измерения утечки.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = DATA_DIR / "runs"
TMP_DIR = DATA_DIR / "duckdb_tmp"

TRAIN_PARTS = RAW_DIR / "train_data" / "part_*.parquet"
ITEM_FEATURES = RAW_DIR / "item_features.parquet"
CONTACT_EIDS = RAW_DIR / "contact_eids.csv"
LOCAL_EVAL = RAW_DIR / "local_eval.csv"
# замороженный val-сплит synth-юзеров для ранкера (в git, одинаковый во всех прогонах)
VAL_USERS = DATA_DIR / "splits" / "synth_val_users.csv"

# Фактический порог local_eval.csv: prepare_local_eval.py читает "2026-04-08T00:00:00"
# как naive datetime в MSK -> 2026-04-07T21:00:00Z (проверено: 0 таргет-пар раньше).
SYNTH_CUTOFF_MS = 1775595600000
# Конец train: 2026-04-15T00:00:00Z.
TRAIN_END_MS = 1776211200000

DAY_MS = 24 * 60 * 60 * 1000
WINDOW_DAYS = 28
SHOW_EID = 7  # показы не используем
ALLOWED_VERTICALS = (0, 2, 3, 4, 5, 7)
K = 160


@dataclass(frozen=True)
class Run:
    name: str
    cutoff_ms: int
    leaky: bool = False  # cutoff перекрывает окно таргета — только для сравнения

    @property
    def start_ms(self) -> int:
        return self.cutoff_ms - WINDOW_DAYS * DAY_MS

    @property
    def root(self) -> Path:
        return RUNS_DIR / self.name

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def features(self) -> Path:
        return self.root / "features"


RUNS = {
    "synth": Run("synth", SYNTH_CUTOFF_MS),
    "leaky": Run("leaky", TRAIN_END_MS, leaky=True),
}


def parse_run(description: str | None = None) -> Run:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run", choices=RUNS, required=True)
    run = RUNS[parser.parse_args().run]
    for d in (run.interim, run.processed, run.models, run.features, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"[run={run.name}] cutoff_ms={run.cutoff_ms}, окно {WINDOW_DAYS} дней")
    return run


def window_filter(run: Run) -> pl.Expr:
    """События из окна [cutoff - 28d, cutoff) без показов."""
    return (
        (pl.col("timestamp") >= run.start_ms)
        & (pl.col("timestamp") < run.cutoff_ms)
        & (pl.col("eid") != SHOW_EID)
    )


def contact_eids() -> list[int]:
    return pl.read_csv(CONTACT_EIDS)["mapped_eid"].to_list()


def target_users() -> np.ndarray:
    return pl.read_csv(LOCAL_EVAL)["user_id"].unique().sort().to_numpy()


def ids_to_idx(sorted_ids: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """id -> индекс в отсортированном массиве через searchsorted (без dict на 34М ключей).
    Возвращает (idx, mask найденных)."""
    idx = np.searchsorted(sorted_ids, ids)
    clipped = np.clip(idx, 0, len(sorted_ids) - 1)
    valid = (idx < len(sorted_ids)) & (sorted_ids[clipped] == ids)
    return idx, valid
