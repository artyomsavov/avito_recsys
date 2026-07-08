from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ALLOWED_VERTICALS = {0, 2, 3, 4, 5, 7}
MIN_USERS_PER_ITEM = 2


def build_item_pool() -> None:
    print("1. Загружаем маппинг item_id -> idx (train)...")
    mappings = np.load(PROCESSED_DIR / "mappings.npz")
    train_item_ids = mappings["item_ids"]  # уже отсортирован (см. build_matrix.py)
    n_train_items = len(train_item_ids)

    print("2. Читаем item_features (только нужные колонки)...")
    item_features = (
        pl.scan_parquet(RAW_DIR / "item_features.parquet")
        .select(["item_id", "vertical_id"])
        .collect(engine="streaming")
    )

    print("3. Считаем число уникальных юзеров на item (из train-агрегата)...")
    interactions = pl.scan_parquet(INTERIM_DIR / "interactions_agg.parquet").select(
        ["user_id", "item_id"]
    )
    item_user_counts = (
        interactions.group_by("item_id")
        .agg(pl.col("user_id").n_unique().alias("n_users"))
        .collect(engine="streaming")
    )

    print("4. Собираем финальный разрешённый список item_id...")
    allowed = (
        item_features.filter(pl.col("vertical_id").is_in(ALLOWED_VERTICALS))
        .join(item_user_counts, on="item_id", how="inner")
        .filter(pl.col("n_users") >= MIN_USERS_PER_ITEM)
        .select("item_id")
    )
    allowed_item_ids = allowed["item_id"].to_numpy()
    print(f"   Разрешённых item: {len(allowed_item_ids):,} из {n_train_items:,} train-item")

    print("5. Переводим item_id -> idx через searchsorted (без dict!)...")
    idx = np.searchsorted(train_item_ids, allowed_item_ids)
    # проверка корректности searchsorted — на случай item_id, которых нет в train_item_ids
    valid_mask = (idx < n_train_items) & (
        train_item_ids[np.clip(idx, 0, n_train_items - 1)] == allowed_item_ids
    )
    allowed_item_idx = idx[valid_mask]
    dropped = len(allowed_item_ids) - len(allowed_item_idx)
    if dropped:
        print(f"   Внимание: {dropped:,} allowed item_id не найдены в train mappings (пропущены)")

    allowed_item_idx = np.sort(allowed_item_idx).astype(np.int64)

    out_path = PROCESSED_DIR / "allowed_item_pool.npz"
    np.savez_compressed(out_path, allowed_item_idx=allowed_item_idx)
    print(f"Готово! Пул из {len(allowed_item_idx):,} item сохранён в {out_path}")


if __name__ == "__main__":
    build_item_pool()
