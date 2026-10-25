"""Пул разрешённых для рекомендации item: нужные вертикали + >= 2 юзеров до cutoff."""

import numpy as np
import polars as pl

from config import ALLOWED_VERTICALS, ITEM_FEATURES, ids_to_idx, parse_run

MIN_USERS_PER_ITEM = 2


def build_item_pool() -> None:
    run = parse_run(__doc__)
    print("1. Загружаем маппинг item_id -> idx (train)...")
    train_item_ids = np.load(run.processed / "mappings.npz")["item_ids"]

    print("2. Читаем item_features (только нужные колонки)...")
    item_features = (
        pl.scan_parquet(ITEM_FEATURES)
        .select(["item_id", "vertical_id"])
        .filter(pl.col("vertical_id").is_in(ALLOWED_VERTICALS))
        .collect(engine="streaming")
    )

    print("3. Считаем число юзеров на item (агрегат уже уникален по (user, item))...")
    item_user_counts = (
        pl.scan_parquet(run.interim / "interactions_agg.parquet")
        .group_by("item_id")
        .agg(pl.len().alias("n_users"))
        .collect(engine="streaming")
    )

    print("4. Собираем финальный разрешённый список item_id...")
    allowed_item_ids = (
        item_features.join(item_user_counts, on="item_id", how="inner")
        .filter(pl.col("n_users") >= MIN_USERS_PER_ITEM)["item_id"]
        .to_numpy()
    )
    print(f"   Разрешённых item: {len(allowed_item_ids):,} из {len(train_item_ids):,} train-item")

    idx, valid = ids_to_idx(train_item_ids, allowed_item_ids)
    allowed_item_idx = np.sort(idx[valid]).astype(np.int64)

    out_path = run.processed / "allowed_item_pool.npz"
    np.savez_compressed(out_path, allowed_item_idx=allowed_item_idx)
    print(f"Готово! Пул из {len(allowed_item_idx):,} item сохранён в {out_path}")


if __name__ == "__main__":
    build_item_pool()
