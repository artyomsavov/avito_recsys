from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

N_CANDIDATES = 400


def add_popular_fallback() -> None:
    print("1. Загружаем список всех eval-юзеров и текущих кандидатов...")
    eval_users = pl.read_csv(RAW_DIR / "eval_users.csv")["user_id"]
    candidates = pl.read_parquet(INTERIM_DIR / "candidates.parquet")
    covered_users = set(candidates["user_id"].unique().to_list())

    all_users = set(eval_users.to_list())
    cold_users = sorted(all_users - covered_users)
    print(f"   Всего eval-юзеров: {len(all_users):,}")
    print(f"   Уже покрыто ALS: {len(covered_users):,}")
    print(f"   Холодных (нужен fallback): {len(cold_users):,}")

    if not cold_users:
        print("Холодных юзеров нет, fallback не требуется.")
        return

    print("2. Восстанавливаем allowed item_id из индексов пула...")
    mappings = np.load(PROCESSED_DIR / "mappings.npz")
    train_item_ids = mappings["item_ids"]
    allowed_item_idx = np.load(PROCESSED_DIR / "allowed_item_pool.npz")["allowed_item_idx"]
    allowed_item_ids = set(train_item_ids[allowed_item_idx].tolist())

    print("3. Считаем популярность item по train-агрегату (потоково)...")
    popularity = (
        pl.scan_parquet(INTERIM_DIR / "interactions_agg.parquet")
        .group_by("item_id")
        .agg(pl.col("weight").sum().alias("total_weight"))
        .collect(engine="streaming")
    )

    print("4. Фильтруем по разрешённому пулу и берём top-N...")
    popular_allowed = (
        popularity.filter(pl.col("item_id").is_in(allowed_item_ids))
        .sort("total_weight", descending=True)
        .head(N_CANDIDATES)
    )
    popular_item_ids = popular_allowed["item_id"].to_list()
    print(f"   Топ item для fallback: {len(popular_item_ids)} (нужно {N_CANDIDATES})")

    if len(popular_item_ids) < N_CANDIDATES:
        print(f"   ВНИМАНИЕ: доступно только {len(popular_item_ids)} item, меньше N_CANDIDATES")

    print("5. Строим fallback-строки для холодных юзеров...")
    fallback_rows = [
        (uid, item_id, 0.0, rank)
        for uid in cold_users
        for rank, item_id in enumerate(popular_item_ids)
    ]
    fallback_df = pl.DataFrame(
        fallback_rows,
        schema=["user_id", "item_id", "als_score", "als_rank"],
        orient="row",
    )

    print("6. Помечаем источник и объединяем с основными кандидатами...")
    candidates = candidates.with_columns(pl.lit("als").alias("source"))
    fallback_df = fallback_df.with_columns(pl.lit("popular_fallback").alias("source"))

    combined = pl.concat([candidates, fallback_df])

    out_path = INTERIM_DIR / "candidates_full.parquet"
    combined.write_parquet(out_path)

    print(f"Готово! Итого строк: {len(combined):,}")
    print(f"Уникальных юзеров: {combined['user_id'].n_unique():,} (должно быть {len(all_users):,})")
    print(f"Сохранено в {out_path}")


if __name__ == "__main__":
    add_popular_fallback()
