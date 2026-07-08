from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"

K = 160
USED_VERTICALS = (0, 2, 3, 4, 5, 7)


def main() -> None:
    # те же val-юзеры, что и в eval_recall.py — для честного сравнения
    val_users = pl.read_parquet(FEATURES_DIR / "val_predictions.parquet")["user_id"].unique()

    full_target = pl.read_csv(RAW_DIR / "local_eval.csv").filter(
        pl.col("user_id").is_in(val_users.to_list())
    )
    n_users = full_target["user_id"].n_unique()
    print(f"Val-юзеров с таргетом: {n_users:,}")

    print(
        f"Считаем топ-{K} популярных item (источник: interactions_agg.parquet, вертикали {USED_VERTICALS})..."
    )
    items = (
        pl.scan_parquet(RAW_DIR / "item_features.parquet")
        .select(["item_id", "vertical_id"])
        .filter(pl.col("vertical_id").is_in(list(USED_VERTICALS)))
    )
    top_items = (
        pl.scan_parquet(INTERIM_DIR / "interactions_agg.parquet")
        .join(items, on="item_id", how="inner")
        .group_by("item_id")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(K)
        .select("item_id")
        .collect(engine="streaming")
    )
    top_item_ids = top_items["item_id"].to_list()
    print(f"   Отобрано {len(top_item_ids)} item (один и тот же список для всех юзеров).")

    hits = (
        full_target.filter(pl.col("item_id").is_in(top_item_ids))
        .group_by("user_id")
        .agg(pl.len().alias("hits"))
    )
    totals = full_target.group_by("user_id").agg(pl.len().alias("total_target"))

    per_user = (
        totals.join(hits, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .with_columns((pl.col("hits") / pl.col("total_target")).alias("recall"))
    )
    recall = per_user["recall"].mean()
    print(f"\nRecall@{K} — Popular baseline: {recall:.4f}")


if __name__ == "__main__":
    main()
