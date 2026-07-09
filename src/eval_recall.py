from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
RAW_DIR = DATA_DIR / "raw"
K = 160


def recall_at_k(preds: pl.DataFrame, score_col: str, full_target: pl.DataFrame) -> float:
    topk = (
        preds.sort(["user_id", score_col], descending=[False, True])
        .group_by("user_id", maintain_order=True)
        .head(K)
        .select(["user_id", "item_id"])
    )
    hits = (
        topk.join(
            full_target.with_columns(pl.lit(1).alias("is_t")),
            on=["user_id", "item_id"],
            how="inner",
        )
        .group_by("user_id")
        .agg(pl.len().alias("hits"))
    )
    # знаменатель — полное число таргетов юзера из local_eval.csv
    totals = full_target.group_by("user_id").agg(pl.len().alias("total_target"))

    per_user = (
        totals.join(hits, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .with_columns((pl.col("hits") / pl.col("total_target")).alias("recall"))
    )
    return per_user["recall"].mean()


def main() -> None:
    val = pl.read_parquet(FEATURES_DIR / "val_predictions.parquet")
    feats = pl.read_parquet(FEATURES_DIR / "train_features_merged.parquet").select(
        ["user_id", "item_id", "als_score", "covisit_score"]
    )
    val = val.join(feats, on=["user_id", "item_id"], how="left")

    val_users = val["user_id"].unique()
    full_target = pl.read_csv(RAW_DIR / "local_eval.csv").filter(pl.col("user_id").is_in(val_users))

    n_users = full_target["user_id"].n_unique()

    print(f"Val-юзеров с таргетом (из local_eval.csv): {n_users:,}")

    recall_als = recall_at_k(val, "als_score", full_target)
    recall_ranker = recall_at_k(val, "rank_score", full_target)
    recall_covisit = recall_at_k(val, "covisit_score", full_target)

    print(f"Recall@{K} — голый covisit:    {recall_covisit:.4f}")
    print(f"Recall@{K} — голый ALS:        {recall_als:.4f}")
    print(f"Recall@{K} — CatBoost ranker:  {recall_ranker:.4f}")
    if recall_als > 0:
        d = recall_ranker - recall_als
        print(f"Прирост от ранкера: {d:+.4f} ({d / recall_als:+.1%})")


if __name__ == "__main__":
    main()
