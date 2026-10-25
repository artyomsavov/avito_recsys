"""Recall@160 на val-юзерах: popular, ALS, covisit, ранкер + потолок retrieval."""

import json

import polars as pl

from config import ALLOWED_VERTICALS, ITEM_FEATURES, LOCAL_EVAL, K, parse_run


def recall_at_k(preds: pl.DataFrame, score_col: str, full_target: pl.DataFrame) -> float:
    topk = (
        # item_id — детерминированный tie-break при равных скорах
        preds.sort(["user_id", score_col, "item_id"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True)
        .head(K)
        .select(["user_id", "item_id"])
    )
    hits = (
        topk.join(full_target, on=["user_id", "item_id"], how="semi")
        .group_by("user_id")
        .agg(pl.len().alias("hits"))
    )
    # знаменатель — полное число таргетов юзера из local_eval.csv
    per_user = (
        full_target.group_by("user_id")
        .agg(pl.len().alias("total_target"))
        .join(hits, on="user_id", how="left")
        .with_columns(pl.col("hits").fill_null(0))
        .with_columns((pl.col("hits") / pl.col("total_target")).alias("recall"))
    )
    return per_user["recall"].mean()


def popular_top_k(run) -> pl.DataFrame:
    """Top-K популярных item (число юзеров до cutoff) в разрешённых вертикалях."""
    items = (
        pl.scan_parquet(ITEM_FEATURES)
        .select(["item_id", "vertical_id"])
        .filter(pl.col("vertical_id").is_in(ALLOWED_VERTICALS))
    )
    return (
        pl.scan_parquet(run.interim / "interactions_agg.parquet")
        .join(items, on="item_id", how="inner")
        .group_by("item_id")
        .agg(pl.len().alias("popularity"))
        .sort("popularity", descending=True)
        .head(K)
        .collect(engine="streaming")
    )


def main() -> None:
    run = parse_run(__doc__)
    val = pl.read_parquet(run.features / "val_predictions.parquet")
    feats = pl.read_parquet(
        run.features / "features.parquet",
        columns=["user_id", "item_id", "als_score", "covisit_score", "in_als", "in_covisit"],
    )
    val = val.join(feats, on=["user_id", "item_id"], how="left")
    val_users = val.select("user_id").unique()
    full_target = pl.read_csv(LOCAL_EVAL).join(val_users, on="user_id", how="semi").unique()
    popular = val_users.join(popular_top_k(run), how="cross")

    metrics = {
        "run": run.name,
        "cutoff_ms": run.cutoff_ms,
        "n_val_users": full_target["user_id"].n_unique(),
        "n_val_targets": len(full_target),
        "recall_popular": recall_at_k(popular, "popularity", full_target),
        # голый источник: ранжируем только его собственных кандидатов
        "recall_als": recall_at_k(val.filter(pl.col("in_als") == 1), "als_score", full_target),
        "recall_covisit": recall_at_k(
            val.filter(pl.col("in_covisit") == 1), "covisit_score", full_target
        ),
        "recall_ranker": recall_at_k(val, "rank_score", full_target),
        # доля таргет-пар, попавших в пул кандидатов — потолок для ранкера
        "retrieval_ceiling": val["target"].sum() / len(full_target),
    }

    print(f"Val-юзеров с таргетом: {metrics['n_val_users']:,}")
    for key in ("recall_popular", "recall_als", "recall_covisit", "recall_ranker"):
        print(f"Recall@{K} {key.removeprefix('recall_'):10s} {metrics[key]:.4f}")
    print(f"Потолок retrieval:   {metrics['retrieval_ceiling']:.2%}")

    out = run.root / "metrics.json"
    # 6 знаков: порядок float-суммирования в polars даёт шум в ~17-м знаке
    rounded = {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()}
    out.write_text(json.dumps(rounded, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
