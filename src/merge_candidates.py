"""Union ALS ∪ covisit + popular-fallback для юзеров без кандидатов."""

import numpy as np
import polars as pl

from config import parse_run, target_users

N_FALLBACK = 400
MISSING_RANK = 999  # "не было среди топ-400 источника"


def popular_items(run, n: int) -> list[int]:
    """Топ-n разрешённых item по суммарному весу взаимодействий до cutoff."""
    train_item_ids = np.load(run.processed / "mappings.npz")["item_ids"]
    allowed_idx = np.load(run.processed / "allowed_item_pool.npz")["allowed_item_idx"]
    allowed = pl.DataFrame({"item_id": train_item_ids[allowed_idx]})
    return (
        pl.scan_parquet(run.interim / "interactions_agg.parquet")
        .join(allowed.lazy(), on="item_id", how="semi")
        .group_by("item_id")
        .agg(pl.col("weight").sum().alias("total_weight"))
        .sort("total_weight", descending=True)
        .head(n)
        .collect(engine="streaming")["item_id"]
        .to_list()
    )


def merge_candidates() -> None:
    run = parse_run(__doc__)
    als = pl.read_parquet(run.interim / "candidates_als.parquet").select(
        ["user_id", "item_id", "als_score", "als_rank"]
    )
    covisit = pl.read_parquet(run.interim / "candidates_covisit.parquet").select(
        ["user_id", "item_id", "covisit_score", "covisit_rank"]
    )

    # full outer join по (user_id, item_id) — кандидат может быть у одного источника,
    # у другого, или у обоих сразу (тогда получит обе пары фичей)
    merged = als.join(covisit, on=["user_id", "item_id"], how="full", coalesce=True)
    merged = merged.with_columns(
        pl.col("als_score").fill_null(0.0),
        pl.col("als_rank").fill_null(MISSING_RANK),
        pl.col("covisit_score").fill_null(0.0),
        pl.col("covisit_rank").fill_null(MISSING_RANK),
        pl.col("als_rank").is_not_null().alias("in_als"),
        pl.col("covisit_rank").is_not_null().alias("in_covisit"),
        pl.lit(False).alias("is_fallback"),
    )

    covered = merged["user_id"].unique().to_numpy()
    cold_users = np.setdiff1d(target_users(), covered)
    print(f"   Холодных юзеров (popular-fallback): {len(cold_users):,}")
    if len(cold_users):
        fallback = (
            pl.DataFrame({"user_id": cold_users.astype(np.uint32)})
            .join(pl.DataFrame({"item_id": popular_items(run, N_FALLBACK)}), how="cross")
            .with_columns(
                pl.lit(0.0).alias("als_score"),
                pl.lit(MISSING_RANK).alias("als_rank"),
                pl.lit(0.0).alias("covisit_score"),
                pl.lit(MISSING_RANK).alias("covisit_rank"),
                pl.lit(False).alias("in_als"),
                pl.lit(False).alias("in_covisit"),
                pl.lit(True).alias("is_fallback"),
            )
        )
        merged = pl.concat([merged, fallback], how="vertical_relaxed")

    out_path = run.interim / "candidates_merged.parquet"
    merged = merged.sort(["user_id", "item_id"])  # full join не гарантирует порядок
    merged.write_parquet(out_path)
    print(f"Готово! {len(merged):,} строк, {merged['user_id'].n_unique():,} юзеров -> {out_path}")
    print(f"   Только ALS: {merged.filter(pl.col('in_als') & ~pl.col('in_covisit')).height:,}")
    print(f"   Только covisit: {merged.filter(~pl.col('in_als') & pl.col('in_covisit')).height:,}")
    print(f"   В обоих: {merged.filter(pl.col('in_als') & pl.col('in_covisit')).height:,}")


if __name__ == "__main__":
    merge_candidates()
