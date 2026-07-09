# добавить в src/generate_candidates_covisit.py или отдельным скриптом src/merge_candidates.py
from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"


def merge_candidates() -> None:
    als = pl.read_parquet(INTERIM_DIR / "candidates_synth.parquet").select(
        ["user_id", "item_id", "als_score", "als_rank"]
    )
    covisit = pl.read_parquet(INTERIM_DIR / "candidates_covisit_synth.parquet").select(
        ["user_id", "item_id", "covisit_score", "covisit_rank"]
    )

    # full outer join по (user_id, item_id) — кандидат может быть у одного источника,
    # у другого, или у обоих сразу (тогда получит обе пары фичей)
    merged = als.join(covisit, on=["user_id", "item_id"], how="full", coalesce=True)

    merged = merged.with_columns(
        [
            pl.col("als_score").fill_null(0.0),
            pl.col("als_rank").fill_null(999),  # 999 = "не было среди ALS-топ-400"
            pl.col("covisit_score").fill_null(0.0),
            pl.col("covisit_rank").fill_null(999),
            pl.col("als_rank").is_not_null().alias("in_als"),
            pl.col("covisit_rank").is_not_null().alias("in_covisit"),
        ]
    )

    out_path = INTERIM_DIR / "candidates_merged_synth.parquet"
    merged.write_parquet(out_path)
    print(f"Готово! {len(merged):,} строк, {merged['user_id'].n_unique():,} юзеров -> {out_path}")
    print(f"   Только ALS: {merged.filter(pl.col('in_als') & ~pl.col('in_covisit')).height:,}")
    print(f"   Только covisit: {merged.filter(~pl.col('in_als') & pl.col('in_covisit')).height:,}")
    print(f"   В обоих: {merged.filter(pl.col('in_als') & pl.col('in_covisit')).height:,}")


if __name__ == "__main__":
    merge_candidates()
