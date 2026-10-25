"""Сырой кликстрим -> агрегат (user, item, weight, last_timestamp) за окно до cutoff."""

from pathlib import Path

import polars as pl

from config import TRAIN_PARTS, Run, contact_eids, parse_run, window_filter

MAX_ITEMS_PER_USER = 100
WEIGHT_CLICK = 1
WEIGHT_CONTACT = 5


def aggregate(lf: pl.LazyFrame, run: Run, out_path: Path) -> None:
    lf = lf.filter(window_filter(run)).with_columns(
        # если контакт 5, если клик/избранное 1
        pl.when(pl.col("eid").is_in(contact_eids()))
        .then(WEIGHT_CONTACT)
        .otherwise(WEIGHT_CLICK)
        .cast(pl.UInt8)
        .alias("event_weight")
    )
    df_agg = (
        lf.group_by(["user_id", "item_id"])
        .agg(
            pl.col("event_weight").sum().cast(pl.UInt16).alias("weight"),
            pl.col("timestamp").max().alias("last_timestamp"),
        )
        .collect(engine="streaming")
    )
    print(f"   {len(df_agg):,} уникальных пар user-item")

    # топ-100 свежих действий на пользователя
    df_final = (
        df_agg.sort(["user_id", "last_timestamp"], descending=[False, True])
        .group_by("user_id")
        .head(MAX_ITEMS_PER_USER)
    )
    assert df_final["last_timestamp"].max() < run.cutoff_ms, "событие после cutoff"
    df_final.write_parquet(out_path)
    print(f"   {len(df_final):,} строк -> {out_path}")


def main() -> None:
    run = parse_run(__doc__)
    aggregate(pl.scan_parquet(TRAIN_PARTS), run, run.interim / "interactions_agg.parquet")


if __name__ == "__main__":
    main()
