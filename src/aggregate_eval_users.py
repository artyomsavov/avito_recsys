from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_MS = 1776211200000
DAYS_28_MS = 28 * 24 * 60 * 60 * 1000
START_MS = CUTOFF_MS - DAYS_28_MS
MAX_ITEMS_PER_USER = 100
WEIGHT_CLICK = 1
WEIGHT_CONTACT = 5


def aggregate_eval_user_interactions() -> None:
    print("1. Читаем список целевых контактов...")
    contact_df = pl.read_csv(RAW_DIR / "contact_eids.csv")
    contact_eids = contact_df["mapped_eid"].to_list()

    print("2. Сканируем eval_user_events.pq...")
    eval_events_path = RAW_DIR / "eval_user_events.pq"
    if not eval_events_path.exists():
        raise FileNotFoundError(
            f"{eval_events_path} не найден. Скачай и распакуй eval_user_events.zip:\n"
            "curl -O https://storage.yandexcloud.net/datafest2026/datafest_2026_v2_v4/eval_user_events.zip"
        )

    lf = pl.scan_parquet(eval_events_path)

    # Тот же фильтр по времени и дропу показов, что и в train-агрегации
    lf = lf.filter(
        (pl.col("timestamp") >= START_MS) & (pl.col("timestamp") < CUTOFF_MS) & (pl.col("eid") != 7)
    )

    lf = lf.with_columns(
        pl.when(pl.col("eid").is_in(contact_eids))
        .then(WEIGHT_CONTACT)
        .otherwise(WEIGHT_CLICK)
        .cast(pl.UInt8)
        .alias("event_weight")
    )

    agg_lf = lf.group_by(["user_id", "item_id"]).agg(
        [
            pl.col("event_weight").sum().cast(pl.UInt16).alias("weight"),
            pl.col("timestamp").max().alias("last_timestamp"),
        ]
    )

    print("3. Запускаем потоковую обработку...")
    df_agg = agg_lf.collect()
    print(f"   Получено {len(df_agg):,} уникальных пар (eval_user, item).")

    print(f"4. Оставляем топ-{MAX_ITEMS_PER_USER} свежих действий на юзера...")
    df_final = (
        df_agg.sort(["user_id", "last_timestamp"], descending=[False, True])
        .group_by("user_id")
        .head(MAX_ITEMS_PER_USER)
    )

    out_path = INTERIM_DIR / "eval_interactions_agg.parquet"
    df_final.write_parquet(out_path)
    print(f"Готово! {len(df_final):,} строк сохранено в {out_path}")


if __name__ == "__main__":
    aggregate_eval_user_interactions()
