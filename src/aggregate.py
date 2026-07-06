from pathlib import Path

import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

# --- Конфигурация логики (из EDA) ---
CUTOFF_MS = 1776211200000
DAYS_28_MS = 28 * 24 * 60 * 60 * 1000
START_MS = CUTOFF_MS - DAYS_28_MS

MAX_ITEMS_PER_USER = 100
WEIGHT_CLICK = 1
WEIGHT_CONTACT = 5


def aggregate_interactions() -> None:
    print("1. Читаем список целевых контактов...")
    contact_df = pl.read_csv(RAW_DIR / "contact_eids.csv")
    contact_eids = contact_df["mapped_eid"].to_list()

    print("2. Сканируем партиции и строим план запроса...")
    lf = pl.scan_parquet(RAW_DIR / "train_data" / "part_*.parquet")

    # Жесткий фильтр: отрезаем по времени и дропаем показы (eid=7)
    lf = lf.filter(
        (pl.col("timestamp") >= START_MS)
        & (pl.col("timestamp") < CUTOFF_MS)
        & (pl.col("eid") != 7)
    )

    # Назначаем веса: если контакт -> 5, если клик/избранное -> 1
    lf = lf.with_columns(
        pl.when(pl.col("eid").is_in(contact_eids))
        .then(WEIGHT_CONTACT)
        .otherwise(WEIGHT_CLICK)
        .cast(pl.UInt8)
        .alias("event_weight")
    )

    # Агрегируем взаимодействия пользователя с конкретным объявлением
    agg_lf = lf.group_by(["user_id", "item_id"]).agg(
        [
            pl.col("event_weight").sum().cast(pl.UInt16).alias("weight"),
            pl.col("timestamp").max().alias("last_timestamp"),
        ]
    )

    print("3. Запускаем потоковую обработку (это займет несколько минут)...")
    df_agg = agg_lf.collect(engine="streaming") # Прогоняем батчами

    print(f"   Сжатие завершено. Получено {len(df_agg):,} уникальных пар user-item.")

    print("4. Применяем лимит: оставляем топ-100 свежих действий на пользователя...")
    # Так как df_agg уже компактный, оконная функция легко отработает в памяти
    df_final = (
        df_agg.sort(["user_id", "last_timestamp"], descending=[False, True])
        .group_by("user_id")
        .head(MAX_ITEMS_PER_USER)
    )

    out_path = INTERIM_DIR / "interactions_agg.parquet"
    df_final.write_parquet(out_path)

    print(f"Готово! Финальный размер: {len(df_final):,} строк.")
    print(f"Файл сохранен в: {out_path}")


if __name__ == "__main__":
    aggregate_interactions()
