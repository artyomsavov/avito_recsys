import os, gc, time
import polars as pl
import pandas as pd
import numpy as np

tic = time.time()

# --- НАСТРОЙКИ ПУТЕЙ ---
DATA_DIR = "../data/"
RAW_DIR = f"{DATA_DIR}raw/"
SUBMITS_DIR = f"{DATA_DIR}submits/"

os.makedirs(SUBMITS_DIR, exist_ok=True)

print("Загрузка пользователей...")
users = pl.read_csv(f"{RAW_DIR}eval_users.csv").select(pl.col("user_id").cast(pl.UInt32))

print("Загрузка истории...")
history = (
    pl.read_parquet(f"{RAW_DIR}eval_user_events.pq", columns=["user_id", "item_id"])
    .with_columns([
        pl.col("user_id").cast(pl.UInt32),
        pl.col("item_id").cast(pl.UInt32)
    ])
)

print("Построение локального графа ковизитов...")
sample = pl.read_parquet(f"{RAW_DIR}train_data/part_000.parquet", columns=["user_id", "item_id", "timestamp"]).head(1_500_000)

pop_items = sample.group_by("item_id").agg(pl.len().alias("c")).sort("c", descending=True).head(200).get_column("item_id").to_list()

covis = sample.sort(["user_id", "timestamp"]).select([pl.col("item_id").alias("A"), pl.col("item_id").shift(-1).over("user_id").alias("B")]).drop_nulls()
covis = covis.group_by(["A", "B"]).agg(pl.len().alias("w")).sort(["A", "w"], descending=[False, True]).group_by("A").head(15)
covis_dict = covis.group_by("A").agg(pl.col("B").alias("recs")).to_pandas().set_index("A")["recs"].to_dict()
del sample, covis; gc.collect()

print("Сборка сессий...")
history_agg = history.group_by("user_id").agg([
    pl.col("item_id").unique().alias("seen"),
    pl.col("item_id").tail(5).alias("last_seen")
])

final_df = users.join(history_agg, on="user_id", how="left").to_pandas()

print("Сборка сабмита в памяти...")
res_users = []
res_items = []

for row in final_df.itertuples():
    uid = row.user_id
    seen = set(row.seen) if isinstance(row.seen, (list, np.ndarray)) else set()
    last = row.last_seen if isinstance(row.last_seen, (list, np.ndarray)) else []

    recs = []
    for item in reversed(last):
        recs.extend(covis_dict.get(item, []))

    clean_recs = []
    for r in recs + pop_items:
        if r not in seen and r not in clean_recs:
            clean_recs.append(r)
        if len(clean_recs) == 160:
            break

    res_users.extend([uid] * 160)
    res_items.extend(clean_recs[:160])

print("Запись в файл...")
pd.DataFrame({"user_id": res_users, "item_id": res_items}).to_csv(f"{SUBMITS_DIR}submission.csv", index=False)

print(f"ГОТОВО ЗА {time.time() - tic:.1f} сек!")
