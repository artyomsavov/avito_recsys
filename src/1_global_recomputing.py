import os
import glob
import polars as pl
import scipy.sparse as sp
import implicit
import numpy as np
import gc
import pickle
from datetime import datetime, timezone
from tqdm import tqdm

# --- НАСТРОЙКИ ПУТЕЙ ---
DATA_DIR = "../data/"
RAW_DIR = f"{DATA_DIR}raw/"
INTERIM_DIR = f"{DATA_DIR}interim/"
MODELS_DIR = f"{DATA_DIR}models/"

for d in [INTERIM_DIR, MODELS_DIR]: os.makedirs(d, exist_ok=True)

cutoff_dt = datetime(2026, 4, 15, 0, 0, 0, tzinfo=timezone.utc)
CUTOFF_DATE = int(cutoff_dt.timestamp() * 1000)
start_dt = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
START_DATE = int(start_dt.timestamp() * 1000)

eval_file = f"{RAW_DIR}eval_user_events.pq"
train_files = sorted(glob.glob(f"{RAW_DIR}train_data/part_*.parquet"))[::5]
all_paths = [eval_file] + train_files

print("Загрузка справочника контактов...")
contact_eids = pl.read_csv(f"{RAW_DIR}contact_eids.csv").get_column("mapped_eid").to_list()

print("\n=== ПАСС 1: Сборка словарей ===")
global_item_counts = pl.DataFrame(schema={"item_id": pl.UInt32, "cnt": pl.UInt32})
unique_users = pl.DataFrame(schema={"user_id": pl.UInt32})

for path in tqdm(all_paths, desc="Сканирование файлов"):
    if "eval" in path:
        lazy_df = pl.scan_parquet(path).filter(pl.col("timestamp") < CUTOFF_DATE)
    else:
        lazy_df = pl.scan_parquet(path).filter((pl.col("timestamp") >= START_DATE) & (pl.col("timestamp") < CUTOFF_DATE))
        
    local_items = lazy_df.group_by("item_id").agg(pl.len().alias("cnt")).collect()
    global_item_counts = pl.concat([global_item_counts, local_items]).group_by("item_id").agg(pl.col("cnt").sum())
    
    local_users = lazy_df.select("user_id").unique().collect()
    unique_users = pl.concat([unique_users, local_users]).unique()

valid_items = global_item_counts.filter(pl.col("cnt") >= 5).select("item_id")
global_item_mapping = valid_items.unique().with_row_index("item_idx").with_columns(pl.col("item_idx").cast(pl.UInt32))
global_user_mapping = unique_users.with_row_index("user_idx").with_columns(pl.col("user_idx").cast(pl.UInt32))

del global_item_counts, valid_items, unique_users; gc.collect()

print("\n=== ПАСС 2: Проекция матрицы ===")
rows_list, cols_list, data_list = [], [], []

for path in tqdm(all_paths, desc="Проекция в матрицу"):
    if "eval" in path:
        lazy_df = pl.scan_parquet(path).filter(pl.col("timestamp") < CUTOFF_DATE)
    else:
        lazy_df = pl.scan_parquet(path).filter((pl.col("timestamp") >= START_DATE) & (pl.col("timestamp") < CUTOFF_DATE))
        
    chunk = (
        lazy_df
        .with_columns(pl.when(pl.col("eid").is_in(contact_eids)).then(10).otherwise(1).cast(pl.Float32).alias("weight"))
        .group_by(["user_id", "item_id"]).agg(pl.col("weight").sum())
        .join(global_item_mapping.lazy(), on="item_id", how="inner")
        .join(global_user_mapping.lazy(), on="user_id", how="inner")
        .collect()
    )
    
    if chunk.height > 0:
        rows_list.append(chunk["user_idx"].to_numpy())
        cols_list.append(chunk["item_idx"].to_numpy())
        data_list.append(chunk["weight"].to_numpy())
    del chunk; gc.collect()

user_item_csr = sp.csr_matrix((np.concatenate(data_list), (np.concatenate(rows_list), np.concatenate(cols_list))), 
                              shape=(global_user_mapping.height, global_item_mapping.height))
del rows_list, cols_list, data_list; gc.collect()

print("\n=== ОБУЧЕНИЕ (CPU - 3 ядра) ===")
als_model = implicit.als.AlternatingLeastSquares(factors=64, iterations=5, regularization=0.01, 
                                                 random_state=42, use_gpu=False, num_threads=3)
als_model.fit(user_item_csr)

print("\n=== СОХРАНЕНИЕ АРТЕФАКТОВ ===")
global_user_mapping.write_parquet(f"{INTERIM_DIR}als_user_mapping.parquet")
global_item_mapping.write_parquet(f"{INTERIM_DIR}als_item_mapping.parquet")
with open(f"{MODELS_DIR}als_model.pkl", "wb") as f: pickle.dump(als_model, f)
sp.save_npz(f"{INTERIM_DIR}user_item_csr.npz", user_item_csr)
print("Готово!")
