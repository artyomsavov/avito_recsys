import polars as pl
import os
import gc
import subprocess
from datetime import datetime, timezone

def kill_zombie_processes():
    try:
        pid = os.getpid()
        cmd = f"ps aux | grep 'python' | grep 'generate_features.py' | grep -v '{pid}' | awk '{{print $2}}' | xargs kill -9"
        subprocess.run(cmd, shell=True, capture_output=True)
    except: pass
kill_zombie_processes()

MAX_USERS = 500         
CUTOFF_DATE = 1775558400000 
GAP_END_DATE = CUTOFF_DATE + (12 * 60 * 60 * 1000)

# --- НАСТРОЙКИ ПУТЕЙ ---
DATA_DIR = "../data/"
RAW_DIR = f"{DATA_DIR}raw/"
PROCESSED_DIR = f"{DATA_DIR}processed/train_matrices/"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.environ['POLARS_MAX_THREADS'] = '2'

print("Генерация фичей...")

contact_eids_df = pl.read_csv(f"{RAW_DIR}contact_eids.csv").select(pl.col("mapped_eid").alias("eid"))
item_features = pl.scan_parquet(f"{RAW_DIR}item_features.parquet").select(
    ["item_id", "vertical_id", "category_ext_y", "region_id_y"]
).collect()

print("Сборка графа Co-visitation...")
train_sample = pl.read_parquet(f"{RAW_DIR}train_data/part_000.parquet").head(1_000_000).filter(pl.col("timestamp") < CUTOFF_DATE)

global_popular = (
    train_sample.group_by("item_id").agg(pl.len().alias("pop"))
    .sort("pop", descending=True).head(100)
    .select([pl.col("item_id").cast(pl.UInt32), pl.lit(1).cast(pl.Int8).alias("source_pop")])
)

co_visitation = (
    train_sample.sort(["user_id", "timestamp"])
    .select([pl.col("item_id").alias("item_A"), pl.col("item_id").shift(-1).over("user_id").alias("item_B")])
    .drop_nulls()
    .group_by(["item_A", "item_B"]).agg(pl.len().alias("weight"))
    .sort(["item_A", "weight"], descending=[False, True])
    .group_by("item_A").head(10)
)

del train_sample
gc.collect()

for part in range(20): 
    file_path = f"{RAW_DIR}train_data/part_{part:03d}.parquet"
    if not os.path.exists(file_path): continue
    
    print(f"Обработка партиции {part:03d} из 20...")
    
    lazy_df = pl.scan_parquet(file_path)
    eval_users = lazy_df.filter(pl.col("timestamp") >= GAP_END_DATE).select("user_id").unique().head(MAX_USERS).collect()["user_id"].to_list()
    
    if not eval_users: continue
    
    df_batch = lazy_df.filter(pl.col("user_id").is_in(eval_users)).collect()
    
    hist = df_batch.filter(pl.col("timestamp") < CUTOFF_DATE)
    contacts = df_batch.filter(pl.col("timestamp") >= GAP_END_DATE).join(contact_eids_df, on="eid", how="inner")
    
    targets = (
        contacts.join(hist.select(["user_id", "item_id"]).unique(), on=["user_id", "item_id"], how="anti")
        .unique(subset=["user_id", "item_id"])
        .with_columns(pl.lit(1).cast(pl.Int8).alias("target"))
        .select([pl.col("user_id").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32), "target"])
    )
    
    users_df = pl.DataFrame({"user_id": eval_users}, schema={"user_id": pl.UInt32})
    
    cand_pop = users_df.join(global_popular, how="cross").with_columns([pl.col("user_id").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)])
    cand_covis = (
        hist.sort(["user_id", "timestamp"], descending=[False, True])
        .group_by("user_id").head(5)
        .join(co_visitation, left_on="item_id", right_on="item_A", how="inner")
        .select([pl.col("user_id").cast(pl.UInt32), pl.col("item_B").alias("item_id").cast(pl.UInt32), pl.lit(1).cast(pl.Int8).alias("source_covis")])
    )
    
    super_pool = pl.concat([cand_pop, cand_covis], how="diagonal").fill_null(0).group_by(["user_id", "item_id"]).agg([
        pl.col("source_pop").max(), pl.col("source_covis").max()
    ])
    
    final_pool = (
        super_pool.join(hist.select([pl.col("user_id").cast(pl.UInt32), pl.col("item_id").cast(pl.UInt32)]).unique(), on=["user_id", "item_id"], how="anti")
        .with_columns((pl.col("source_covis") * 5 + pl.col("source_pop") * 1).alias("score"))
        .sort(["user_id", "score"], descending=[False, True])
        .group_by("user_id").head(150)
    )

    user_feats = hist.join(item_features, on="item_id", how="left").group_by("user_id").agg([
        pl.col("vertical_id").mode().first().alias("u_top_vertical"),
        pl.col("category_ext_y").mode().first().alias("u_top_category"),
        pl.col("region_id_y").mode().first().alias("u_top_region")
    ]).with_columns(pl.col("user_id").cast(pl.UInt32))
    
    X = (
        final_pool.join(item_features.with_columns(pl.col("item_id").cast(pl.UInt32)), on="item_id", how="left")
        .join(user_feats, on="user_id", how="left")
        .with_columns([
            (pl.col("vertical_id") == pl.col("u_top_vertical")).cast(pl.Int8).alias("is_top_vertical")
        ])
        .join(targets, on=["user_id", "item_id"], how="left").fill_null(0)
    )
    
    X.write_parquet(f"{PROCESSED_DIR}chunk_train_p{part:03d}.parquet", compression="zstd")
    
    del lazy_df, df_batch, hist, contacts, targets, users_df, cand_pop, cand_covis, super_pool, final_pool, user_feats, X
    gc.collect()

print("\nГОТОВО! Матрицы сохранены.")
