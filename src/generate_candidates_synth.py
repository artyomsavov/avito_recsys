import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
from pathlib import Path

import numpy as np
import polars as pl
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

N_CANDIDATES = 400
FACTORS = 32
BATCH_SIZE = 15
FLUSH_EVERY = 200
CHUNK_DIR = INTERIM_DIR / "candidates_synth_chunks"


def generate_candidates_synth() -> None:
    print("1. Загружаем маппинги, пул, модель...")
    train_user_ids = np.load(PROCESSED_DIR / "mappings.npz")["user_ids"]
    train_item_ids = np.load(PROCESSED_DIR / "mappings.npz")["item_ids"]
    n_train_items = len(train_item_ids)
    allowed_item_idx = np.load(PROCESSED_DIR / "allowed_item_pool.npz")["allowed_item_idx"]

    item_factors = np.load(MODELS_DIR / "item_factors_extracted.npy", mmap_mode="r+")
    user_factors = np.load(MODELS_DIR / "als_model.npz")["user_factors"].astype(np.float32)

    model = AlternatingLeastSquares(factors=FACTORS)
    model.item_factors = item_factors
    model.user_factors = user_factors

    print("2. Маппим synth-юзеров...")
    synth_users = pl.read_csv(RAW_DIR / "local_eval.csv")["user_id"].unique().to_numpy()
    user_idx = np.searchsorted(train_user_ids, synth_users)
    valid = (user_idx < len(train_user_ids)) & (
        train_user_ids[np.clip(user_idx, 0, len(train_user_ids) - 1)] == synth_users
    )
    assert valid.all()
    user_idx = user_idx[valid]
    synth_users = synth_users[valid]

    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for p in CHUNK_DIR.glob("chunk_*.parquet"):
        p.unlink()

    empty_history = csr_matrix((len(synth_users), n_train_items), dtype=np.float32)

    n = len(synth_users)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"3. Генерация: {n:,} юзеров, {n_batches:,} батчей...")

    rows = []
    chunk_i = 0
    t0 = time.time()

    for b in range(n_batches):
        s = b * BATCH_SIZE
        e = min(s + BATCH_SIZE, n)
        ids_batch, scores_batch = model.recommend(
            user_idx[s:e],
            empty_history[s:e],
            N=N_CANDIDATES,
            filter_already_liked_items=False,
            items=allowed_item_idx,
        )
        for row_i in range(e - s):
            uid = int(synth_users[s + row_i])
            for rank, (it_idx, score) in enumerate(zip(ids_batch[row_i], scores_batch[row_i])):
                if it_idx < 0:
                    continue
                rows.append((uid, int(train_item_ids[it_idx]), float(score), rank, "als"))

        # периодический сброс на диск — rows не растёт бесконечно
        if (b + 1) % FLUSH_EVERY == 0 or b == n_batches - 1:
            df = pl.DataFrame(
                rows, schema=["user_id", "item_id", "als_score", "als_rank", "source"], orient="row"
            )
            df.write_parquet(CHUNK_DIR / f"chunk_{chunk_i:04d}.parquet")
            chunk_i += 1
            rows = []
            el = time.time() - t0
            print(
                f"   батч {b + 1}/{n_batches} | {el / (b + 1):.2f} сек/батч | ~{el / (b + 1) * (n_batches - b - 1) / 60:.0f} мин осталось | чанк {chunk_i}"
            )

    print("4. Склеиваем чанки...")
    out_path = INTERIM_DIR / "candidates_synth.parquet"
    pl.scan_parquet(CHUNK_DIR / "chunk_*.parquet").sink_parquet(out_path)
    total = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    users = pl.scan_parquet(out_path).select(pl.col("user_id").n_unique()).collect().item()
    print(f"Готово! {total:,} строк, {users:,} юзеров -> {out_path}")


if __name__ == "__main__":
    generate_candidates_synth()
