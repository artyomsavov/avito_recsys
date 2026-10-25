"""ALS-кандидаты (top-400 из пула) для synth-юзеров по их обученным user_factors."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "4"

import shutil
import time

import numpy as np
import polars as pl

from config import ids_to_idx, parse_run, target_users

N_CANDIDATES = 400
BATCH_SIZE = 15  # скоры батча: 15 x 16.5М float32 ~ 1 GB
FLUSH_EVERY = 100


def top_n(scores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-n по строкам: (индексы, скоры) по убыванию скора. Построчно — argpartition
    по всей матрице батча держал бы int64-индексы на 15 x 16.5М (~2 GB)."""
    ids = np.empty((len(scores), n), dtype=np.int64)
    for i, row in enumerate(scores):
        part = np.argpartition(row, len(row) - n)[-n:]
        ids[i] = part[np.argsort(-row[part], kind="stable")]
    return ids, np.take_along_axis(scores, ids, axis=1)


def generate_candidates_als() -> None:
    run = parse_run(__doc__)
    print("1. Загружаем маппинги, пул, факторы...")
    mappings = np.load(run.processed / "mappings.npz")
    train_user_ids, train_item_ids = mappings["user_ids"], mappings["item_ids"]
    allowed_item_idx = np.load(run.processed / "allowed_item_pool.npz")["allowed_item_idx"]

    # mmap: item_factors 4.4 GB; срез по пулу один раз (~2 GB) — скоринг только по пулу
    item_factors = np.load(run.models / "item_factors.npy", mmap_mode="r")
    pool_factors = np.ascontiguousarray(item_factors[allowed_item_idx])
    pool_item_ids = train_item_ids[allowed_item_idx]

    users = target_users()
    user_idx, valid = ids_to_idx(train_user_ids, users)
    print(f"   synth-юзеров с историей до cutoff: {valid.sum():,} из {len(users):,}")
    users = users[valid]
    user_factors = np.load(run.models / "user_factors.npy", mmap_mode="r")
    user_vectors = np.asarray(user_factors[user_idx[valid]], dtype=np.float32)

    chunk_dir = run.interim / "candidates_als_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir()

    n = len(users)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"2. Генерация top-{N_CANDIDATES}: {n:,} юзеров, {n_batches:,} батчей...")
    frames: list[pl.DataFrame] = []
    chunk_i = 0
    t0 = time.time()
    for b in range(n_batches):
        s, e = b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, n)
        ids, scores = top_n(user_vectors[s:e] @ pool_factors.T, N_CANDIDATES)
        frames.append(
            pl.DataFrame(
                {
                    "user_id": np.repeat(users[s:e], N_CANDIDATES),
                    "item_id": pool_item_ids[ids.ravel()],
                    "als_score": scores.ravel().astype(np.float64),
                    "als_rank": np.tile(np.arange(N_CANDIDATES), e - s),
                }
            )
        )

        # периодический сброс на диск — frames не растёт бесконечно
        if (b + 1) % FLUSH_EVERY == 0 or b == n_batches - 1:
            pl.concat(frames).write_parquet(chunk_dir / f"chunk_{chunk_i:05d}.parquet")
            chunk_i += 1
            frames = []
            el = time.time() - t0
            left = el / (b + 1) * (n_batches - b - 1) / 60
            print(f"   батч {b + 1}/{n_batches} | {el / (b + 1):.2f} с/батч, ~{left:.0f} мин")

    print("3. Склеиваем чанки...")
    out_path = run.interim / "candidates_als.parquet"
    pl.scan_parquet(chunk_dir / "chunk_*.parquet").sink_parquet(out_path)
    shutil.rmtree(chunk_dir)
    total, n_users = (
        pl.scan_parquet(out_path).select(pl.len(), pl.col("user_id").n_unique()).collect().row(0)
    )
    print(f"Готово! {total:,} строк, {n_users:,} юзеров -> {out_path}")


if __name__ == "__main__":
    generate_candidates_als()
