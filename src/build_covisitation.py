"""Item-item co-visitation матрица по соседним событиям юзеров за окно до cutoff."""

import os

os.environ["POLARS_MAX_THREADS"] = "4"

import shutil
import time
from pathlib import Path

import duckdb
import polars as pl

from config import TMP_DIR, TRAIN_PARTS, Run, parse_run, window_filter

DEPTH = 3  # сколько следующих событий связываем с текущим
MAX_GAP_MS = 24 * 60 * 60 * 1000  # переход засчитывается только если dt <= 24ч
TOP_N_NEIGHBORS = 50  # хранить top-50 соседей на item
MIN_PAIR_USERS = 3  # пара должна встретиться у >= 3 разных юзеров


def build_pairs_for_partition(part_path: Path, out_path: Path, run: Run) -> None:
    """
    Читаем одну партицию, генерируем симметричные пары (item_a < item_b)
    из DEPTH соседних событий в пределах MAX_GAP_MS.
    Пишем (user_id, item_a, item_b) — user_id нужен, чтобы потом считать число
    УНИКАЛЬНЫХ юзеров на пару (для MIN_PAIR_USERS).
    """
    df = (
        pl.scan_parquet(part_path)
        .filter(window_filter(run))
        .select(["user_id", "item_id", "timestamp"])
        .sort(["user_id", "timestamp"])
        .collect(engine="streaming")
    )

    pair_frames = []
    for k in range(1, DEPTH + 1):
        shifted = df.with_columns(
            pl.col("item_id").shift(-k).over("user_id").alias("item_next"),
            pl.col("timestamp").shift(-k).over("user_id").alias("ts_next"),
        )
        valid = shifted.filter(
            pl.col("item_next").is_not_null()  # не вышли за границу юзера
            & (pl.col("item_next") != pl.col("item_id"))  # не self-loop
            & ((pl.col("ts_next") - pl.col("timestamp")) <= MAX_GAP_MS)
        )
        # canonical order: item_a < item_b (симметрично)
        pair_frames.append(
            valid.select(
                "user_id",
                pl.min_horizontal("item_id", "item_next").alias("item_a"),
                pl.max_horizontal("item_id", "item_next").alias("item_b"),
            )
        )
    # одна пара на юзера считается один раз (для честного user-count)
    pl.concat(pair_frames).unique().write_parquet(out_path)


def build_covisitation() -> None:
    run = parse_run(__doc__)
    chunk_dir = run.interim / "covisit_chunks"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True)

    parts = sorted(TRAIN_PARTS.parent.glob(TRAIN_PARTS.name))
    print(f"1. Генерируем пары по {len(parts)} партициям...")
    t0 = time.time()
    for i, part in enumerate(parts):
        build_pairs_for_partition(part, chunk_dir / f"pairs_{i:03d}.parquet", run)
        if (i + 1) % 10 == 0 or i == len(parts) - 1:
            el = time.time() - t0
            left = el / (i + 1) * (len(parts) - i - 1) / 60
            print(f"   {i + 1}/{len(parts)} | {el / (i + 1):.1f} с/партиция, ~{left:.0f} мин")

    print("2. Финальная агрегация через DuckDB (уник. юзеры на пару, обрезка top-N)...")
    con = duckdb.connect()
    con.execute("SET memory_limit='7GB';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")

    out_path = run.processed / "covisit_matrix.parquet"
    con.execute(f"""
        COPY (
            WITH pair_counts AS (
                SELECT item_a, item_b, COUNT(DISTINCT user_id) AS n_users
                FROM read_parquet('{(chunk_dir / "pairs_*.parquet").as_posix()}')
                GROUP BY item_a, item_b
                HAVING COUNT(DISTINCT user_id) >= {MIN_PAIR_USERS}
            ),
            -- симметризуем: каждая пара даёт связь в обе стороны (a->b и b->a),
            -- чтобы для любой "затравки" можно было найти соседей
            directed AS (
                SELECT item_a AS src, item_b AS dst, n_users AS weight FROM pair_counts
                UNION ALL
                SELECT item_b AS src, item_a AS dst, n_users AS weight FROM pair_counts
            ),
            ranked AS (
                SELECT src, dst, weight,
                       ROW_NUMBER() OVER (PARTITION BY src ORDER BY weight DESC, dst) AS rn
                FROM directed
            )
            SELECT src AS item_id, dst AS neighbor_id, weight, rn AS neighbor_rank
            FROM ranked
            WHERE rn <= {TOP_N_NEIGHBORS}
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n, n_items = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT item_id) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()
    print(f"Готово! {n:,} связей, {n_items:,} item с соседями -> {out_path}")
    con.close()
    shutil.rmtree(chunk_dir)  # ~3 GB промежуточных пар


if __name__ == "__main__":
    build_covisitation()
