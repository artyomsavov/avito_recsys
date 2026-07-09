import os

os.environ["POLARS_MAX_THREADS"] = "4"

from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

CUTOFF_MS = 1776211200000
DAYS_28_MS = 28 * 24 * 60 * 60 * 1000
START_MS = CUTOFF_MS - DAYS_28_MS

DEPTH = 3  # сколько следующих событий связываем с текущим
MAX_GAP_MS = 24 * 60 * 60 * 1000  # переход засчитывается только если dt <= 24ч
TOP_N_NEIGHBORS = 50  # хранить top-50 соседей на item
MIN_PAIR_USERS = 3  # пара должна встретиться у >= 3 разных юзеров

CHUNK_DIR = INTERIM_DIR / "covisit_chunks"


def build_pairs_for_partition(part_path: Path, out_path: Path) -> None:
    """
    Читаем одну партицию, генерируем симметричные пары (item_a < item_b)
    из DEPTH соседних событий в пределах MAX_GAP_MS, агрегируем локально.
    Пишем частичный счётчик пар: (item_a, item_b, pair_weight, user_id) — user_id
    нужен, чтобы потом считать число УНИКАЛЬНЫХ юзеров на пару (для MIN_PAIR_USERS).
    """
    df = (
        pl.scan_parquet(part_path)
        .filter(
            (pl.col("timestamp") >= START_MS)
            & (pl.col("timestamp") < CUTOFF_MS)
            & (pl.col("eid") != 7)
        )
        .select(["user_id", "item_id", "timestamp"])
        .sort(["user_id", "timestamp"])
        .collect(engine="streaming")
    )

    pair_frames = []
    for k in range(1, DEPTH + 1):
        shifted = df.with_columns(
            [
                pl.col("item_id").shift(-k).over("user_id").alias("item_next"),
                pl.col("timestamp").shift(-k).over("user_id").alias("ts_next"),
                pl.col("user_id").shift(-k).over("user_id").alias("user_next"),
            ]
        )
        valid = shifted.filter(
            (pl.col("user_next") == pl.col("user_id"))  # не вышли за границу юзера
            & (pl.col("item_next") != pl.col("item_id"))  # не self-loop
            & ((pl.col("ts_next") - pl.col("timestamp")) <= MAX_GAP_MS)
        )
        # canonical order: item_a < item_b (симметрично)
        canon = valid.select(
            [
                "user_id",
                pl.min_horizontal("item_id", "item_next").alias("item_a"),
                pl.max_horizontal("item_id", "item_next").alias("item_b"),
            ]
        )
        pair_frames.append(canon)

    if not pair_frames:
        return
    all_pairs = pl.concat(pair_frames)
    # локальная дедупликация: одна пара на юзера считается один раз (для честного user-count)
    dedup = all_pairs.unique(subset=["user_id", "item_a", "item_b"])
    dedup.write_parquet(out_path)


def build_covisitation() -> None:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for p in CHUNK_DIR.glob("*.parquet"):
        p.unlink()

    parts = sorted((RAW_DIR / "train_data").glob("part_*.parquet"))
    print(f"1. Генерируем пары по {len(parts)} партициям...")
    import time

    t0 = time.time()
    for i, part in enumerate(parts):
        build_pairs_for_partition(part, CHUNK_DIR / f"pairs_{i:03d}.parquet")
        if (i + 1) % 10 == 0 or i == len(parts) - 1:
            el = time.time() - t0
            print(
                f"   {i + 1}/{len(parts)} | {el / (i + 1):.1f} сек/партиция | ~{el / (i + 1) * (len(parts) - i - 1) / 60:.0f} мин осталось"
            )

    print("2. Финальная агрегация через DuckDB (счётчик уник. юзеров на пару, обрезка top-N)...")
    con = duckdb.connect()
    con.execute("SET memory_limit='7GB';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{(DATA_DIR / 'duckdb_tmp').as_posix()}';")

    out_path = PROCESSED_DIR / "covisit_matrix.parquet"
    con.execute(f"""
        COPY (
            WITH pair_counts AS (
                SELECT item_a, item_b, COUNT(DISTINCT user_id) AS n_users
                FROM read_parquet('{(CHUNK_DIR / "pairs_*.parquet").as_posix()}')
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
                       ROW_NUMBER() OVER (PARTITION BY src ORDER BY weight DESC) AS rn
                FROM directed
            )
            SELECT src AS item_id, dst AS neighbor_id, weight, rn AS neighbor_rank
            FROM ranked
            WHERE rn <= {TOP_N_NEIGHBORS}
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_items = con.execute(
        f"SELECT COUNT(DISTINCT item_id) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()[0]
    print(f"Готово! {n:,} связей, {n_items:,} item с соседями -> {out_path}")
    con.close()


if __name__ == "__main__":
    build_covisitation()
