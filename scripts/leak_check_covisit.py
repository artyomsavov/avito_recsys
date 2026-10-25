# ruff: noqa: E501
"""Честный замер covisit + popular: всё строится строго на событиях < THR (порог synth-таргета).
Повторяет логику src/build_covisitation.py, build_item_pool.py, generate_candidates_covisit.py,
eval_popular_baseline.py, меняется только временное окно. Запуск из корня: python scripts/leak_check_covisit.py <out_dir>."""

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

RAW = Path("data/raw")
LEAKY = Path("data/runs/leaky")  # артефакты старого пайплайна (cutoff 04-15)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
CHUNKS = OUT / "pairs"
CHUNKS.mkdir(exist_ok=True)
CNT = OUT / "item_counts"
CNT.mkdir(exist_ok=True)

THR = 1775595600000  # 2026-04-07T21:00Z — фактический порог local_eval.csv (naive MSK)
START = THR - 28 * 24 * 3600 * 1000
DEPTH, MAX_GAP, TOP_N, MIN_PAIR_USERS = 3, 24 * 3600 * 1000, 50, 3
N_SEED, N_CAND, K = 15, 400, 160
VERTS = [0, 2, 3, 4, 5, 7]

contact = pl.read_csv(RAW / "contact_eids.csv")["mapped_eid"].to_list()
parts = sorted((RAW / "train_data").glob("part_*.parquet"))
t0 = time.time()
for i, part in enumerate(parts):
    if (CHUNKS / f"p{i:03d}.parquet").exists():
        continue
    df = (
        pl.scan_parquet(part)
        .filter((pl.col("timestamp") >= START) & (pl.col("timestamp") < THR) & (pl.col("eid") != 7))
        .select(["user_id", "item_id", "timestamp", "eid"])
        .sort(["user_id", "timestamp"])
        .collect(engine="streaming")
    )
    # агрегат как в aggregate.py (вес, last_ts, top-100 свежих)
    agg = (
        df.with_columns(
            pl.when(pl.col("eid").is_in(contact)).then(5).otherwise(1).cast(pl.UInt16).alias("w")
        )
        .group_by(["user_id", "item_id"])
        .agg(pl.col("w").sum().alias("weight"), pl.col("timestamp").max().alias("last_timestamp"))
        .sort(["user_id", "last_timestamp"], descending=[False, True])
        .group_by("user_id")
        .head(100)
    )
    agg.group_by("item_id").agg(
        pl.len().cast(pl.UInt32).alias("n_users"), pl.col("weight").sum().alias("w")
    ).write_parquet(CNT / f"p{i:03d}.parquet")
    if i == 0:
        agg.write_parquet(OUT / "agg_part000.parquet")
    frames = []
    for k in range(1, DEPTH + 1):
        s = df.select(
            "user_id",
            "item_id",
            "timestamp",
            pl.col("item_id").shift(-k).over("user_id").alias("nx"),
            pl.col("timestamp").shift(-k).over("user_id").alias("tn"),
        ).filter(
            pl.col("nx").is_not_null()
            & (pl.col("nx") != pl.col("item_id"))
            & ((pl.col("tn") - pl.col("timestamp")) <= MAX_GAP)
        )
        frames.append(
            s.select(
                "user_id",
                pl.min_horizontal("item_id", "nx").alias("item_a"),
                pl.max_horizontal("item_id", "nx").alias("item_b"),
            )
        )
    pl.concat(frames).unique().write_parquet(CHUNKS / f"p{i:03d}.parquet")
    del df, frames, agg
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/100 {(time.time() - t0) / 60:.1f} мин", flush=True)

con = duckdb.connect()
con.execute("SET memory_limit='6GB'; SET threads=4;")
con.execute(f"SET temp_directory='{OUT / 'tmp'}';")
cv = OUT / "covisit_matrix.parquet"
if not cv.exists():
    con.execute(f"""
    COPY (
      WITH pc AS (SELECT item_a, item_b, COUNT(DISTINCT user_id) n FROM read_parquet('{CHUNKS}/*.parquet')
                  GROUP BY 1,2 HAVING COUNT(DISTINCT user_id) >= {MIN_PAIR_USERS}),
      d AS (SELECT item_a src, item_b dst, n w FROM pc UNION ALL SELECT item_b, item_a, n FROM pc),
      r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY src ORDER BY w DESC) rn FROM d)
      SELECT src item_id, dst neighbor_id, w weight, rn neighbor_rank FROM r WHERE rn <= {TOP_N}
    ) TO '{cv}' (FORMAT PARQUET);""")
print("covisit:", con.sql(f"select count(*), count(distinct item_id) from '{cv}'").fetchall())

con.execute(f"""CREATE TABLE item_cnt AS SELECT item_id, SUM(n_users) n_users, SUM(w) w
               FROM read_parquet('{CNT}/*.parquet') GROUP BY 1""")
con.execute(f"""CREATE TABLE allowed AS SELECT c.item_id FROM item_cnt c
               JOIN (SELECT item_id, vertical_id FROM read_parquet('{RAW}/item_features.parquet')) f USING(item_id)
               WHERE f.vertical_id IN {tuple(VERTS)} AND c.n_users >= 2""")
print("allowed pool:", con.sql("select count(*) from allowed").fetchall())

val = pl.read_parquet(LEAKY / "features/val_predictions.parquet")["user_id"].unique()
con.register("val_users", val.to_frame())
con.execute(
    f"CREATE TABLE tgt AS SELECT * FROM read_csv('{RAW}/local_eval.csv') WHERE user_id IN (SELECT user_id FROM val_users)"
)


def recall(cands_sql: str) -> float:
    return con.sql(f"""
      WITH c AS ({cands_sql}),
      top AS (SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC) rn FROM c) WHERE rn <= {K}),
      h AS (SELECT user_id, COUNT(*) hits FROM top JOIN tgt USING(user_id, item_id) GROUP BY 1),
      tot AS (SELECT user_id, COUNT(*) n FROM tgt GROUP BY 1)
      SELECT AVG(COALESCE(hits,0)/n) FROM tot LEFT JOIN h USING(user_id)""").fetchone()[0]


def ceiling(cands_sql: str) -> float:
    return con.sql(f"""WITH c AS ({cands_sql})
      SELECT COUNT(*) / (SELECT COUNT(*) FROM tgt) FROM tgt SEMI JOIN c USING(user_id, item_id)""").fetchone()[
        0
    ]


def covisit_sql(agg: str, matrix: str, pool: str) -> str:
    return f"""
      WITH seeds AS (SELECT user_id, item_id seed FROM (
             SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY last_timestamp DESC) rn
             FROM read_parquet('{agg}') WHERE user_id IN (SELECT user_id FROM val_users)) WHERE rn <= {N_SEED}),
      sc AS (SELECT s.user_id, cv.neighbor_id item_id, SUM(cv.weight) score FROM seeds s
             JOIN read_parquet('{matrix}') cv ON cv.item_id = s.seed
             WHERE cv.neighbor_id IN (SELECT item_id FROM {pool}) GROUP BY 1,2)
      SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC) rn FROM sc) WHERE rn <= {N_CAND}"""


# leaky reproduction (оригинальные артефакты) — для проверки, что метрика воспроизводится
m = np.load(LEAKY / "processed/mappings.npz")["item_ids"]
idx = np.load(LEAKY / "processed/allowed_item_pool.npz")["allowed_item_idx"]
con.register("allowed_leaky_df", pl.DataFrame({"item_id": m[idx]}))
con.execute("CREATE TABLE allowed_leaky AS SELECT * FROM allowed_leaky_df")
con.execute(f"""CREATE TABLE agg_leaky AS SELECT * FROM read_parquet('{LEAKY}/interim/interactions_agg.parquet')
               WHERE user_id IN (SELECT user_id FROM val_users)""")
con.execute(f"COPY agg_leaky TO '{OUT}/agg_leaky_val.parquet' (FORMAT PARQUET)")

res = {}
for name, agg, mat, pool in [
    (
        "leaky",
        f"{OUT}/agg_leaky_val.parquet",
        f"{LEAKY}/processed/covisit_matrix.parquet",
        "allowed_leaky",
    ),
    ("honest", f"{OUT}/agg_part000.parquet", str(cv), "allowed"),
    (
        "honest_seeds+leaky_matrix",
        f"{OUT}/agg_part000.parquet",
        f"{LEAKY}/processed/covisit_matrix.parquet",
        "allowed_leaky",
    ),
    ("leaky_seeds+honest_matrix", f"{OUT}/agg_leaky_val.parquet", str(cv), "allowed"),
]:
    q = covisit_sql(agg, mat, pool)
    res[name] = (recall(q), ceiling(q))
    print(
        f"covisit {name:28s} Recall@160={res[name][0]:.4f}  ceiling@400={res[name][1]:.4f}",
        flush=True,
    )

for name, cnt_sql in [
    (
        "leaky",
        f"SELECT item_id, COUNT(*) n FROM read_parquet('{LEAKY}/interim/interactions_agg.parquet') GROUP BY 1",
    ),
    ("honest", "SELECT item_id, n_users n FROM item_cnt"),
]:
    q = f"""WITH top AS (SELECT c.item_id, c.n FROM ({cnt_sql}) c
              JOIN (SELECT item_id, vertical_id FROM read_parquet('{RAW}/item_features.parquet')) f USING(item_id)
              WHERE f.vertical_id IN {tuple(VERTS)} ORDER BY n DESC LIMIT {K})
            SELECT u.user_id, top.item_id, top.n score FROM val_users u CROSS JOIN top"""
    print(f"popular {name:28s} Recall@160={recall(q):.4f}", flush=True)
