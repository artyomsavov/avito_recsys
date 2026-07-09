from pathlib import Path

import duckdb
import polars as pl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

N_SEED_ITEMS = 15  # последние N item из истории юзера
N_CANDIDATES = 400  # итоговых covisit-кандидатов на юзера


def generate_candidates_covisit(users_path: Path, out_path: Path) -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='7GB';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{(DATA_DIR / 'duckdb_tmp').as_posix()}';")

    covisit = (PROCESSED_DIR / "covisit_matrix.parquet").as_posix()
    interactions = (INTERIM_DIR / "interactions_agg.parquet").as_posix()
    allowed_pool_path = PROCESSED_DIR / "allowed_item_pool.npz"

    print("1. Восстанавливаем allowed item_id из индексов пула...")
    import numpy as np

    mappings = np.load(PROCESSED_DIR / "mappings.npz")
    train_item_ids = mappings["item_ids"]
    allowed_idx = np.load(allowed_pool_path)["allowed_item_idx"]
    allowed_ids = train_item_ids[allowed_idx].tolist()

    con.execute(f"CREATE TEMP TABLE allowed_items AS SELECT UNNEST({allowed_ids}) AS item_id;")

    print("2. Отбираем последние N затравок на юзера...")
    con.execute(f"""
        CREATE TEMP VIEW seeds AS
        WITH ranked AS (
            SELECT user_id, item_id, last_timestamp,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY last_timestamp DESC) AS rn
            FROM read_parquet('{interactions}')
            WHERE user_id IN (SELECT user_id FROM read_csv('{users_path.as_posix()}'))
        )
        SELECT user_id, item_id AS seed_item_id
        FROM ranked WHERE rn <= {N_SEED_ITEMS};
    """)

    print("3. Джойним затравки с co-visitation матрицей, агрегируем скор кандидата...")
    con.execute(f"""
        CREATE TEMP VIEW scored AS
        SELECT
            s.user_id,
            cv.neighbor_id AS item_id,
            SUM(cv.weight) AS covisit_score
        FROM seeds s
        JOIN read_parquet('{covisit}') cv ON cv.item_id = s.seed_item_id
        JOIN allowed_items ai ON ai.item_id = cv.neighbor_id
        GROUP BY s.user_id, cv.neighbor_id;
    """)

    print("4. Top-N на юзера, пишем результат...")
    con.execute(f"""
        COPY (
            WITH ranked AS (
                SELECT user_id, item_id, covisit_score,
                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY covisit_score DESC) - 1 AS covisit_rank
                FROM scored
            )
            SELECT user_id, item_id, covisit_score, covisit_rank
            FROM ranked WHERE covisit_rank < {N_CANDIDATES}
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    n_users = con.execute(
        f"SELECT COUNT(DISTINCT user_id) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()[0]
    print(f"Готово! {n:,} строк, {n_users:,} юзеров покрыто -> {out_path}")
    con.close()


if __name__ == "__main__":
    generate_candidates_covisit(
        users_path=RAW_DIR / "local_eval_users.csv",
        out_path=INTERIM_DIR / "candidates_covisit_synth.parquet",
    )
