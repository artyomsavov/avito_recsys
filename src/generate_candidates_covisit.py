"""Co-visitation кандидаты: последние N item истории юзера -> соседи -> агрегированный скор."""

import duckdb
import numpy as np
import polars as pl

from config import TMP_DIR, parse_run, target_users

N_SEED_ITEMS = 15  # последние N item из истории юзера
N_CANDIDATES = 400  # итоговых covisit-кандидатов на юзера


def generate_candidates_covisit() -> None:
    run = parse_run(__doc__)
    con = duckdb.connect()
    con.execute("SET memory_limit='7GB';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")

    covisit = (run.processed / "covisit_matrix.parquet").as_posix()
    history = (run.interim / "interactions_agg.parquet").as_posix()
    out_path = run.interim / "candidates_covisit.parquet"

    print("1. Восстанавливаем allowed item_id из индексов пула...")
    train_item_ids = np.load(run.processed / "mappings.npz")["item_ids"]
    allowed_idx = np.load(run.processed / "allowed_item_pool.npz")["allowed_item_idx"]
    con.register("allowed_items", pl.DataFrame({"item_id": train_item_ids[allowed_idx]}))
    con.register("users", pl.DataFrame({"user_id": target_users()}))

    print(f"2. Отбираем последние {N_SEED_ITEMS} затравок на юзера...")
    con.execute(f"""
        CREATE TEMP VIEW seeds AS
        WITH ranked AS (
            SELECT user_id, item_id, last_timestamp,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id ORDER BY last_timestamp DESC, item_id
                   ) AS rn
            FROM read_parquet('{history}')
            WHERE user_id IN (SELECT user_id FROM users)
        )
        SELECT user_id, item_id AS seed_item_id
        FROM ranked WHERE rn <= {N_SEED_ITEMS};
    """)
    max_seed_ts = con.execute(f"""
        SELECT MAX(h.last_timestamp) FROM seeds s
        JOIN read_parquet('{history}') h ON h.user_id = s.user_id AND h.item_id = s.seed_item_id
    """).fetchone()[0]
    assert max_seed_ts < run.cutoff_ms, "затравка после cutoff"

    print("3. Джойним затравки с co-visitation матрицей, агрегируем скор кандидата...")
    con.execute(f"""
        CREATE TEMP VIEW scored AS
        SELECT s.user_id, cv.neighbor_id AS item_id, SUM(cv.weight) AS covisit_score
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
                       ROW_NUMBER() OVER (
                           PARTITION BY user_id ORDER BY covisit_score DESC, item_id
                       ) - 1
                           AS covisit_rank
                FROM scored
            )
            SELECT user_id, item_id, covisit_score, covisit_rank
            FROM ranked WHERE covisit_rank < {N_CANDIDATES}
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n, n_users = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT user_id) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()
    print(f"Готово! {n:,} строк, {n_users:,} юзеров покрыто -> {out_path}")
    con.close()


if __name__ == "__main__":
    generate_candidates_covisit()
