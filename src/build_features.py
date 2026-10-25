"""Фичи для ранкера (DuckDB) + таргет из local_eval.csv и проверка темпорального сплита."""

import duckdb

from config import ITEM_FEATURES, LOCAL_EVAL, TMP_DIR, parse_run

MEM_LIMIT = "7GB"


def build_features() -> None:
    run = parse_run(__doc__)
    candidates = (run.interim / "candidates_merged.parquet").as_posix()
    train_agg = (run.interim / "interactions_agg.parquet").as_posix()
    item_features = ITEM_FEATURES.as_posix()
    out_path = run.features / "features.parquet"

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM_LIMIT}';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR.as_posix()}';")

    print("1. item_popularity (сумма весов + число юзеров) до cutoff...")
    con.execute(f"""
        CREATE TEMP VIEW item_pop AS
        SELECT item_id, SUM(weight) AS item_popularity, COUNT(*) AS item_n_users
        FROM read_parquet('{train_agg}') GROUP BY item_id;
    """)

    print("2. user_top_vertical (самая весомая вертикаль в истории юзера)...")
    con.execute(f"""
        CREATE TEMP VIEW user_top_vertical AS
        WITH uv AS (
            SELECT h.user_id, f.vertical_id, SUM(h.weight) AS v_weight
            FROM read_parquet('{train_agg}') h
            JOIN read_parquet('{item_features}') f USING (item_id)
            GROUP BY h.user_id, f.vertical_id
        ),
        ranked AS (
            SELECT user_id, vertical_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id ORDER BY v_weight DESC, vertical_id
                   ) AS rn
            FROM uv
        )
        SELECT user_id, vertical_id AS user_top_vertical FROM ranked WHERE rn = 1;
    """)

    print("3. Таргет из local_eval.csv + проверка темпорального сплита...")
    con.execute(f"""
        CREATE TEMP VIEW target AS
        SELECT DISTINCT user_id, item_id, TRUE AS is_target FROM read_csv('{LOCAL_EVAL}');
    """)
    # таргет по построению новый (user, item) после cutoff -> не может быть в истории
    overlap = con.execute(f"""
        SELECT COUNT(*) FROM target t
        JOIN read_parquet('{train_agg}') a USING (user_id, item_id)
    """).fetchone()[0]
    if run.leaky:
        print(f"   ВНИМАНИЕ (leaky): {overlap:,} таргет-пар присутствуют в истории")
    else:
        assert overlap == 0, f"утечка: {overlap:,} таргет-пар в истории до cutoff"

    print("4. Финальный запрос -> parquet...")
    con.execute(f"""
        COPY (
            SELECT
                c.user_id, c.item_id,
                c.als_score, c.als_rank,
                c.covisit_score, c.covisit_rank,
                CAST(c.in_als AS INTEGER) AS in_als,
                CAST(c.in_covisit AS INTEGER) AS in_covisit,
                c.is_fallback,
                f.vertical_id, f.category_ext_y, f.region_id_y, f.loc_id_y,
                COALESCE(p.item_popularity, 0) AS item_popularity,
                COALESCE(p.item_n_users, 0) AS item_n_users,
                CAST((f.vertical_id = utv.user_top_vertical) AS INTEGER) AS user_top_vertical_match,
                COALESCE(t.is_target, FALSE) AS target
            FROM read_parquet('{candidates}') c
            LEFT JOIN read_parquet('{item_features}') f USING (item_id)
            LEFT JOIN item_pop p USING (item_id)
            LEFT JOIN user_top_vertical utv USING (user_id)
            LEFT JOIN target t USING (user_id, item_id)
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    print(f"Готово! {n:,} строк -> {out_path}")
    pos = con.execute(
        f"SELECT SUM(CAST(target AS INTEGER)) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()[0]
    print(f"   Positive: {pos:,} ({pos / n:.5%})")
    con.close()


if __name__ == "__main__":
    build_features()
