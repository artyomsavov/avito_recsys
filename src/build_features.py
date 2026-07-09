from pathlib import Path

import duckdb

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = (INTERIM_DIR / "candidates_full.parquet").as_posix()
ITEM_FEATURES = (RAW_DIR / "item_features.parquet").as_posix()
TRAIN_AGG = (INTERIM_DIR / "interactions_agg.parquet").as_posix()
EVAL_AGG = (INTERIM_DIR / "eval_interactions_agg.parquet").as_posix()
LOCAL_EVAL = (RAW_DIR / "local_eval.csv").as_posix()

MEM_LIMIT = "7GB"
TMP_DIR = (DATA_DIR / "duckdb_tmp").as_posix()


def build_features(out_path: Path, with_target: bool) -> None:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM_LIMIT}';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR}';")

    print("1. Строим item_popularity (сумма весов + число юзеров) как VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW item_pop AS
        SELECT item_id,
               SUM(weight) AS item_popularity,
               COUNT(*)    AS item_n_users
        FROM read_parquet('{TRAIN_AGG}')
        GROUP BY item_id;
    """)

    print("2. Строим user_top_vertical (самая частая вертикаль на юзера) как VIEW...")
    # объединяем train+eval историю, джойним вертикаль, берём топ-1 по весу на юзера
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW user_top_vertical AS
        WITH all_inter AS (
            SELECT user_id, item_id, weight FROM read_parquet('{TRAIN_AGG}')
            UNION ALL
            SELECT user_id, item_id, weight FROM read_parquet('{EVAL_AGG}')
        ),
        uv AS (
            SELECT a.user_id, f.vertical_id, SUM(a.weight) AS v_weight
            FROM all_inter a
            JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            GROUP BY a.user_id, f.vertical_id
        ),
        ranked AS (
            SELECT user_id, vertical_id,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY v_weight DESC) AS rn
            FROM uv
        )
        SELECT user_id, vertical_id AS user_top_vertical
        FROM ranked WHERE rn = 1;
    """)

    target_select = ""
    target_join = ""
    if with_target:
        print("3. Подключаем таргет из local_eval.csv...")
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW target AS
            SELECT DISTINCT user_id, item_id, TRUE AS is_target
            FROM read_csv('{LOCAL_EVAL}');
        """)
        target_select = ", COALESCE(t.is_target, FALSE) AS target"
        target_join = "LEFT JOIN target t USING (user_id, item_id)"

    print("4. Финальный запрос: джойним всё и пишем в parquet потоково...")
    con.execute(f"""
        COPY (
            SELECT
                c.user_id,
                c.item_id,
                c.als_score,
                c.als_rank,
                (c.source = 'popular_fallback')                     AS is_cold,
                f.vertical_id,
                f.category_ext_y,
                f.region_id_y,
                f.loc_id_y,
                COALESCE(p.item_popularity, 0)                      AS item_popularity,
                COALESCE(p.item_n_users, 0)                         AS item_n_users,
                (f.vertical_id = utv.user_top_vertical)             AS user_top_vertical_match
                {target_select}
            FROM read_parquet('{CANDIDATES}') c
            LEFT JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            LEFT JOIN item_pop p           USING (item_id)
            LEFT JOIN user_top_vertical utv USING (user_id)
            {target_join}
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[
        0
    ]
    print(f"Готово! {n_rows:,} строк -> {out_path}")

    if with_target:
        pos = con.execute(
            f"SELECT AVG(CAST(target AS DOUBLE)) FROM read_parquet('{out_path.as_posix()}')"
        ).fetchone()[0]
        print(f"   Доля positive (target): {pos:.5%}")

    con.close()


CANDIDATES_SYNTH = (INTERIM_DIR / "candidates_synth.parquet").as_posix()


def build_features_synth() -> None:
    out_path = FEATURES_DIR / "train_features_synth.parquet"
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM_LIMIT}';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR}';")

    print("1. item_popularity VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW item_pop AS
        SELECT item_id, SUM(weight) AS item_popularity, COUNT(*) AS item_n_users
        FROM read_parquet('{TRAIN_AGG}') GROUP BY item_id;
    """)

    print("2. user_top_vertical VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW user_top_vertical AS
        WITH all_inter AS (
            SELECT user_id, item_id, weight FROM read_parquet('{TRAIN_AGG}')
            UNION ALL
            SELECT user_id, item_id, weight FROM read_parquet('{EVAL_AGG}')
        ),
        uv AS (
            SELECT a.user_id, f.vertical_id, SUM(a.weight) AS v_weight
            FROM all_inter a
            JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            GROUP BY a.user_id, f.vertical_id
        ),
        ranked AS (
            SELECT user_id, vertical_id,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY v_weight DESC) AS rn
            FROM uv
        )
        SELECT user_id, vertical_id AS user_top_vertical FROM ranked WHERE rn = 1;
    """)

    print("3. target VIEW из local_eval.csv...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW target AS
        SELECT DISTINCT user_id, item_id, TRUE AS is_target
        FROM read_csv('{LOCAL_EVAL}');
    """)

    print("4. Финальный запрос -> parquet...")
    con.execute(f"""
        COPY (
            SELECT
                c.user_id, c.item_id, c.als_score, c.als_rank,
                (c.source = 'popular_fallback') AS is_cold,
                f.vertical_id, f.category_ext_y, f.region_id_y, f.loc_id_y,
                COALESCE(p.item_popularity, 0) AS item_popularity,
                COALESCE(p.item_n_users, 0) AS item_n_users,
                (f.vertical_id = utv.user_top_vertical) AS user_top_vertical_match,
                COALESCE(t.is_target, FALSE) AS target
            FROM read_parquet('{CANDIDATES_SYNTH}') c
            LEFT JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            LEFT JOIN item_pop p USING (item_id)
            LEFT JOIN user_top_vertical utv USING (user_id)
            LEFT JOIN target t USING (user_id, item_id)
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    pos = con.execute(
        f"SELECT AVG(CAST(target AS DOUBLE)) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()[0]
    print(f"Готово! {n:,} строк -> {out_path}")
    print(f"   Доля positive (target): {pos:.5%}")
    con.close()


CANDIDATES_MERGED = (INTERIM_DIR / "candidates_merged_synth.parquet").as_posix()


def build_features_merged() -> None:
    out_path = FEATURES_DIR / "train_features_merged.parquet"
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{MEM_LIMIT}';")
    con.execute("SET threads=4;")
    con.execute(f"SET temp_directory='{TMP_DIR}';")

    print("1. item_popularity VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW item_pop AS
        SELECT item_id, SUM(weight) AS item_popularity, COUNT(*) AS item_n_users
        FROM read_parquet('{TRAIN_AGG}') GROUP BY item_id;
    """)

    print("2. user_top_vertical VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW user_top_vertical AS
        WITH all_inter AS (
            SELECT user_id, item_id, weight FROM read_parquet('{TRAIN_AGG}')
            UNION ALL
            SELECT user_id, item_id, weight FROM read_parquet('{EVAL_AGG}')
        ),
        uv AS (
            SELECT a.user_id, f.vertical_id, SUM(a.weight) AS v_weight
            FROM all_inter a
            JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            GROUP BY a.user_id, f.vertical_id
        ),
        ranked AS (
            SELECT user_id, vertical_id,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY v_weight DESC) AS rn
            FROM uv
        )
        SELECT user_id, vertical_id AS user_top_vertical FROM ranked WHERE rn = 1;
    """)

    print("3. target VIEW...")
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW target AS
        SELECT DISTINCT user_id, item_id, TRUE AS is_target
        FROM read_csv('{LOCAL_EVAL}');
    """)

    print("4. Финальный запрос -> parquet...")
    con.execute(f"""
        COPY (
            SELECT
                c.user_id, c.item_id,
                c.als_score, c.als_rank,
                c.covisit_score, c.covisit_rank,
                CAST(c.in_als AS INTEGER) AS in_als,
                CAST(c.in_covisit AS INTEGER) AS in_covisit,
                f.vertical_id, f.category_ext_y, f.region_id_y, f.loc_id_y,
                COALESCE(p.item_popularity, 0) AS item_popularity,
                COALESCE(p.item_n_users, 0) AS item_n_users,
                CAST((f.vertical_id = utv.user_top_vertical) AS INTEGER) AS user_top_vertical_match,
                COALESCE(t.is_target, FALSE) AS target
            FROM read_parquet('{CANDIDATES_MERGED}') c
            LEFT JOIN read_parquet('{ITEM_FEATURES}') f USING (item_id)
            LEFT JOIN item_pop p USING (item_id)
            LEFT JOIN user_top_vertical utv USING (user_id)
            LEFT JOIN target t USING (user_id, item_id)
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET);
    """)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    pos = con.execute(
        f"SELECT SUM(CAST(target AS INTEGER)) FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()[0]
    print(f"Готово! {n:,} строк -> {out_path}")
    print(f"   Positive: {pos:,} ({pos / n:.5%})")
    con.close()


if __name__ == "__main__":
    Path(TMP_DIR).mkdir(parents=True, exist_ok=True)

    # build_features(FEATURES_DIR / "infer_features.parquet", with_target=False)
    # build_features(FEATURES_DIR / "train_features.parquet", with_target=True)
    # build_features_synth()
    build_features_merged()
