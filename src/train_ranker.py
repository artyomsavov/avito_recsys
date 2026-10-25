"""CatBoostRanker (YetiRank) на synth-фичах. Split по юзерам — замороженный VAL_USERS."""

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from config import VAL_USERS, parse_run

CAT_FEATURES = ["vertical_id", "category_ext_y", "region_id_y", "loc_id_y"]
FEATURE_COLS = [
    "als_score",
    "als_rank",
    "covisit_score",
    "covisit_rank",
    "in_als",
    "in_covisit",
    "vertical_id",
    "category_ext_y",
    "region_id_y",
    "loc_id_y",
    "item_popularity",
    "item_n_users",
    "user_top_vertical_match",
]
VAL_FRACTION = 0.2
SEED = 42


def load_val_users(users: np.ndarray) -> set[int]:
    """Один и тот же val-сплит во всех прогонах; создаётся один раз."""
    if not VAL_USERS.exists():
        users = np.sort(users)
        np.random.default_rng(SEED).shuffle(users)
        VAL_USERS.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"user_id": users[: int(len(users) * VAL_FRACTION)]}).write_csv(VAL_USERS)
    return set(pl.read_csv(VAL_USERS)["user_id"].to_list())


def make_pool(df: pl.DataFrame) -> Pool:
    return Pool(
        data=df.select(FEATURE_COLS).to_pandas(),
        label=df["target"].to_numpy(),
        group_id=df["user_id"].to_numpy(),
        cat_features=CAT_FEATURES,
    )


def train_ranker() -> None:
    run = parse_run(__doc__)
    print("1. Загружаем фичи...")
    df = pl.read_parquet(run.features / "features.parquet").with_columns(
        pl.col("target").cast(pl.Int8)
    )

    print("2. Split по ЮЗЕРАМ (не по строкам — иначе утечка)...")
    is_val = df["user_id"].is_in(load_val_users(df["user_id"].unique().to_numpy()))
    # CatBoost Ranker требует сортировки по group_id; item_id внутри группы фиксирует
    # порядок строк (full join в merge его не гарантирует, а от порядка зависит обучение)
    train_df = df.filter(~is_val).sort(["user_id", "item_id"])
    val_df = df.filter(is_val).sort(["user_id", "item_id"])
    for name, part in (("train", train_df), ("val", val_df)):
        print(
            f"   {name}: {len(part):,} строк / {part['user_id'].n_unique():,} юзеров, "
            f"positive {part['target'].sum():,}"
        )

    print("3. Обучаем CatBoostRanker (YetiRank)...")
    val_pool = make_pool(val_df)
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        task_type="CPU",
        verbose=50,
        train_dir=(run.root / "catboost_info").as_posix(),
    )
    model.fit(make_pool(train_df), eval_set=val_pool)
    model.save_model(run.models / "ranker.cbm")
    print(f"Модель сохранена -> {run.models / 'ranker.cbm'}")

    val_out = val_df.select(["user_id", "item_id", "target"]).with_columns(
        pl.Series("rank_score", model.predict(val_pool))
    )
    val_out.write_parquet(run.features / "val_predictions.parquet")

    print("\nFeature importance (PredictionValuesChange):")
    imp = model.get_feature_importance(type="PredictionValuesChange")
    for name, val_imp in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        print(f"   {name:28s} {val_imp:6.2f}")


if __name__ == "__main__":
    train_ranker()
