from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
RAW_DIR = DATA_DIR / "raw"

CAT_FEATURES = ["vertical_id", "category_ext_y", "region_id_y", "loc_id_y"]
FEATURE_COLS = [
    "als_score",
    "als_rank",
    "is_cold",
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


def train_ranker() -> None:
    print("1. Загружаем фичи...")
    df = pl.read_parquet(FEATURES_DIR / "train_features_synth.parquet")

    # CatBoost хочет числа
    df = df.with_columns(
        [
            pl.col("is_cold").cast(pl.Int8),
            pl.col("user_top_vertical_match").cast(pl.Int8),
            pl.col("target").cast(pl.Int8),
        ]
    )

    print("2. Split по ЮЗЕРАМ (не по строкам — иначе утечка)...")
    users = df["user_id"].unique().to_numpy()
    rng = np.random.default_rng(SEED)
    rng.shuffle(users)
    n_val = int(len(users) * VAL_FRACTION)
    val_users = set(users[:n_val].tolist())

    is_val = df["user_id"].is_in(list(val_users))
    train_df = df.filter(~is_val)
    val_df = df.filter(is_val)
    print(f"   train: {len(train_df):,} строк / {train_df['user_id'].n_unique():,} юзеров")
    print(f"   val:   {len(val_df):,} строк / {val_df['user_id'].n_unique():,} юзеров")
    print(f"   positive в train: {train_df['target'].sum():,} | в val: {val_df['target'].sum():,}")

    # CatBoost Ranker требует, чтобы данные были отсортированы по group_id
    train_df = train_df.sort("user_id")
    val_df = val_df.sort("user_id")

    print("3. Формируем Pool...")
    train_pool = Pool(
        data=train_df.select(FEATURE_COLS).to_pandas(),
        label=train_df["target"].to_numpy(),
        group_id=train_df["user_id"].to_numpy(),
        cat_features=CAT_FEATURES,
    )
    val_pool = Pool(
        data=val_df.select(FEATURE_COLS).to_pandas(),
        label=val_df["target"].to_numpy(),
        group_id=val_df["user_id"].to_numpy(),
        cat_features=CAT_FEATURES,
    )

    print("4. Обучаем CatBoostRanker (YetiRank)...")
    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=SEED,
        task_type="CPU",
        verbose=50,
    )
    model.fit(train_pool, eval_set=val_pool)

    MODELS_DIR.mkdir(exist_ok=True)
    model.save_model(MODELS_DIR / "ranker.cbm")
    print(f"Модель сохранена -> {MODELS_DIR / 'ranker.cbm'}")

    # сохраним val-предсказания для расчёта Recall@160
    val_scores = model.predict(val_pool)
    val_out = val_df.select(["user_id", "item_id", "target"]).with_columns(
        pl.Series("rank_score", val_scores)
    )
    val_out.write_parquet(FEATURES_DIR / "val_predictions.parquet")
    print(f"Val-предсказания сохранены -> {FEATURES_DIR / 'val_predictions.parquet'}")

    # заодно feature importance
    print("\nFeature importance (PredictionValuesChange):")
    imp = model.get_feature_importance(type="PredictionValuesChange")
    for name, val_imp in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        print(f"   {name:28s} {val_imp:6.2f}")

if __name__ == "__main__":
    train_ranker()
