import polars as pl
import catboost as cb
import os
import gc

SEED = 42

# --- НАСТРОЙКИ ПУТЕЙ ---
DATA_DIR = "../data/"
MATRIX_DIR = f"{DATA_DIR}processed/train_matrices/"
MODELS_DIR = f"{DATA_DIR}models/"

os.makedirs(MODELS_DIR, exist_ok=True)

# --- 1. ЗАГРУЗКА ---
print("Загрузка обучающих матриц...")
df = pl.scan_parquet(f"{MATRIX_DIR}*.parquet").fill_null(0).collect()

# --- 2. ПРИЗНАКИ ---
CAT_FEATURES = ["vertical_id", "category_ext_y", "region_id_y", "u_top_vertical", "u_top_category", "u_top_region"]
NUMERIC_FEATURES = ["score", "source_pop", "source_covis", "is_top_vertical"]
FEATURES = NUMERIC_FEATURES + CAT_FEATURES

df = df.with_columns([pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in CAT_FEATURES])

# --- 3. АГРЕССИВНОЕ СЭМПЛИРОВАНИЕ ---
print("Применяем агрессивное сэмплирование для обучения...")
df = df.with_row_index("row_id")
pos_df = df.filter(pl.col("target") == 1)
neg_df = df.filter(pl.col("target") == 0).sample(n=500000, seed=SEED) 

train_pos = pos_df.sample(fraction=0.8, seed=42)
train_neg = neg_df.sample(fraction=0.8, seed=42)

train_df = pl.concat([train_pos, train_neg]).sample(fraction=1.0, seed=SEED)
val_df = pl.concat([
    pos_df.join(train_df.select("row_id"), on="row_id", how="anti"),
    neg_df.join(train_df.select("row_id"), on="row_id", how="anti").head(100000)
])

pos_weight = (train_df.filter(pl.col("target") == 0).height / 
              train_df.filter(pl.col("target") == 1).height)

del df, pos_df, neg_df, train_pos, train_neg
gc.collect()

# --- 4. ПУЛЫ ---
print("Инициализация пулов...")
train_pool = cb.Pool(
    data=train_df.select(FEATURES).to_pandas(),
    label=train_df.select("target").to_pandas(),
    cat_features=CAT_FEATURES
)
val_pool = cb.Pool(
    data=val_df.select(FEATURES).to_pandas(),
    label=val_df.select("target").to_pandas(),
    cat_features=CAT_FEATURES
)

del train_df, val_df
gc.collect()

# --- 5. ОБУЧЕНИЕ ---
print("\nСтарт обучения CatBoost...")
model = cb.CatBoostClassifier(
    iterations=1000, 
    learning_rate=0.05,        
    depth=6,                   
    scale_pos_weight=pos_weight, 
    eval_metric='Logloss', 
    random_seed=SEED,
    task_type="CPU",
    thread_count=3,            
    early_stopping_rounds=100, 
    verbose=100                
)

model.fit(train_pool, eval_set=val_pool)

model.save_model(f"{MODELS_DIR}catboost_ranker.cbm")
print(f"Модель сохранена в {MODELS_DIR}catboost_ranker.cbm!")
