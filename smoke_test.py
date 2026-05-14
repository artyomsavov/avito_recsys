import polars as pl
import catboost as cb
import lightgbm as lgb
from tqdm import tqdm
import time

print(f"Polars: {pl.__version__}")
print(f"CatBoost: {cb.__version__}")

# Тест tqdm и Polars
for _ in tqdm(range(100), desc="Проверка tqdm"):
    time.sleep(0.01)

df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
print("Polars DataFrame работает!")
