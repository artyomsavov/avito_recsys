from pathlib import Path

import numpy as np
import polars as pl


def create_eval():
    np.random.seed(42)
    maps = np.load("data/processed/mappings.npz")
    train_users = maps["user_ids"]

    sampled_users = np.random.choice(train_users, size=50000, replace=False)

    df = pl.DataFrame({"user_id": sampled_users}).with_columns(pl.col("user_id").cast(pl.Int64))

    out_path = Path("data/interim/local_eval_users.csv")
    df.write_csv(out_path)
    print(f"Создан локальный таргет-сет на 50 000 юзеров: {out_path}")


if __name__ == "__main__":
    create_eval()
