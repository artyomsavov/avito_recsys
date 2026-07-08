from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, save_npz

DATA_DIR = Path("data")
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def build_sparse_matrix() -> None:
    print("1. Читаем агрегированные данные...")
    df = pl.read_parquet(INTERIM_DIR / "interactions_agg.parquet")

    print("2. Создаем словари маппинга (ID -> Index)...")
    unique_users = df["user_id"].unique().sort().to_numpy()
    unique_items = df["item_id"].unique().sort().to_numpy()

    user_map = pl.DataFrame(
        {"user_id": unique_users, "user_idx": np.arange(len(unique_users), dtype=np.int32)}
    )

    item_map = pl.DataFrame(
        {"item_id": unique_items, "item_idx": np.arange(len(unique_items), dtype=np.int32)}
    )

    print("3. Перекодируем ID в индексы...")
    df = df.join(user_map, on="user_id", how="left")
    df = df.join(item_map, on="item_id", how="left")

    print("4. Собираем COO матрицу и конвертируем в CSR...")
    rows = df["user_idx"].to_numpy()
    cols = df["item_idx"].to_numpy()
    data = df["weight"].to_numpy().astype(np.float32)

    shape = (len(unique_users), len(unique_items))
    sparse_mat = coo_matrix((data, (rows, cols)), shape=shape).tocsr()

    sparsity = (sparse_mat.nnz / (shape[0] * shape[1])) * 100
    print(f"   Размер матрицы: {shape[0]:,} юзеров х {shape[1]:,} айтемов.")
    print(f"   Ненулевых элементов: {sparse_mat.nnz:,}")
    print(f"   Плотность: {sparsity:.5f}%")

    print("5. Сохраняем артефакты...")
    save_npz(PROCESSED_DIR / "interactions.npz", sparse_mat)
    np.savez_compressed(
        PROCESSED_DIR / "mappings.npz", user_ids=unique_users, item_ids=unique_items
    )

    print(f"Готово! Артефакты сохранены в {PROCESSED_DIR}/")


if __name__ == "__main__":
    build_sparse_matrix()
