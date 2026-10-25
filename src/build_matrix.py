"""interactions_agg -> CSR-матрица (юзеры x item) + id-маппинги."""

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, save_npz

from config import parse_run


def build_sparse_matrix() -> None:
    run = parse_run(__doc__)
    print("1. Читаем агрегированные данные...")
    df = pl.read_parquet(run.interim / "interactions_agg.parquet")

    print("2. Создаем маппинги (ID -> Index)...")
    unique_users = df["user_id"].unique().sort().to_numpy()
    unique_items = df["item_id"].unique().sort().to_numpy()
    rows = np.searchsorted(unique_users, df["user_id"].to_numpy())
    cols = np.searchsorted(unique_items, df["item_id"].to_numpy())
    data = df["weight"].to_numpy().astype(np.float32)

    print("3. Собираем COO матрицу и конвертируем в CSR...")
    shape = (len(unique_users), len(unique_items))
    sparse_mat = coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    print(f"   Размер матрицы: {shape[0]:,} юзеров х {shape[1]:,} айтемов.")
    print(f"   Ненулевых элементов: {sparse_mat.nnz:,}")

    print("4. Сохраняем артефакты...")
    save_npz(run.processed / "interactions.npz", sparse_mat)
    np.savez_compressed(
        run.processed / "mappings.npz", user_ids=unique_users, item_ids=unique_items
    )
    print(f"Готово! Артефакты сохранены в {run.processed}/")


if __name__ == "__main__":
    build_sparse_matrix()
