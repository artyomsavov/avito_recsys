"""ALS (implicit, CPU) на CSR-матрице. Факторы сохраняются отдельными .npy для mmap."""

import implicit
import numpy as np
from scipy.sparse import load_npz

from config import parse_run

FACTORS = 32
SEED = 42


def train_als() -> None:
    run = parse_run(__doc__)
    print("1. Загружаем CSR-матрицу...")
    sparse_mat = load_npz(run.processed / "interactions.npz")

    print(f"2. Обучаем ALS (factors={FACTORS})...")
    model = implicit.cpu.als.AlternatingLeastSquares(
        factors=FACTORS,
        regularization=0.1,
        iterations=15,
        calculate_training_loss=True,
        num_threads=16,
        random_state=SEED,
    )
    model.fit(sparse_mat)

    print("3. Сохраняем факторы...")
    # отдельные .npy, а не model.save (.npz-zip): item_factors (4.4 GB) читаются через mmap
    np.save(run.models / "item_factors.npy", model.item_factors)
    np.save(run.models / "user_factors.npy", model.user_factors)
    print(f"Готово! Факторы сохранены в {run.models}")


if __name__ == "__main__":
    train_als()
