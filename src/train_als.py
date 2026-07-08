from pathlib import Path

import implicit
from scipy.sparse import load_npz

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_als() -> None:
    print("1. Загружаем CSR-матрицу...")
    sparse_mat = load_npz(PROCESSED_DIR / "interactions.npz")

    print("2. Инициализируем ALS (factors=32)...")
    model = implicit.cpu.als.AlternatingLeastSquares(
        factors=32, regularization=0.1, iterations=15, calculate_training_loss=True, num_threads=16
    )

    print("3. Начинаем обучение...")
    model.fit(sparse_mat)

    print("4. Сохраняем веса модели...")
    model.save(MODELS_DIR / "als_model.npz")
    print(f"Готово! Модель сохранена в {MODELS_DIR}")


if __name__ == "__main__":
    train_als()
