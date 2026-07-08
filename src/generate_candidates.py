import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

N_CANDIDATES = 400
BATCH_SIZE = 15
FACTORS = 32  # как в train_als.py
CHUNK_WRITE_EVERY = 50


def ensure_item_factors_npy() -> Path:
    """
    implicit сохраняет модель как .npz (zip). Чтобы использовать честный OS-level mmap,
    один раз вытаскиваем item_factors.npy на диск как самостоятельный файл.
    """
    npy_path = MODELS_DIR / "item_factors_extracted.npy"
    if npy_path.exists():
        return npy_path

    print("Извлекаем item_factors.npy из als_model.npz на диск (однократно)...")
    with zipfile.ZipFile(MODELS_DIR / "als_model.npz") as zf:
        with zf.open("item_factors.npy") as src, open(npy_path, "wb") as dst:
            # копируем чанками, не читая всё в память
            while chunk := src.read(64 * 1024 * 1024):
                dst.write(chunk)
    return npy_path


def build_user_history_matrix(
    eval_interactions: pl.DataFrame,
    train_item_ids: np.ndarray,
    n_train_items: int,
) -> tuple[dict[int, int], csr_matrix]:
    """
    Строим CSR (n_eval_users x n_train_items) — историю eval-юзеров в ПОЛНОМ
    индексном пространстве train-item (важно: не фильтруем по вертикалям здесь,
    т.к. для recalculate_user нужна вся история юзера, включая "неразрешённые" вертикали).
    """
    unique_eval_users = eval_interactions["user_id"].unique().sort().to_numpy()
    eval_user_id_to_row = {int(u): i for i, u in enumerate(unique_eval_users)}

    # item_id -> idx через searchsorted (без огромного словаря)
    item_idx = np.searchsorted(train_item_ids, eval_interactions["item_id"].to_numpy())
    valid = (item_idx < n_train_items) & (
        train_item_ids[np.clip(item_idx, 0, n_train_items - 1)]
        == eval_interactions["item_id"].to_numpy()
    )

    rows = eval_interactions["user_id"].to_numpy()
    row_idx = np.array([eval_user_id_to_row[int(u)] for u in rows[valid]], dtype=np.int64)
    col_idx = item_idx[valid].astype(np.int64)
    data = eval_interactions["weight"].to_numpy()[valid].astype(np.float32)

    dropped = (~valid).sum()
    if dropped:
        print(
            f"   {dropped:,} строк истории eval-юзеров ссылаются на \
                item вне train mappings (пропущены)"
        )

    mat = csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(unique_eval_users), n_train_items),
    )
    return unique_eval_users, mat


def generate_candidates() -> None:
    print("1. Загружаем маппинги train item...")
    mappings = np.load(PROCESSED_DIR / "mappings.npz")
    train_item_ids = mappings["item_ids"]
    n_train_items = len(train_item_ids)

    print("2. Загружаем разрешённый пул item...")
    allowed_item_idx = np.load(PROCESSED_DIR / "allowed_item_pool.npz")["allowed_item_idx"]
    print(f"   Пул для scoring: {len(allowed_item_idx):,} item")

    print("3. Загружаем агрегированную историю eval-юзеров...")
    eval_interactions = pl.read_parquet(INTERIM_DIR / "eval_interactions_agg.parquet")

    print("4. Строим sparse-историю eval-юзеров в полном item-индексе...")
    eval_user_ids, history_matrix = build_user_history_matrix(
        eval_interactions, train_item_ids, n_train_items
    )

    print("5. Готовим item_factors через честный OS-level mmap (не грузим 4.4GB целиком)...")
    item_factors_path = ensure_item_factors_npy()
    item_factors = np.load(item_factors_path, mmap_mode="r+")
    assert item_factors.shape == (n_train_items, FACTORS), item_factors.shape

    print("6. Строим модель-обёртку без user_factors (не нужны для recalculate_user)...")
    model = AlternatingLeastSquares(factors=FACTORS)
    model.item_factors = item_factors

    print(f"7. Генерируем top-{N_CANDIDATES} кандидатов батчами по {BATCH_SIZE} eval-юзеров...")
    print(f"   Всего eval-юзеров с историей: {len(eval_user_ids):,}")

    out_dir = INTERIM_DIR / "candidates_chunks"
    out_dir.mkdir(exist_ok=True)
    chunk_rows: list[tuple] = []
    chunk_i = 0

    n_batches = (len(eval_user_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    t_start = time.time()
    for b in range(n_batches):
        if (b + 1) % 10 == 0:
            elapsed = time.time() - t_start
            per_batch = elapsed / (b + 1)
            remaining_min = per_batch * (n_batches - b - 1) / 60
            print(
                f"   батч {b + 1}/{n_batches} | {per_batch:.2f} сек/батч | \
                осталось ~{remaining_min:.1f} мин"
            )

        start = b * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(eval_user_ids))
        batch_uids = eval_user_ids[start:end]
        batch_history = history_matrix[start:end]

        # dummy userid — recalculate_user=True игнорирует хранимые факторы,
        # реальные значения id здесь не важны, важна только длина совпадения с батчем
        dummy_userid = np.arange(end - start, dtype=np.int64)
        model.user_factors = np.zeros((end - start, FACTORS), dtype=np.float32)

        ids_batch, scores_batch = model.recommend(
            dummy_userid,
            batch_history,
            N=N_CANDIDATES,
            filter_already_liked_items=False,  # сами решаем на этапе фичей/ранкера
            recalculate_user=True,
            items=allowed_item_idx,
        )

        for row_i, uid in enumerate(batch_uids):
            for rank, (it_idx, score) in enumerate(zip(ids_batch[row_i], scores_batch[row_i])):
                if it_idx < 0:
                    continue
                chunk_rows.append((int(uid), int(train_item_ids[it_idx]), float(score), rank))

        if (b + 1) % CHUNK_WRITE_EVERY == 0 or b == n_batches - 1:
            chunk_df = pl.DataFrame(
                chunk_rows,
                schema=["user_id", "item_id", "als_score", "als_rank"],
                orient="row",
            )
            chunk_df.write_parquet(out_dir / f"chunk_{chunk_i:05d}.parquet")
            chunk_i += 1
            chunk_rows = []
            print(f"   батч {b + 1}/{n_batches} — чанк {chunk_i} записан на диск")

    print("8. Склеиваем чанки в единый файл (потоково, без загрузки всего в RAM)...")
    pl.scan_parquet(out_dir / "chunk_*.parquet").sink_parquet(INTERIM_DIR / "candidates.parquet")
    print(f"Готово! Кандидаты сохранены в {INTERIM_DIR / 'candidates.parquet'}")


if __name__ == "__main__":
    generate_candidates()
