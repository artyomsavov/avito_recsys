"""Этап 1: почему таргеты W1 не попадают в пул кандидатов (1 партиция).

Каждая таргет-пара (юзер, item) получает класс по первому сработавшему правилу:
  hit500  — в top-500 хотя бы одного источника (это и есть потолок пула final.ipynb)
  U       — у юзера нет событий в окне истории (нет seed-ов, блужданию не с чего начать)
  A1      — у item нет ни одного события до порога (холодный)
  A2      — события есть, но у < 2 юзеров (не входит в universe, блуждание его отбрасывает)
  A3      — в universe, но нет событий за 28 дней истории (нет в графе)
  C       — блуждание доходит, но ранг ≥ 500 во всех источниках (подкласс по глубине)
  B       — в графе, но не достигнут даже в top-5000

    python exp/diag_retrieval.py   # ~10 мин, пишет строку в results.jsonl и exp/cache/diag_w1.parquet
"""

import sys
import time

sys.path.insert(0, "exp")
from run import BASE, CACHE, EV, ITEMS, ctx, record, run_walk, target  # noqa: E402

import polars as pl  # noqa: E402

W, DEEP = 1, 5000
SRC = {  # как в final.ipynb: cr + s24 + u0, β = 0
    "cr": {"entity": ["cat", "reg"], "beta": 0.0},
    "s24": {"entity": "sess", "beta": 0.0},
    "u0": {"entity": "user", "beta": 0.0},
}

t0 = time.time()
tgt = target(W)
cx = ctx(W, BASE["hist_days"])
c = cx.c

# ранг каждой таргет-пары в каждом источнике (top-DEEP, ранги с 0)
ranks = tgt
for name, cfg in SRC.items():
    res = run_walk({**cfg, "n_out": DEEP}, W, name=name)
    ranks = ranks.join(res.select("user_id", "item_id", pl.col(f"{name}_r").cast(pl.Int32)), on=["user_id", "item_id"], how="left")
    print(f"[{time.time() - t0:5.0f}s] {name}: {res.height:,} строк", flush=True)
    del res

pre_items = EV.filter(pl.col("timestamp") < c).select("item_id").unique()
graph_items = cx.pairs.select("item_id").unique()
seed_users = cx.pairs.select("user_id").unique()
best = pl.min_horizontal(*[pl.col(f"{n}_r") for n in SRC])
df = ranks.with_columns(best=best).with_columns(
    cls=pl.when(pl.col("best") < 500).then(pl.lit("hit500"))
    .when(~pl.col("user_id").is_in(seed_users["user_id"].implode())).then(pl.lit("U"))
    .when(~pl.col("item_id").is_in(pre_items["item_id"].implode())).then(pl.lit("A1"))
    .when(~pl.col("item_id").is_in(cx.universe["item_id"].implode())).then(pl.lit("A2"))
    .when(~pl.col("item_id").is_in(graph_items["item_id"].implode())).then(pl.lit("A3"))
    .when(pl.col("best").is_not_null()).then(pl.lit("C"))
    .otherwise(pl.lit("B"))
).join(ITEMS.select("item_id", "vert"), on="item_id", how="left")
df.write_parquet(f"{CACHE}/diag_w{W}.parquet")

n = df.height
share = {k: round(v / n, 4) for k, v in df.group_by("cls").len().sort("cls").rows()}
depth = {f"pool@{d}": round(df.filter(pl.col("best") < d).height / n, 4) for d in (500, 1000, 2000, DEEP)}
only = {}  # уникальный вклад источника в top-500
for s in SRC:
    others = pl.min_horizontal(*[pl.col(f"{o}_r") for o in SRC if o != s]).fill_null(DEEP)
    only[s] = dict(pool500=round(df.filter(pl.col(f"{s}_r") < 500).height / n, 4),
                   only500=round(df.filter((pl.col(f"{s}_r") < 500) & (others >= 500)).height / n, 4))
by_vert = df.group_by("vert").agg(n=pl.len(), **{k: (pl.col("cls") == k).mean().round(3) for k in ("hit500", "U", "A1", "A2", "A3", "C", "B")}).sort("n", descending=True)

print(f"\n{n:,} таргет-пар, {df['user_id'].n_unique():,} юзеров (W{W})\nклассы: {share}\nглубина: {depth}\nисточники: {only}")
print(by_vert)
record("diag", "retrieval misses W1", {"src": SRC, "deep": DEEP}, classes=share, depth=depth, sources=only, sec=round(time.time() - t0, 1))
