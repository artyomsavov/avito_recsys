"""Этап 1: почему таргеты W1 не попадают в пул кандидатов.

Каждая таргет-пара (юзер, item) получает класс по первому сработавшему правилу:
  hit500  — в top-500 хотя бы одного источника (это и есть потолок пула final.ipynb)
  U       — у юзера нет событий в окне истории (нет seed-ов, блужданию не с чего начать)
  A1      — у item нет ни одного события до порога (холодный)
  A2      — события есть, но у < 2 юзеров (не входит в universe, блуждание его отбрасывает)
  A3      — в universe, но нет событий за 28 дней истории (нет в графе)
  C       — блуждание доходит, но ранг ≥ 500 во всех источниках (подкласс по глубине)
  B       — в графе, но не достигнут даже в top-5000

    python exp/diag_retrieval.py      # 1 партиция, ~2 мин
    python exp/diag_retrieval.py 2 2 2000  # 2 партиции, 2 потока, top-2000: как меняются классы с объёмом
"""

import json
import sys
import time

sys.path.insert(0, "exp")
from lib import T_END, T_TEST, T_TRAIN, build_ctx, build_target, entity_frame, load_events, load_items, sess_pairs, walk  # noqa: E402

import polars as pl  # noqa: E402

NP, THREADS, DEEP = (int(a) for a in (sys.argv[1:] + ["1", "6", "5000"][len(sys.argv) - 1:]))  # партиций, потоков, глубина
W = 1
WIN = {0: (T_TEST, T_END), 1: (T_TRAIN, T_TEST)}
HIST, W_CONTACT, W_SHOW = 28, 3.0, 0.3
SRC = {"cr": ("cat", "reg"), "s24": "sess", "u0": "user"}  # как в final.ipynb, β = 0

t0 = time.time()
lap = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
EV = load_events(range(NP))
ITEMS = load_items(EV["item_id"].unique())
tgt, _ = build_target(EV, ITEMS, *WIN[W])
cx = build_ctx(EV, WIN[W][0], HIST, W_CONTACT, W_SHOW)
c, users = cx.c, tgt.select("user_id").unique()
lap(f"{NP} партиций: {EV.height:,} событий, таргет {tgt.height:,} пар")
sess = sess_pairs(EV, c, HIST, 24, W_CONTACT, W_SHOW)
pre_items = EV.filter(pl.col("timestamp") < c).select("item_id").unique()
del EV  # дальше хватает контекста: события — главный расход памяти

ranks = tgt
for name, ent in SRC.items():  # ранг каждой таргет-пары в каждом источнике (top-DEEP, с 0)
    if ent == "sess":
        ef = entity_frame(sess, "user", ITEMS)
    else:
        ef = entity_frame(cx.pairs, ent, ITEMS)
    res = walk(cx, users, ef, DEEP, beta=0.0, name=name, threads=THREADS)
    ranks = ranks.join(res.select("user_id", "item_id", pl.col(f"{name}_r").cast(pl.Int32)), on=["user_id", "item_id"], how="left")
    lap(f"{name}: {res.height:,} строк")
    del res, ef

best = pl.min_horizontal(*[pl.col(f"{n}_r") for n in SRC])
df = ranks.with_columns(best=best).with_columns(
    cls=pl.when(pl.col("best") < 500).then(pl.lit("hit500"))
    .when(~pl.col("user_id").is_in(cx.pairs["user_id"].unique().implode())).then(pl.lit("U"))
    .when(~pl.col("item_id").is_in(pre_items["item_id"].implode())).then(pl.lit("A1"))
    .when(~pl.col("item_id").is_in(cx.universe["item_id"].implode())).then(pl.lit("A2"))
    .when(~pl.col("item_id").is_in(cx.pairs["item_id"].unique().implode())).then(pl.lit("A3"))
    .when(pl.col("best").is_not_null()).then(pl.lit("C"))
    .otherwise(pl.lit("B"))
).join(ITEMS.select("item_id", "vert"), on="item_id", how="left")
df.write_parquet(f"exp/cache/diag_w{W}_p{NP}_d{DEEP}.parquet")

n = df.height
share = {k: round(v / n, 4) for k, v in df.group_by("cls").len().sort("cls").rows()}
depth = {f"pool@{d}": round(df.filter(pl.col("best") < d).height / n, 4) for d in sorted({500, 1000, 2000, DEEP})}
only = {}  # уникальный вклад источника в top-500
for s in SRC:
    others = pl.min_horizontal(*[pl.col(f"{o}_r") for o in SRC if o != s]).fill_null(DEEP)
    only[s] = dict(pool500=round(df.filter(pl.col(f"{s}_r") < 500).height / n, 4),
                   only500=round(df.filter((pl.col(f"{s}_r") < 500) & (others >= 500)).height / n, 4))
by_vert = df.group_by("vert").agg(n=pl.len(), **{k: (pl.col("cls") == k).mean().round(3) for k in ("hit500", "U", "A3", "C", "B")}).sort("n", descending=True)

print(f"\n{n:,} таргет-пар, {df['user_id'].n_unique():,} юзеров (W{W}, {NP} партиций)\nклассы: {share}\nглубина: {depth}\nисточники: {only}")
print(by_vert)
row = {"group": "diag", "name": f"retrieval misses W{W}, {NP} part", "cfg": {"src": SRC, "deep": DEEP, "parts": NP},
       "classes": share, "depth": depth, "sources": only, "sec": round(time.time() - t0, 1), "ts": time.strftime("%H:%M:%S")}
with open("exp/results.jsonl", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
