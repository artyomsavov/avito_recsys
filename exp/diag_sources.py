"""Этап 2: дают ли co-visitation и BM25 таргеты сверх cr + s24 + u0 (W1, 1 партиция).

Оба источника — двухшаговое произведение seed-ы × H1 × H2, как у блуждания, но с другими весами:
  covis — item ↔ сессия (юзер × день, последние 30 item сессии), сырые веса событий без нормировки
          (классическая co-visitation: score_j = Σ_i x_i · Σ_сессий w_si · w_sj);
  bm25  — item ↔ юзер, веса BM25 (K1 = 100, B = 0.8, как в implicit): длинные истории и
          популярные item приглушены, вместо деления на степень как в u0.
Старые ранги — из exp/cache/diag_w1_p1.parquet (diag_retrieval.py).

    python exp/diag_sources.py   # ~5 мин
"""

import json
import sys
import time

sys.path.insert(0, "exp")
from lib import DAY, T_TRAIN, build_ctx, load_events, sess_pairs  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402

N_OUT, SEEDS, TAU, SESS_CAP, K1, B = 500, 100, 7.0, 30, 100.0, 0.8
OLD = ("cr", "s24", "u0")

t0 = time.time()
lap = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
d = pl.read_parquet("exp/cache/diag_w1_p1.parquet")
EV = load_events((0,))
cx = build_ctx(EV, T_TRAIN, 28, 3.0, 0.3)
sess = sess_pairs(EV, T_TRAIN, 28, 24, 3.0, 0.3)
del EV
users = d.select("user_id").unique().sort("user_id").with_row_index("ui")
imap = pl.concat([cx.pairs.select("item_id"), cx.universe]).unique().sort("item_id").with_row_index("ii")
ni, nu = imap.height, users.height
allowed = np.zeros(ni, bool)
allowed[imap.join(cx.universe, on="item_id")["ii"].to_numpy()] = True
lap(f"{nu:,} юзеров, {ni:,} item")

sd = (cx.pairs.join(users, on="user_id").join(imap, on="item_id")
      .with_columns(x=pl.col("w") * (-(T_TRAIN - pl.col("ts")) / (TAU * DAY)).exp())
      .sort("x", descending=True).group_by("user_id", maintain_order=True).head(SEEDS))
X = sp.csr_matrix((sd["x"].to_numpy().astype(np.float32), (sd["ui"].to_numpy(), sd["ii"].to_numpy())), shape=(nu, ni))
X = sp.diags(1 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1e-9)).astype(np.float32) @ X
seen = cx.seen.join(users, on="user_id").join(imap, on="item_id")
seen_key = np.sort(seen["ui"].to_numpy().astype(np.int64) * ni + seen["ii"].to_numpy())


def score(H1, H2, name, batch=256):
    """top-N_OUT новых item из universe на юзера для seed-ов X через H1 @ H2."""
    out = []
    for s in range(0, nu, batch):
        S = ((X[s : s + batch] @ H1) @ H2).tocoo()
        r, c, v = S.row.astype(np.int64) + s, S.col.astype(np.int64), S.data
        k = r * ni + c
        pos = np.minimum(np.searchsorted(seen_key, k), len(seen_key) - 1)
        m = allowed[c] & (seen_key[pos] != k) & (v > 0)
        df = pl.DataFrame({"ui": r[m].astype(np.uint32), "ii": c[m].astype(np.uint32), "s": v[m]})
        out.append(df.sort(["ui", "s"], descending=[False, True]).group_by("ui", maintain_order=True).head(N_OUT)
                   .with_columns(pl.int_range(pl.len()).over("ui").cast(pl.Int32).alias(f"{name}_r")))
    res = pl.concat(out).join(users, on="ui").join(imap, on="ii").select("user_id", "item_id", f"{name}_r")
    lap(f"{name}: {res.height:,} кандидатов, {res.height / nu:.0f} на юзера")
    return res


# covis: item × сессия (последние SESS_CAP item каждой сессии), сырые веса
ss = (sess.join(imap, on="item_id").sort("ts", descending=True).group_by("user_id", "b", maintain_order=True).head(SESS_CAP)
      .with_columns(si=pl.struct("user_id", "b").rank("dense") - 1))
Sm = sp.csr_matrix((ss["w"].to_numpy().astype(np.float32), (ss["si"].to_numpy(), ss["ii"].to_numpy())), shape=(int(ss["si"].max()) + 1, ni))
covis = score(Sm.T.tocsr(), Sm, "covis")
del Sm, ss

# bm25: item × юзер, веса BM25 (документ — item, термы — юзеры)
pr = cx.pairs.join(imap, on="item_id").with_columns(uj=pl.col("user_id").rank("dense") - 1)
M = sp.csr_matrix((pr["w"].to_numpy().astype(np.float32), (pr["ii"].to_numpy(), pr["uj"].to_numpy())), shape=(ni, int(pr["uj"].max()) + 1)).tocoo()
idf = (np.log(M.shape[0]) - np.log1p(np.bincount(M.col, minlength=M.shape[1]))).astype(np.float32)
ln = (1 - B) + B * np.asarray(M.sum(axis=1)).ravel() / max(np.asarray(M.sum(axis=1)).mean(), 1e-9)
M.data = M.data * (K1 + 1) / (K1 * ln[M.row] + M.data) * idf[M.col]
M = M.tocsr()
bm25 = score(M, M.T.tocsr(), "bm25")
del M, pr

df = d.join(covis, on=["user_id", "item_id"], how="left").join(bm25, on=["user_id", "item_id"], how="left")
n = df.height
old = pl.min_horizontal(*[pl.col(f"{s}_r") for s in OLD]).fill_null(10**6)
res = {"old_pool500": round(df.filter(old < 500).height / n, 4)}
for s in ("covis", "bm25"):
    new = pl.col(f"{s}_r").is_not_null()
    res[s] = dict(pool500=round(df.filter(new).height / n, 4),
                  gain=round(df.filter(new & (old >= 500)).height / n, 4),
                  from_C=round(df.filter(new & (pl.col("cls") == "C")).height / n, 4),
                  from_B=round(df.filter(new & (pl.col("cls") == "B")).height / n, 4))
both = (pl.col("covis_r").is_not_null() | pl.col("bm25_r").is_not_null()) & (old >= 500)
res["both_gain"] = round(df.filter(both).height / n, 4)
print(json.dumps(res, ensure_ascii=False, indent=1))
row = {"group": "diag", "name": "covis + bm25 vs cr+s24+u0 (W1, top-500)", "cfg": {"n_out": N_OUT, "sess_cap": SESS_CAP, "k1": K1, "b": B},
       **res, "sec": round(time.time() - t0, 1), "ts": time.strftime("%H:%M:%S")}
with open("exp/results.jsonl", "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
