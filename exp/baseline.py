"""Старый пайплайн (ALS + covisit depth-3 + 13 признаков + CatBoost) в новой схеме W2 -> W1.

    python exp/baseline.py [novelty]
"""

import sys
import time

sys.path.insert(0, "exp")
import implicit  # noqa: E402
from run import EV, ITEMS, WIN, record, target  # noqa: E402

from lib import *  # noqa: E402,F403

NOVELTY = "novelty" in sys.argv


def old_ctx(c):
    win = EV.filter((pl.col("timestamp") < c) & (pl.col("timestamp") >= c - 28 * DAY) & (pl.col("eid") != SHOW))
    agg = (
        win.with_columns(pl.when(pl.col("eid").is_in(CONTACT)).then(5).otherwise(1).alias("w"))
        .group_by("user_id", "item_id").agg(pl.col("w").sum().cast(pl.Float32), pl.col("timestamp").max().alias("ts"))
        .sort(["user_id", "ts"], descending=[False, True]).group_by("user_id", maintain_order=True).head(100)
    )
    iu = agg.group_by("item_id").agg(pl.len().alias("n_users"), pl.col("w").sum().alias("pop"))
    allowed = iu.filter(pl.col("n_users") >= 2).join(ITEMS.filter(pl.col("vert").is_in([0, 2, 3, 4, 5, 7])), on="item_id", how="semi")
    return win, agg, iu, allowed.select("item_id")


def als_cands(agg, allowed, users, n=400):
    umap = agg.select("user_id").unique().sort("user_id").with_row_index("ui")
    imap = agg.select("item_id").unique().sort("item_id").with_row_index("ii")
    a = agg.join(umap, on="user_id").join(imap, on="item_id")
    M = sp.csr_matrix((a["w"].to_numpy(), (a["ui"].to_numpy(), a["ii"].to_numpy())), shape=(umap.height, imap.height))
    m = implicit.cpu.als.AlternatingLeastSquares(factors=32, regularization=0.1, iterations=15, random_state=42, num_threads=16)
    m.fit(M, show_progress=False)
    allow_ii = imap.join(allowed, on="item_id", how="semi")["ii"].to_numpy()
    uu = umap.join(users, on="user_id", how="semi")
    ids, sc = m.recommend(uu["ui"].to_numpy(), M[uu["ui"].to_numpy()], N=n, filter_already_liked_items=False, items=allow_ii)
    iid = imap["item_id"].to_numpy()
    return pl.DataFrame({
        "user_id": np.repeat(uu["user_id"].to_numpy(), n), "item_id": iid[ids.ravel()].astype(np.uint32),
        "als_s": sc.ravel().astype(np.float32), "als_r": np.tile(np.arange(n), uu.height).astype(np.uint16),
    }).filter(pl.col("als_s") > -1e30)


def covisit_cands(win, agg, allowed, users, n=400):
    df = win.select("user_id", "item_id", "timestamp").sort(["user_id", "timestamp"])
    frames = []
    for k in (1, 2, 3):
        s = df.with_columns(pl.col("item_id").shift(-k).over("user_id").alias("nx"), pl.col("timestamp").shift(-k).over("user_id").alias("tn"))
        s = s.filter(pl.col("nx").is_not_null() & (pl.col("nx") != pl.col("item_id")) & (pl.col("tn") - pl.col("timestamp") <= DAY))
        frames.append(s.select("user_id", pl.min_horizontal("item_id", "nx").alias("a"), pl.max_horizontal("item_id", "nx").alias("b")))
    pc = pl.concat(frames).unique().group_by("a", "b").len().filter(pl.col("len") >= 3)
    d = pl.concat([pc.select(pl.col("a").alias("src"), pl.col("b").alias("dst"), "len"), pc.select(pl.col("b").alias("src"), pl.col("a").alias("dst"), "len")])
    cv = d.sort(["src", "len"], descending=[False, True]).group_by("src", maintain_order=True).head(50)
    seeds = agg.join(users, on="user_id", how="semi").sort(["user_id", "ts"], descending=[False, True]).group_by("user_id", maintain_order=True).head(15)
    sc = seeds.join(cv, left_on="item_id", right_on="src").join(allowed, left_on="dst", right_on="item_id", how="semi")
    sc = sc.group_by("user_id", pl.col("dst").alias("item_id")).agg(pl.col("len").sum().cast(pl.Float32).alias("cov_s"))
    return (sc.sort(["user_id", "cov_s", "item_id"], descending=[False, True, False]).group_by("user_id", maintain_order=True).head(n)
            .with_columns(pl.int_range(pl.len()).over("user_id").cast(pl.UInt16).alias("cov_r")))


def window(w):
    c = WIN[w][0]
    users = target(w).select("user_id").unique()
    win, agg, iu, allowed = old_ctx(c)
    cands = als_cands(agg, allowed, users).join(covisit_cands(win, agg, allowed, users), on=["user_id", "item_id"], how="full", coalesce=True)
    if NOVELTY:
        seen = EV.filter(pl.col("timestamp") < c).select("user_id", "item_id").unique().join(users, on="user_id", how="semi")
        cands = cands.join(seen, on=["user_id", "item_id"], how="anti")
    uv = agg.join(ITEMS.select("item_id", "vert"), on="item_id").group_by("user_id", "vert").agg(pl.col("w").sum())
    utop = uv.sort(["user_id", "w"], descending=[False, True]).group_by("user_id", maintain_order=True).head(1).select("user_id", pl.col("vert").alias("utv"))
    df = (cands.with_columns(pl.col("als_r").fill_null(999), pl.col("cov_r").fill_null(999), pl.col("als_s").fill_null(0), pl.col("cov_s").fill_null(0))
          .join(ITEMS.select("item_id", "vert", "cat", "reg", "loc"), on="item_id", how="left")
          .join(iu, on="item_id", how="left").join(utop, on="user_id", how="left")
          .with_columns((pl.col("vert") == pl.col("utv")).cast(pl.Int8).alias("utv_match")).drop("utv"))
    return label(df, target(w))


if __name__ == "__main__":
    t0 = time.time()
    tr, va = window(2), window(1)
    log(f"train rows {tr.height}, val rows {va.height}")
    m, feats, _ = train_ranker(tr, neg=10_000, iters=500, lr=0.05, verbose=100)
    pred = predict(m, feats, va)
    tgt = target(1)
    rec, per = recall_at_k(pred, tgt, "score")
    per.write_parquet(f"exp/cache/per_baseline{'_nov' if NOVELTY else ''}.parquet")
    d, lo, hi = boot_ci(per)
    record("baseline", "old pipeline" + (" + novelty filter" if NOVELTY else ""), {"novelty": NOVELTY},
           recall=round(rec, 5), ci=[round(lo, 5), round(hi, 5)], pool=round(pool_recall(va, tgt), 4), sec=round(time.time() - t0, 1),
           als_only=round(recall_at_k(va.filter(pl.col("als_r") < 999), tgt, "als_s")[0], 5),
           cov_only=round(recall_at_k(va.filter(pl.col("cov_r") < 999), tgt, "cov_s")[0], 5))
