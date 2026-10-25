"""Экспериментальный стенд: таргет, графовые блуждания, признаки, ранкер, метрика.

Одна схема валидации для всех экспериментов:
  * тест:   признаки на T, таргет — новые контакты в [T+12h, конец train);
  * обучение ранкера: признаки на T-7d, метки — новые контакты в [T-7d+12h, T);
  * early stopping — по 10% обучающих юзеров (тест не участвует в подборе).
"""

import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import scipy.sparse as sp

RAW = "data/raw"
DAY = 86_400_000
HOUR = 3_600_000
T_TEST = 1_775_595_600_000  # 2026-04-07T21:00Z — фактический порог local_eval.csv
T_TRAIN = T_TEST - 7 * DAY
T_END = 1_776_211_200_000  # конец train
GAP = 12 * HOUR
K = 160
SHOW = 7
CONTACT = [0, 2, 4, 5, 6, 8, 9, 11, 14, 15, 16]
BUCKETS = (("v0", (0,)), ("v2", (2,)), ("v3", (3,)), ("v4", (4,)), ("v57", (5, 7)))
QUOTA = 10_000


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- данные
def load_events(parts=(0,)):
    return pl.concat(
        [pl.read_parquet(f"{RAW}/train_data/part_{p:03d}.parquet") for p in parts]
    ).with_columns(pl.col("eid").cast(pl.UInt8))


def load_items(item_ids):
    """Атрибуты только для встречающихся item (int16/32 вместо int64)."""
    return (
        pl.scan_parquet(f"{RAW}/item_features.parquet")
        .join(pl.LazyFrame({"item_id": item_ids}), on="item_id", how="semi")
        .select(
            "item_id",
            pl.col("vertical_id").cast(pl.UInt8).alias("vert"),
            pl.col("category_ext_y").cast(pl.UInt8).alias("cat"),
            pl.col("region_id_y").cast(pl.UInt8).alias("reg"),
            pl.col("loc_id_y").cast(pl.UInt16).alias("loc"),
            *[pl.col(f"sid_{k}_y").cast(pl.UInt16).alias(f"s{k}") for k in range(4)],
        )
        .collect()
    )


# ---------------------------------------------------------------- таргет (логика организаторов)
def build_target(ev, items, c, end):
    """Повторяет prepare_local_eval.py для порога c, окно таргета [c+12h, end)."""
    pre = ev.filter(pl.col("timestamp") < c)
    seen = pre.select("user_id", "item_id").unique()
    items_2u = seen.group_by("item_id").len().filter(pl.col("len") >= 2).select("item_id")
    cand = (
        ev.filter(
            (pl.col("timestamp") >= c + GAP)
            & (pl.col("timestamp") < end)
            & pl.col("eid").is_in(CONTACT)
        )
        .join(items_2u, on="item_id", how="inner")
        .join(seen, on=["user_id", "item_id"], how="anti")
    )
    eligible = cand.select("user_id").unique().sort("user_id")
    uv = (
        seen.join(items.select("item_id", pl.col("vert").cast(pl.UInt32)), on="item_id")
        .group_by("user_id", "vert")
        .agg(pl.len().alias("n"))
        .join(eligible, on="user_id")
    )
    total = uv.group_by("user_id").agg(pl.col("n").sum().alias("tot"))
    parts = []
    for name, verts in BUCKETS:
        focused = (
            uv.filter(pl.col("vert").is_in(verts))
            .group_by("user_id")
            .agg(pl.col("n").sum().alias("nb"))
            .join(total, on="user_id")
            .filter(pl.col("nb") / pl.col("tot") >= 0.9)
            .select("user_id")
            .unique()
            .sort("user_id")
        )
        s = focused.sample(n=min(QUOTA, focused.height), seed=42 + sum(verts))
        parts.append(s.with_columns(pl.lit(name).alias("bucket")))
    pool = eligible.join(pl.concat([p.select("user_id") for p in parts]).unique(), on="user_id", how="anti").sort("user_id")
    for i in range(5):
        if pool.height == 0:
            break
        s = pool.sample(n=min(QUOTA, pool.height), seed=100 + i)
        parts.append(s.with_columns(pl.lit(f"h{i}").alias("bucket")))
        pool = pool.join(s, on="user_id", how="anti").sort("user_id")
    users = pl.concat(parts)
    tgt = (
        cand.join(users.select("user_id"), on="user_id")
        .sort(["user_id", "timestamp"])
        .group_by("user_id", maintain_order=True)
        .agg(pl.col("item_id").unique(maintain_order=True).head(K))
        .explode("item_id")
    )
    return tgt, users


# ---------------------------------------------------------------- контекст на пороге c
@dataclass
class Ctx:
    c: int
    pairs: pl.DataFrame  # (user_id, item_id, w, ts, contact) — окно истории
    seen: pl.DataFrame  # (user_id, item_id) любые события до c (для novelty)
    universe: pl.DataFrame  # item_id с >=2 юзерами до c (фильтр таргета)
    istat: pl.DataFrame  # статистики item на c
    extra: dict = field(default_factory=dict)


def event_weight(w_contact, w_show):
    return (
        pl.when(pl.col("eid").is_in(CONTACT))
        .then(w_contact)
        .when(pl.col("eid") == SHOW)
        .then(w_show)
        .otherwise(1.0)
        .cast(pl.Float32)
    )


def build_ctx(ev, c, hist_days, w_contact=3.0, w_show=0.3, use_show=True):
    pre = ev.filter(pl.col("timestamp") < c)
    seen = pre.select("user_id", "item_id").unique()
    universe = seen.group_by("item_id").len().filter(pl.col("len") >= 2).select("item_id")
    win = pre.filter(pl.col("timestamp") >= c - hist_days * DAY)
    if not use_show:
        win = win.filter(pl.col("eid") != SHOW)
    win = win.with_columns(event_weight(w_contact, w_show).alias("w"))
    pairs = win.group_by("user_id", "item_id").agg(
        pl.col("w").max(),
        pl.col("timestamp").max().alias("ts"),
        pl.col("eid").is_in(CONTACT).any().alias("contact"),
    )
    # статистики item по всем событиям до c (окно 14 дней для динамики)
    d = pre.filter(pl.col("timestamp") >= c - 14 * DAY).with_columns(
        ((c - pl.col("timestamp")) // DAY).alias("ago"), (pl.col("eid") == SHOW).alias("show")
    )
    istat = d.group_by("item_id").agg(
        *[
            ((pl.col("ago") < k) & ~pl.col("show")).sum().cast(pl.UInt32).alias(f"ev{k}")
            for k in (1, 3, 7)
        ],
        ((pl.col("ago") >= 7) & ~pl.col("show")).sum().cast(pl.UInt32).alias("ev7p"),
        ((pl.col("ago") < 7) & pl.col("show")).sum().cast(pl.UInt32).alias("sh7"),
        ((pl.col("ago") >= 7) & pl.col("show")).sum().cast(pl.UInt32).alias("sh7p"),
        ((pl.col("ago") < 7) & pl.col("eid").is_in(CONTACT)).sum().cast(pl.UInt32).alias("ct7"),
        pl.col("user_id").n_unique().cast(pl.UInt32).alias("nu14"),
        ((c - pl.col("timestamp").max()) / DAY).cast(pl.Float32).alias("last_ago"),
    )
    first = pre.group_by("item_id").agg(((c - pl.col("timestamp").min()) / DAY).cast(pl.Float32).alias("age"))
    istat = universe.join(istat, on="item_id", how="left").join(first, on="item_id", how="left")
    return Ctx(c, pairs, seen, universe, istat)


# ---------------------------------------------------------------- графовое блуждание
from concurrent.futures import ThreadPoolExecutor


def dense_ids(df, col, name):
    """Плотные индексы 0..n-1 для значений столбца (через join с таблицей уникальных)."""
    m = df.select(col).unique().sort(col).with_row_index(name)
    return df.join(m, on=col), m


def top_per_row(m, n):
    """CSR -> (row, col, val) с n лучшими значениями в каждой строке."""
    m = m.tocsr()
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    order = np.lexsort((-m.data, rows))
    r, c, v = rows[order], m.indices[order], m.data[order]
    start = np.searchsorted(r, np.arange(m.shape[0]))
    rank = np.arange(len(r)) - start[r]
    keep = rank < n
    return r[keep], c[keep], v[keep]


def entity_frame(pairs, kind, items=None):
    """(e, item_id, w, ts): сущность = юзер × набор атрибутов item (kind — кортеж колонок items),
    "user" — сам юзер; если в pairs есть колонка b (бакет времени), она входит в ключ (сессии)."""
    attrs = () if kind == "user" else tuple(kind)
    df = pairs.join(items.select("item_id", *attrs), on="item_id") if attrs else pairs
    keys = ["user_id", *attrs] + (["b"] if "b" in pairs.columns else [])
    return df.select(pl.struct(keys).hash(seed=7).alias("e"), "item_id", "w", "ts")


def sess_pairs(ev, c, hist_days, bucket_hours, w_contact, w_show, use_show=True):
    win = ev.filter((pl.col("timestamp") < c) & (pl.col("timestamp") >= c - hist_days * DAY))
    if not use_show:
        win = win.filter(pl.col("eid") != SHOW)
    return (
        win.with_columns(event_weight(w_contact, w_show).alias("w"), ((c - pl.col("timestamp")) // (bucket_hours * HOUR)).alias("b"))
        .group_by("user_id", "item_id", "b")
        .agg(pl.col("w").max(), pl.col("timestamp").max().alias("ts"))
    )


def walk(ctx, users, ef, n_out, a1=1.0, a2=1.0, beta=0.5, tau=7.0, max_seeds=100,
         cap_item=300, cap_ent=200, frontier=0, batch=256, threads=8, name="w", seeds=None):
    """Блуждание seed-item -> сущность -> item.

    H1[i, e] = w_ei / deg_i^a1 (переход от item к его сущностям, top cap_item свежих),
    H2[e, j] = w_ej / deg_e^a2 (от сущности к её item, top cap_ent свежих),
    score_uj = sum_i x_ui (H1 H2)_ij / pop_j^beta, x_ui = w_ui * exp(-age/tau).
    Возвращает top-n_out новых (не виденных) item из universe на юзера.
    """
    c = ctx.c
    ef, imap = dense_ids(ef, "item_id", "ii")
    ef, _ = dense_ids(ef, "e", "ei")
    ni, ne = imap.height, int(ef["ei"].max()) + 1
    h1 = ef.sort("ts", descending=True).group_by("ii", maintain_order=True).head(cap_item)
    h2 = ef.sort("ts", descending=True).group_by("ei", maintain_order=True).head(cap_ent)

    def mat(df, r, cc, shape, a):
        m = sp.csr_matrix((df["w"].to_numpy().astype(np.float32), (df[r].to_numpy(), df[cc].to_numpy())), shape=shape)
        deg = np.asarray(m.sum(axis=1)).ravel()
        return sp.diags(np.power(np.maximum(deg, 1e-9), -a).astype(np.float32)) @ m

    H1 = mat(h1, "ii", "ei", (ni, ne), a1)
    H2 = mat(h2, "ei", "ii", (ne, ni), a2)
    pop = np.diff(H2.tocsc().indptr).astype(np.float32)
    colw = np.power(np.maximum(pop, 1.0), -beta).astype(np.float32)
    allowed = np.zeros(ni, bool)
    allowed[imap.join(ctx.universe, on="item_id", how="semi")["ii"].to_numpy()] = True
    colw[~allowed] = 0.0
    Dcol = sp.diags(colw)

    umap = users.select("user_id").unique().sort("user_id").with_row_index("ui")
    sd = (
        (ctx.pairs if seeds is None else seeds)
        .join(umap, on="user_id")
        .join(imap, on="item_id")
        .with_columns((pl.col("w") * (-(c - pl.col("ts")) / (tau * DAY)).exp()).alias("x"))
        .sort("x", descending=True)
        .group_by("user_id", maintain_order=True)
        .head(max_seeds)
    )
    X = sp.csr_matrix((sd["x"].to_numpy().astype(np.float32), (sd["ui"].to_numpy(), sd["ii"].to_numpy())), shape=(umap.height, ni))
    X = sp.diags((1.0 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1e-9)).astype(np.float32)) @ X
    seen = ctx.seen.join(umap, on="user_id").join(imap, on="item_id").select("ui", "ii")
    uid, iid = umap["user_id"].to_numpy(), imap["item_id"].to_numpy()

    def one(s):
        F = X[s : s + batch] @ H1
        if frontier:
            r, cc, v = top_per_row(F, frontier)
            F = sp.csr_matrix((v, (r, cc)), shape=F.shape)
        S = ((F @ H2) @ Dcol).tocoo()
        df = pl.DataFrame({"ui": (S.row + s).astype(np.uint32), "ii": S.col.astype(np.uint32), "s": S.data})
        df = df.filter(pl.col("s") > 0).join(seen, on=["ui", "ii"], how="anti")
        return df.sort(["ui", "s"], descending=[False, True]).group_by("ui", maintain_order=True).head(n_out)

    with ThreadPoolExecutor(threads) as ex:
        res = pl.concat(list(ex.map(one, range(0, umap.height, batch))))
    res = res.with_columns(pl.int_range(pl.len()).over("ui").cast(pl.UInt16).alias("r"))
    return res.select(
        pl.Series("user_id", uid[res["ui"].to_numpy()]).cast(pl.UInt32),
        pl.Series("item_id", iid[res["ii"].to_numpy()]).cast(pl.UInt32),
        pl.col("s").alias(f"{name}_s"),
        pl.col("r").alias(f"{name}_r"),
    )


# ---------------------------------------------------------------- метрики
def recall_at_k(pred, tgt, score, k=K):
    """pred: (user_id, item_id, score); tgt: (user_id, item_id). Возвращает (mean, per_user)."""
    top = (
        pred.sort(["user_id", score, "item_id"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True)
        .head(k)
    )
    hits = top.join(tgt, on=["user_id", "item_id"], how="semi").group_by("user_id").len()
    per = (
        tgt.group_by("user_id")
        .len()
        .rename({"len": "n"})
        .join(hits, on="user_id", how="left")
        .with_columns((pl.col("len").fill_null(0) / pl.col("n")).alias("rec"))
        .sort("user_id")
    )
    return per["rec"].mean(), per


def pool_recall(pool, tgt):
    return tgt.join(pool.select("user_id", "item_id").unique(), on=["user_id", "item_id"], how="semi").height / tgt.height


def boot_ci(per_a, per_b=None, n=1000, seed=0):
    """95% CI среднего recall (или разницы b-a по тем же юзерам)."""
    rng = np.random.default_rng(seed)
    a = per_a["rec"].to_numpy()
    d = a if per_b is None else per_b["rec"].to_numpy() - a
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# ---------------------------------------------------------------- кандидаты и признаки
def pop_source(ctx, users, n_out, days=7):
    """Глобально популярные по контактам за последние days дней (минус виденное)."""
    c = ctx.c
    top = (
        ctx.pairs.filter(pl.col("contact") & (pl.col("ts") >= c - days * DAY))
        .group_by("item_id").len()
        .join(ctx.universe, on="item_id", how="semi")
        .sort("len", descending=True).head(n_out * 3)
    )
    df = users.select("user_id").join(top, how="cross").join(ctx.seen, on=["user_id", "item_id"], how="anti")
    return (
        df.sort(["user_id", "len"], descending=[False, True]).group_by("user_id", maintain_order=True).head(n_out)
        .with_columns(pl.int_range(pl.len()).over("user_id").cast(pl.UInt16).alias("pop_r"))
        .select("user_id", "item_id", pl.col("len").cast(pl.Float32).alias("pop_s"), "pop_r")
    )


def merge_sources(srcs):
    out = srcs[0]
    for s in srcs[1:]:
        out = out.join(s, on=["user_id", "item_id"], how="full", coalesce=True)
    return out


AFF_KEYS = {"cat": ["cat"], "reg": ["reg"], "loc": ["loc"], "s0": ["s0"], "s01": ["s0", "s1"], "s012": ["s0", "s1", "s2"], "vert": ["vert"]}


def oracle_future(ev, c, days=7):
    """X1: будущая активность item в [c, c+days) — верхняя граница для прогноза динамики."""
    f = ev.filter((pl.col("timestamp") >= c) & (pl.col("timestamp") < c + days * DAY) & (pl.col("eid") != SHOW))
    return f.group_by("item_id").agg(pl.len().cast(pl.UInt32).alias("or_ev"), pl.col("eid").is_in(CONTACT).sum().cast(pl.UInt32).alias("or_ct"))


def bucket_pop_source(ctx, users, items, n_out, days=7):
    """X2: популярное по контактам в любимой (категория, регион) юзера."""
    c = ctx.c
    rec = ctx.pairs.filter(pl.col("contact") & (pl.col("ts") >= c - days * DAY)).join(items.select("item_id", "cat", "reg"), on="item_id")
    top = (rec.group_by("cat", "reg", "item_id").len().join(ctx.universe, on="item_id", how="semi")
           .sort(["cat", "reg", "len"], descending=[False, False, True]).group_by("cat", "reg", maintain_order=True).head(n_out))
    fav = (ctx.pairs.join(users, on="user_id", how="semi").join(items.select("item_id", "cat", "reg"), on="item_id")
           .group_by("user_id", "cat", "reg").agg(pl.col("w").sum()).sort(["user_id", "w"], descending=[False, True])
           .group_by("user_id", maintain_order=True).head(1).select("user_id", "cat", "reg"))
    df = fav.join(top, on=["cat", "reg"]).join(ctx.seen, on=["user_id", "item_id"], how="anti")
    return (df.sort(["user_id", "len"], descending=[False, True]).group_by("user_id", maintain_order=True).head(n_out)
            .with_columns(pl.int_range(pl.len()).over("user_id").cast(pl.UInt16).alias("bp_r"))
            .select("user_id", "item_id", pl.col("len").cast(pl.Float32).alias("bp_s"), "bp_r"))


def featurize(cands, ctx, items, aff=True, dyn=True, oracle=None):
    c = ctx.c
    users = cands.select("user_id").unique()
    up = ctx.pairs.join(users, on="user_id", how="semi").join(items, on="item_id", how="left")
    ufeat = up.group_by("user_id").agg(
        pl.len().cast(pl.UInt32).alias("u_n"), pl.col("w").sum().alias("u_w"), pl.col("contact").sum().cast(pl.UInt32).alias("u_nc"),
        ((c - pl.col("ts").max()) / DAY).cast(pl.Float32).alias("u_last"), pl.col("cat").n_unique().cast(pl.UInt16).alias("u_ncat"),
    )
    ideg = ctx.pairs.group_by("item_id").agg(pl.len().cast(pl.UInt32).alias("i_deg"), pl.col("contact").sum().cast(pl.UInt32).alias("i_nc"))
    df = cands.join(items, on="item_id", how="left").join(ufeat, on="user_id", how="left").join(ideg, on="item_id", how="left")
    if dyn:
        df = df.join(ctx.istat, on="item_id", how="left").with_columns(
            ((pl.col("ev7") + 1) / (pl.col("ev7p") + 1)).cast(pl.Float32).alias("i_trend"),
            ((pl.col("sh7") + 1) / (pl.col("sh7p") + 1)).cast(pl.Float32).alias("i_shtrend"),
        )
    if oracle is not None:
        df = df.join(oracle, on="item_id", how="left")
    if aff:
        for name, keys in AFF_KEYS.items():
            a = up.group_by("user_id", *keys).agg(
                pl.col("w").sum().alias("_w"), pl.col("ts").max().alias("_ts"), pl.col("contact").sum().cast(pl.UInt16).alias(f"a_{name}_nc")
            )
            df = df.join(a, on=["user_id", *keys], how="left").with_columns(
                (pl.col("_w") / pl.col("u_w")).cast(pl.Float32).alias(f"a_{name}_sh"),
                ((c - pl.col("_ts")) / DAY).cast(pl.Float32).alias(f"a_{name}_ago"),
            ).drop("_w", "_ts")
    return df


CAT_FEATS = ["vert", "cat", "reg", "loc", "s0"]
DROP = {"user_id", "item_id", "target", "s1", "s2", "s3"}


def label(df, tgt):
    return df.join(tgt.with_columns(pl.lit(1, pl.Int8).alias("target")), on=["user_id", "item_id"], how="left").with_columns(
        pl.col("target").fill_null(0)
    )


def train_ranker(tr, neg=200, holdout=0.1, iters=2000, lr=0.08, depth=6, loss="YetiRank", seed=42, task="CPU", verbose=200):
    from catboost import CatBoostRanker, Pool

    pos_users = tr.filter(pl.col("target") == 1).select("user_id").unique()
    tr = tr.join(pos_users, on="user_id", how="semi")
    tr = tr.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("user_id").alias("_k")).filter(
        (pl.col("target") == 1) | (pl.col("_k") < neg)
    ).drop("_k")
    uu = pos_users.sort("user_id").sample(fraction=1.0, shuffle=True, seed=seed)["user_id"]
    hold = set(uu[: int(len(uu) * holdout)].to_list())
    feats = [c for c in tr.columns if c not in DROP]

    def pool(d):
        d = d.sort(["user_id", "item_id"])
        X = d.select(feats).to_pandas()
        for cf in CAT_FEATS:
            if cf in X:
                X[cf] = X[cf].fillna(-1).astype("int64")
        return Pool(X, d["target"].to_numpy(), group_id=d["user_id"].to_numpy(), cat_features=[f for f in CAT_FEATS if f in feats])

    is_h = pl.col("user_id").is_in(list(hold))
    m = CatBoostRanker(loss_function=loss, iterations=iters, learning_rate=lr, depth=depth, random_seed=seed, task_type=task,
                       od_type="Iter", od_wait=150, use_best_model=True, verbose=verbose, thread_count=-1)
    m.fit(pool(tr.filter(~is_h)), eval_set=pool(tr.filter(is_h)))
    return m, feats, pool


def predict(m, feats, df, chunk=2_000_000):
    out = []
    for i in range(0, df.height, chunk):
        part = df.slice(i, chunk)
        X = part.select(feats).to_pandas()
        for cf in CAT_FEATS:
            if cf in X:
                X[cf] = X[cf].fillna(-1).astype("int64")
        out.append(part.select("user_id", "item_id").with_columns(pl.Series("score", m.predict(X))))
    return pl.concat(out)
