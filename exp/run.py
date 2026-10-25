"""Раннер экспериментов этапа 1 (одна партиция).

Окна: W0 — тест (порог T), W1 — валидация (T-7d), W2 — обучение ранкера в экспериментах (T-14d).
Все решения этапа 1 принимаются на W1; W0 трогается один раз в конце.

    python exp/run.py <group>   # группы см. GROUPS внизу
"""

import json
import sys
import time

sys.path.insert(0, "exp")
from lib import *  # noqa: E402,F403

WIN = {0: (T_TEST, T_END), 1: (T_TRAIN, T_TEST), 2: (T_TRAIN - 7 * DAY, T_TRAIN)}
CACHE = "exp/cache"
RESULTS = "exp/results.jsonl"

EV = load_events((0,))
ITEMS = pl.read_parquet(f"{CACHE}/items_p0.parquet")


def target(w):
    path = f"{CACHE}/tgt_w{w}_p0.parquet"
    try:
        return pl.read_parquet(path)
    except FileNotFoundError:
        tgt, _ = build_target(EV, ITEMS, *WIN[w])
        tgt.write_parquet(path)
        return tgt


_CTX = {}


def ctx(w, hist_days=28, w_contact=3.0, w_show=0.3, use_show=True):
    key = (w, hist_days, w_contact, w_show, use_show)
    if key not in _CTX:
        _CTX.clear()  # держим в памяти один контекст
        _CTX[key] = build_ctx(EV, WIN[w][0], hist_days, w_contact, w_show, use_show)
    return _CTX[key]


def record(group, name, cfg, **metrics):
    row = {"group": group, "name": name, "cfg": cfg, **metrics, "ts": time.strftime("%H:%M:%S")}
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)


BASE = dict(entity="user", hist_days=28, w_contact=3.0, w_show=0.3, use_show=True, bucket_hours=24,
            a1=1.0, a2=1.0, beta=0.5, tau=7.0, max_seeds=100, cap_item=300, cap_ent=200, n_out=1000)
CTX_KEYS = ("hist_days", "w_contact", "w_show", "use_show")
WALK_KEYS = ("a1", "a2", "beta", "tau", "max_seeds", "cap_item", "cap_ent")


def run_walk(cfg, w=1, name="w"):
    cfg = {**BASE, **cfg}
    cx = ctx(w, **{k: cfg[k] for k in CTX_KEYS})
    users = target(w).select("user_id").unique()
    ent = cfg["entity"]
    if isinstance(ent, str) and ent.startswith("sess"):  # "sess" или "sess:cat"
        pairs = sess_pairs(EV, cx.c, cfg["hist_days"], cfg["bucket_hours"], cfg["w_contact"], cfg["w_show"], cfg["use_show"])
        attrs = tuple(ent.split(":")[1].split(",")) if ":" in ent else "user"
        ef = entity_frame(pairs, attrs, ITEMS)
    else:
        ef = entity_frame(cx.pairs, ent if ent == "user" else tuple(ent), ITEMS)
    return walk(cx, users, ef, cfg["n_out"], name=name, threads=cfg.get("threads", 6), **{k: cfg[k] for k in WALK_KEYS})


_REF = {}


def eval_walk(group, name, cfg, w=1):
    t0 = time.time()
    res = run_walk(cfg, w)
    tgt = target(w)
    rec, per = recall_at_k(res, tgt, "w_s")
    pool = pool_recall(res, tgt)
    pool300 = pool_recall(res.filter(pl.col("w_r") < 300), tgt)
    m = dict(recall=round(rec, 5), pool1000=round(pool, 4), pool300=round(pool300, 4), sec=round(time.time() - t0, 1))
    if group in _REF:
        d, lo, hi = boot_ci(_REF[group], per)
        m.update(delta=round(d, 5), ci=[round(lo, 5), round(hi, 5)])
    else:
        _REF[group] = per
    record(group, name, cfg, **m)
    return per


def grid_walk():
    g = "walk"
    eval_walk(g, "base: user-entity RP3, shows 0.3, 28d", {})
    for ws in (0.0, 0.1, 1.0):
        eval_walk(g, f"w_show={ws}", {"w_show": ws} if ws else {"use_show": False})
    for h in (14, 56):
        eval_walk(g, f"hist_days={h}", {"hist_days": h})
    for b in (0.0, 0.25, 0.75, 1.0):
        eval_walk(g, f"beta={b}", {"beta": b})
    for a1, a2 in ((0.5, 1.0), (1.0, 0.5), (0.0, 1.0), (1.0, 0.0)):
        eval_walk(g, f"a1={a1}, a2={a2}", {"a1": a1, "a2": a2})
    for t in (1.0, 3.0, 30.0, 1000.0):
        eval_walk(g, f"tau={t}", {"tau": t})
    for ms in (20, 50, 300):
        eval_walk(g, f"max_seeds={ms}", {"max_seeds": ms})
    for ci, ce in ((100, 200), (1000, 200), (300, 100), (300, 500)):
        eval_walk(g, f"cap_item={ci}, cap_ent={ce}", {"cap_item": ci, "cap_ent": ce})
    for wc in (1.0, 10.0):
        eval_walk(g, f"w_contact={wc}", {"w_contact": wc})
    eval_walk(g, "entity=cat (user x category)", {"entity": "cat"})
    for bh in (24, 6):
        eval_walk(g, f"entity=sess {bh}h", {"entity": "sess", "bucket_hours": bh})


def sources(w, walks, n_pop=100, n_bpop=0):
    """Кандидаты окна w: несколько блужданий + популярное. walks: {name: cfg}."""
    users = target(w).select("user_id").unique()
    srcs = [run_walk(cfg, w, name=name) for name, cfg in walks.items()]
    cx = ctx(w, **{k: {**BASE, **next(iter(walks.values()))}[k] for k in CTX_KEYS})
    if n_pop:
        srcs.append(pop_source(cx, users, n_pop))
    if n_bpop:
        srcs.append(bucket_pop_source(cx, users, ITEMS, n_bpop))
    return merge_sources(srcs), cx


DYN = ["ev1", "ev3", "ev7", "ev7p", "sh7", "sh7p", "ct7", "nu14", "last_ago", "age", "i_trend", "i_shtrend"]


def feats_cached(w, walks, n_pop=100, n_bpop=100):
    """Все признаки окна w (вкл. оракул or_*) — считаются один раз, дальше из parquet."""
    import hashlib
    key = hashlib.md5(json.dumps([walks, n_pop, n_bpop], sort_keys=True).encode()).hexdigest()[:10]
    path = f"{CACHE}/feat_w{w}_{key}.parquet"
    try:
        return pl.read_parquet(path)
    except FileNotFoundError:
        cands, cx = sources(w, walks, n_pop, n_bpop)
        df = label(featurize(cands, cx, ITEMS, oracle=oracle_future(EV, cx.c)), target(w))
        df.write_parquet(path)
        log(f"W{w}: {df.height} rows, pool recall {pool_recall(df, target(w)):.4f} -> {path}")
        return df


def fill_missing(df):
    """null -> «нет сигнала» с правильной стороны шкалы (CatBoost по умолчанию считает NaN меньше всех значений,
    и «не попал в топ источника» без заполнения неотличим от «ранг 0»)."""
    fills = {}
    for c in df.columns:
        if c.endswith("_r"):
            fills[c] = pl.col(c).fill_null(999)
        elif c.endswith("_ago") or c in ("last_ago", "u_last"):
            fills[c] = pl.col(c).fill_null(999)
        elif c in ("i_trend", "i_shtrend"):
            fills[c] = pl.col(c).fill_null(1.0)
        elif c.endswith(("_s", "_sh", "_nc")) or c in DYN or c.startswith(("u_", "or_", "i_")):
            fills[c] = pl.col(c).fill_null(0)
    return df.with_columns(**fills)


def select(df, use_src=None, drop=(), max_rank=None):
    """Абляции: оставить строки только от источников use_src, выбросить группы признаков, урезать пул до max_rank."""
    srcs = [c[:-2] for c in df.columns if c.endswith("_r")]
    if max_rank is not None:
        walks_r = [f"{s}_r" for s in srcs if s not in ("pop", "bp")]
        df = df.with_columns(**{c: pl.when(pl.col(c) < max_rank).then(pl.col(c)) for c in walks_r})
        df = df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in [f"{s}_r" for s in srcs]]))
    if use_src is not None:
        df = df.filter(pl.any_horizontal([pl.col(f"{s}_r").is_not_null() for s in use_src]))
        df = df.drop([c for s in srcs if s not in use_src for c in (f"{s}_s", f"{s}_r")])
    cols = []
    for g in drop:
        cols += DYN if g == "dyn" else [c for c in df.columns if c.startswith({"aff": "a_", "oracle": "or_"}[g])]
    return fill_missing(df.drop([c for c in cols if c in df.columns]))


def eval_rank(group, name, walks, use_src=None, drop=("oracle",), neg=200, n_pop=100, n_bpop=100, ref=None, wins=(2, 1), save=None, max_rank=None, **rk):
    """Ранкер: обучение на wins[0], оценка на wins[1] (по умолчанию W2 -> W1)."""
    t0 = time.time()
    tr_w, te_w = wins
    tr = select(feats_cached(tr_w, walks, n_pop, n_bpop), use_src, drop, max_rank)
    m, fl, _ = train_ranker(tr, neg=neg, **rk)
    del tr
    te = select(feats_cached(te_w, walks, n_pop, n_bpop), use_src, drop, max_rank)
    pred = predict(m, fl, te)
    tgt = target(te_w)
    rec, per = recall_at_k(pred, tgt, "score")
    met = dict(recall=round(rec, 5), pool=round(pool_recall(te, tgt), 4), iters=m.get_best_iteration(),
               sec=round(time.time() - t0, 1), n_feats=len(fl), rows_per_user=round(te.height / te["user_id"].n_unique()))
    if ref:
        d, lo, hi = boot_ci(pl.read_parquet(ref), per)
        met.update(delta=round(d, 5), ci=[round(lo, 5), round(hi, 5)])
    else:
        d, lo, hi = boot_ci(per)
        met.update(ci=[round(lo, 5), round(hi, 5)])
    if save:
        per.write_parquet(save)
    imp = sorted(zip(fl, m.get_feature_importance(type="PredictionValuesChange")), key=lambda x: -x[1])[:8]
    met["top_feats"] = [f"{k}:{v:.1f}" for k, v in imp]
    record(group, name, {"walks": walks, "use_src": use_src, "drop": list(drop), "neg": neg, "wins": wins, "max_rank": max_rank, **rk}, **met)
    return per, m, fl


def grid_walk2():
    g = "walk2"
    cat = {"entity": ["cat"]}
    eval_walk(g, "base2: entity=user x cat", cat)
    for b in (0.0, 0.25):
        eval_walk(g, f"cat, beta={b}", {**cat, "beta": b})
    eval_walk(g, "cat, w_show=1.0", {**cat, "w_show": 1.0})
    eval_walk(g, "cat, max_seeds=300", {**cat, "max_seeds": 300})
    eval_walk(g, "cat, cap_ent=500", {**cat, "cap_ent": 500})
    eval_walk(g, "cat, hist_days=56", {**cat, "hist_days": 56})
    for ent in (["vert"], ["cat", "reg"], ["cat", "loc"], ["s0"], ["s0", "s1"], ["cat", "s0"]):
        eval_walk(g, f"entity=user x {'+'.join(ent)}", {"entity": ent})
    eval_walk(g, "entity=sess 24h x cat", {"entity": "sess:cat"})


def pool_curve(walks, w=1, sizes=(100, 300, 500, 1000, 2000)):
    """Потолок объединённого пула при разном числе кандидатов от каждого источника."""
    tgt = target(w)
    res = {name: run_walk({**cfg, "n_out": max(sizes)}, w, name=name) for name, cfg in walks.items()}
    for n in sizes:
        u = pl.concat([r.filter(pl.col(f"{k}_r") < n).select("user_id", "item_id") for k, r in res.items()]).unique()
        per_user = u.height / tgt["user_id"].n_unique()
        record("pool", f"union top-{n} of {'+'.join(walks)}", {"n": n}, pool=round(pool_recall(u, tgt), 4), cands_per_user=round(per_user))
    return res


def grid_walk3():
    g = "walk3"
    cat0 = {"entity": ["cat"], "beta": 0.0}
    eval_walk(g, "ref3: cat, beta=0", cat0)
    eval_walk(g, "cat+reg, beta=0", {**cat0, "entity": ["cat", "reg"]})
    eval_walk(g, "cat+reg, beta=0.25", {**cat0, "entity": ["cat", "reg"], "beta": 0.25})
    eval_walk(g, "cat, beta=0, max_seeds=300", {**cat0, "max_seeds": 300})
    eval_walk(g, "cat, beta=0, w_show=1.0", {**cat0, "w_show": 1.0})
    eval_walk(g, "cat, beta=0, w_contact=1", {**cat0, "w_contact": 1.0})


def run_pool():
    pool_curve({
        "cr": {"entity": ["cat", "reg"], "beta": 0.0},
        "c5": {"entity": ["cat"], "beta": 0.5},
        "u0": {"entity": "user", "beta": 0.0},
        "s24": {"entity": "sess", "beta": 0.0},
    })


WALKS = {
    "cr": {"entity": ["cat", "reg"], "beta": 0.0, "n_out": 500},
    "c5": {"entity": ["cat"], "beta": 0.5, "n_out": 500},
    "u0": {"entity": "user", "beta": 0.0, "n_out": 500},
    "s24": {"entity": "sess", "beta": 0.0, "n_out": 500},
}
REF_RANK = f"{CACHE}/per_R0fix.parquet"


def grid_rank0():
    eval_rank("rank", "R0: 4 walks + pop + bucket-pop, all feats (null fix)", WALKS, save=REF_RANK)


def grid_rank():
    g, r = "rank", REF_RANK
    eval_rank(g, "R1: - affinity feats", WALKS, drop=("oracle", "aff"), ref=r)
    eval_rank(g, "R2: - item dynamics feats", WALKS, drop=("oracle", "dyn"), ref=r)
    eval_rank(g, "R3: only cr + pop", WALKS, use_src=["cr", "pop"], ref=r)
    for s in ("c5", "u0", "s24", "bp"):
        keep = [x for x in ("cr", "c5", "u0", "s24", "pop", "bp") if x != s]
        eval_rank(g, f"R4: - source {s}", WALKS, use_src=keep, ref=r)
    for n in (50, 500):
        eval_rank(g, f"R5: neg={n}", WALKS, neg=n, ref=r)
    eval_rank(g, "R6: loss=QueryCrossEntropy", WALKS, loss="QueryCrossEntropy", ref=r)
    eval_rank(g, "R7: depth=8", WALKS, depth=8, ref=r)
    eval_rank(g, "X1: + oracle future item activity", WALKS, drop=(), ref=r)


REF_B = f"{CACHE}/per_B.parquet"
SRC_B = ["cr", "u0", "s24", "pop", "bp"]


def grid_rank2():
    """База B = все источники, кроме c5 (R4 показал, что c5 вредит)."""
    g, r = "rank2", REF_B
    eval_rank(g, "B: cr + u0 + s24 + pop + bp", WALKS, use_src=SRC_B, save=r)
    for s in ("u0", "s24", "bp", "pop"):
        eval_rank(g, f"B - {s}", WALKS, use_src=[x for x in SRC_B if x != s], ref=r)
    eval_rank(g, "B, пул top-300 на источник", WALKS, use_src=SRC_B, max_rank=300, ref=r)
    for n in (50, 500):
        eval_rank(g, f"B, neg={n}", WALKS, use_src=SRC_B, neg=n, ref=r)
    eval_rank(g, "B, loss=QueryCrossEntropy", WALKS, use_src=SRC_B, loss="QueryCrossEntropy", ref=r)
    eval_rank(g, "B, depth=8", WALKS, use_src=SRC_B, depth=8, ref=r)
    eval_rank(g, "X1: B + оракул будущей активности item", WALKS, use_src=SRC_B, drop=(), ref=r)


REF_C = f"{CACHE}/per_C.parquet"


def grid_rank3():
    """База C = B с neg=50 (быстрее, не хуже)."""
    g, r = "rank3", REF_C
    eval_rank(g, "C: B, neg=50", WALKS, use_src=SRC_B, neg=50, save=r)
    eval_rank(g, "C - pop - bp", WALKS, use_src=["cr", "u0", "s24"], neg=50, ref=r)
    eval_rank(g, "C, depth=8", WALKS, use_src=SRC_B, neg=50, depth=8, ref=r)
    eval_rank(g, "C, loss=PairLogit", WALKS, use_src=SRC_B, neg=50, loss="PairLogit", ref=r)
    eval_rank(g, "X1: C + оракул будущей активности item", WALKS, use_src=SRC_B, neg=50, drop=(), ref=r)


def grid_oracle_shows():
    """X1': оракул только будущих показов (без контактов — без утечки метки)."""
    g, r = "rank3", REF_C
    orig = feats_cached

    def with_shows(w, *a, **k):
        df = orig(w, *a, **k)
        c = WIN[w][0]
        sh = (EV.filter((pl.col("timestamp") >= c) & (pl.col("timestamp") < c + 7 * DAY) & (pl.col("eid") == SHOW))
              .group_by("item_id").agg(pl.len().cast(pl.UInt32).alias("or_sh")))
        return df.drop("or_ev", "or_ct").join(sh, on="item_id", how="left")

    globals()["feats_cached"] = with_shows
    eval_rank(g, "X1': C + оракул будущих показов item (без контактов)", WALKS, use_src=SRC_B, neg=50, drop=(), ref=r)
    globals()["feats_cached"] = orig


def grid_oracle_loo():
    """X1'': будущие показы item другим юзерам (leave-one-out) — честная граница для прогноза динамики."""
    g, r = "rank3", REF_C
    orig = feats_cached

    def with_loo(w, *a, **k):
        df = orig(w, *a, **k)
        c = WIN[w][0]
        fut = EV.filter((pl.col("timestamp") >= c) & (pl.col("timestamp") < c + 7 * DAY) & (pl.col("eid") == SHOW))
        tot = fut.group_by("item_id").agg(pl.len().cast(pl.Int64).alias("_t"))
        own = fut.group_by("user_id", "item_id").agg(pl.len().cast(pl.Int64).alias("_o"))
        return (df.drop("or_ev", "or_ct").join(tot, on="item_id", how="left").join(own, on=["user_id", "item_id"], how="left")
                .with_columns(or_sh=(pl.col("_t").fill_null(0) - pl.col("_o").fill_null(0)).cast(pl.UInt32)).drop("_t", "_o"))

    globals()["feats_cached"] = with_loo
    eval_rank(g, "X1'': C + оракул будущих показов другим юзерам (LOO)", WALKS, use_src=SRC_B, neg=50, drop=(), ref=r)
    globals()["feats_cached"] = orig


def grid_final():
    """Выбранная конфигурация: обучение на W1, тест на W0 (то же, что делает final.ipynb)."""
    eval_rank("final", "FINAL: cr + s24 + u0, neg=50 — W1 -> W0", WALKS, use_src=["cr", "u0", "s24"], neg=50, wins=(1, 0), save=f"{CACHE}/per_final_W0.parquet")


GROUPS = {"final": grid_final, "oracle_loo": grid_oracle_loo, "oracle_shows": grid_oracle_shows, "rank3": grid_rank3, "walk": grid_walk, "walk2": grid_walk2, "walk3": grid_walk3, "pool": run_pool, "rank0": grid_rank0, "rank": grid_rank, "rank2": grid_rank2}

if __name__ == "__main__":
    for w in (0, 1, 2):
        log(f"target W{w}: {target(w).height} pairs, {target(w)['user_id'].n_unique()} users")
    GROUPS[sys.argv[1]]()
