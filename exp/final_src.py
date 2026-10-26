# %% [markdown]
# # Avito RecSys: финальный пайплайн
# Кандидаты — блуждания item → сущность → item (юзер × категория × регион, юзер × день, юзер), co-visitation и BM25 на тех же графах,
# популярное в сегментах s0 × loc юзера; ранжирование — CatBoost YetiRank.
# Ранкер учится на неделе до порога (W1), тест — неделя после (W0, таргет по логике организаторов).
# Меняется только `SAMPLE`: 0.01 — одна партиция (ноутбук), 1.0 — весь датасет (сервер).

# %% Конфиг и зависимости
SAMPLE = 0.01
import functools, importlib.metadata as md, json, math, os, resource, shutil, subprocess, sys, time
GPU = shutil.which("nvidia-smi") is not None
PKGS = {"polars": "1.40.1", "numpy": "2.4.4", "scipy": "1.17.1", "pandas": "3.0.3", "pyarrow": "24.0.0", "catboost": "1.2.10"} | ({"cupy-cuda12x": "14.2.0"} if GPU else {})
have = {d.metadata["Name"].lower(): d.version for d in md.distributions()}
if need := [f"{p}{'[ctk]' * p.startswith('cupy')}=={v}" for p, v in PKGS.items() if have.get(p) != v]:  # ставим только недостающее
    assert "numpy" not in sys.modules, "numpy уже импортирован: перезапустите ядро"; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
import numpy as np, polars as pl, scipy.sparse as ssp; from catboost import CatBoostRanker, Pool
if GPU: import cupy as cp, cupyx.scipy.sparse as csp
xp, xsp, cpu = (cp, csp, lambda a: a.get()) if GPU else (np, ssp, lambda a: a)
RAW, OUT, DAY, H, K, SHOW, SEED, OOM = "data/raw", "data/final", 86_400_000, 3_600_000, 160, 7, 42, ("emory", "ALLOC", "RESOURCES", "bad_alloc")
T0, END, T1 = 1_775_595_600_000, 1_776_211_200_000, 1_775_595_600_000 - 7 * DAY  # тест: порог local_eval.csv (04-07 21:00Z), конец train, обучение ранкера
CONTACT, BUCKETS = [0, 2, 4, 5, 6, 8, 9, 11, 14, 15, 16], (("v0", (0,)), ("v2", (2,)), ("v3", (3,)), ("v4", (4,)), ("v57", (5, 7)))
HIST, W_CONTACT, W_SHOW, TAU, SEEDS, CAP_ITEM, CAP_ENT, NOUT, NEG = 28, 3.0, 0.3, 7.0, 100, 300, 200, 1000, 50  # β = 0: без деления на популярность
K1, B, NNZ_MAX, SP = 100.0, 0.8, 1_500_000_000, (20, 100, 300)  # BM25; блок графа < 2^31 nnz (индексы cupyx — int32); sp: сегментов, item на сегмент, кандидатов
WALKS = {"cr": ("cat", "reg"), "s24": ("b",), "u0": ()}  # сущность = юзер × атрибуты item; b — день события
ATTRS = {"vertical_id": "vert", "category_ext_y": "cat", "region_id_y": "reg", "loc_id_y": "loc", "sid_0_y": "s0", "sid_1_y": "s1", "sid_2_y": "s2"}
AFF = {"cat": ["cat"], "reg": ["reg"], "loc": ["loc"], "s0": ["s0"], "s01": ["s0", "s1"], "s012": ["s0", "s1", "s2"], "vert": ["vert"]}
CATS, DROP = ["vert", "cat", "reg", "loc", "s0"], {"user_id", "item_id", "ci", "target", "s1", "s2"}
MEM_GB, GB_PER_PART = (cp.cuda.Device().mem_info[1] if GPU else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2**30, 0.4  # графы в VRAM/RAM; пик на граф×партицию (замер 30%: ~0.33 GB)
T_START, day, merge = time.time(), lambda c: time.strftime("%m-%d", time.gmtime(c / 1000)), lambda fs: functools.reduce(lambda a, b: a.join(b, on=["user_id", "ci"], how="full", coalesce=True), fs)
def log(m): print(f"[{(time.time() - T_START) / 60:6.1f} мин | пик RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:5.1f} GB] {m}", flush=True)
eta = lambda i, n, t0, what: log(f"{what} {i + 1}/{n}, осталось ~{(time.time() - t0) / max(i + 1, 1) * max(n - i - 1, 0) / 60:.1f} мин") if (i + 1) % max(1, n // 10) == 0 or i + 1 >= n else None

# %% Таргет по логике prepare_local_eval.py: один проход по партициям сразу для порогов T1 и T0
def item_arrays():  # атрибуты как массивы по item_id; vert=255 — item без атрибутов
    f, A = f"{RAW}/item_features.parquet", {}; ids = pl.read_parquet(f, columns=["item_id"])["item_id"].to_numpy()
    for col, name in ATTRS.items():
        A[name] = np.full(int(ids.max()) + 1, 255 if name == "vert" else 0, np.uint16); A[name][ids] = pl.read_parquet(f, columns=[col])[col].to_numpy()
    return A

def bucket_sample(uv, eligible):  # квоты по вертикалям: те же сиды и порядок, что у организаторов
    tot, parts = uv.group_by("user_id").agg(tot=pl.col("len").sum()), []
    for name, vs in BUCKETS:
        f = (uv.filter(pl.col("vert").is_in(vs)).group_by("user_id").agg(pl.col("len").sum()).join(tot, on="user_id")
             .filter(pl.col("len") / pl.col("tot") >= 0.9).select("user_id").unique().sort("user_id"))
        parts.append(f.sample(n=min(10_000, f.height), seed=42 + sum(vs)).with_columns(bucket=pl.lit(name)))
    pool = eligible.join(pl.concat(parts).select("user_id"), on="user_id", how="anti").sort("user_id")
    for i in range(5):
        s = pool.sample(n=min(10_000, pool.height), seed=100 + i); parts, pool = parts + [s.with_columns(bucket=pl.lit(f"h{i}"))], pool.join(s, on="user_id", how="anti").sort("user_id")
    return pl.concat(parts)

def targets(parts, A):  # + время первого/последнего события item до каждого порога (для признаков age/last)
    acc, t0, M = {c: ([], [], np.zeros(len(A["cat"]), np.uint32)) for c in (T1, T0)}, time.time(), np.uint64(2**32)
    tl = {"f": np.full(len(A["cat"]), np.inf, np.float32)} | {c: np.full(len(A["cat"]), -np.inf, np.float32) for c in (T1, T0)}  # дни от T0
    for k, p in enumerate(parts):
        ev = pl.read_parquet(p).with_columns(k=pl.col("user_id").cast(pl.UInt64) * 2**32 + pl.col("item_id"))  # партиция отсортирована по юзеру
        fl = ev.group_by("item_id").agg(f=pl.col("timestamp").min(), **{str(c): pl.col("timestamp").filter(pl.col("timestamp") < c).max() for c in (T1, T0)}).sort("item_id")
        i, day_ = fl["item_id"].to_numpy(), lambda col: ((fl[col].cast(pl.Float64) - T0) / DAY).fill_null(-np.inf).cast(pl.Float32).to_numpy()  # сортировка по item: запись подряд
        for key, f_ in (("f", np.minimum), (T1, np.maximum), (T0, np.maximum)): tl[key][i] = f_(tl[key][i], day_(str(key)))
        for c, end in ((T1, T0), (T0, END)):
            seen = ev.filter(pl.col("timestamp") < c).select("k").unique()  # быстрее group_by по двум ключам в ~20 раз
            cand = ev.filter((pl.col("timestamp") >= c + 12 * H) & (pl.col("timestamp") < end) & pl.col("eid").is_in(CONTACT)).join(seen, on="k", how="anti")
            sv = seen.with_columns(user_id=(pl.col("k") // 2**32).cast(pl.UInt32)).join(cand.select("user_id").unique(), on="user_id")
            r = (seen["k"] % 2**32).cast(pl.UInt32).sort().rle()  # юзеров на item = длины серий в отсортированных item уникальных пар
            acc[c][2][r.struct.field("value").to_numpy()] += r.struct.field("len").to_numpy().astype(np.uint32)
            acc[c][0].append(cand.select("user_id", "item_id", "timestamp"))
            acc[c][1].append(sv.with_columns(vert=A["vert"][(sv["k"].to_numpy() % M).astype(np.int64)]).filter(pl.col("vert") != 255).group_by("user_id", "vert").len())
        eta(k, len(parts), t0, "таргет: партиция")
    out = {}
    for c, (cands, uvs, ucnt) in acc.items():
        univ = ucnt >= 2; cand = (cc := pl.concat(cands)).filter(univ[cc["item_id"].to_numpy()])
        users = bucket_sample(pl.concat(uvs).join(cand.select("user_id").unique(), on="user_id"), cand.select("user_id").unique().sort("user_id"))
        out[c] = (cand.join(users.select("user_id"), on="user_id").sort(["user_id", "timestamp"]).group_by("user_id", maintain_order=True)
                  .agg(pl.col("item_id").unique(maintain_order=True).head(K)).explode("item_id"), users, univ, tl["f"], tl[c])
        log(f"таргет на {day(c)}: {users.height} юзеров, {out[c][0].height} пар")
    return out

# %% Окно c (по партициям): графы item → сущность → item, статистики item, история целевых юзеров
def window(c, users, univ, first, last, A, parts, walks):
    uni, cid = np.flatnonzero(univ).astype(np.uint32), np.full(len(univ), -1, np.int32)  # только item с >=2 юзерами: прочие не бывают в таргете
    n, cap_part = len(uni), math.ceil(CAP_ITEM / len(parts)); cid[uni] = np.arange(n, dtype=np.int32)  # лимит в партиции = выборка свежих юзеров
    code = {w: sum((A[a][uni].astype(np.uint64) * 100**j for j, a in enumerate(WALKS[w]) if a != "b"), np.zeros(n, np.uint64)) for w in walks}  # код атрибутов item
    def part(p):  # одна партиция -> вклад в статистики item, история целевых юзеров, блоки графов
        ev, ci = pl.read_parquet(p).filter(pl.col("timestamp") < c), lambda d: d.with_columns(ci=cid[d["item_id"].to_numpy()]).filter(pl.col("ci") >= 0)
        sn = ci(ev.join(users, on="user_id", how="semi").select("user_id", "item_id").unique())
        ev = ci(ev.filter(pl.col("timestamp") >= c - HIST * DAY))  # дальше — только окно HIST дней: компактный id для 1/3 событий
        rec = ev.filter(pl.col("timestamp") >= c - 14 * DAY).with_columns(a=(c - pl.col("timestamp")) / DAY, s=pl.col("eid") == SHOW)
        cnt = {name: np.bincount(rec.filter(m)["ci"].to_numpy(), minlength=n) for name, m in {
            "ev1": (pl.col("a") < 1) & ~pl.col("s"), "ev7": (pl.col("a") < 7) & ~pl.col("s"), "ev7p": (pl.col("a") >= 7) & ~pl.col("s"),
            "sh7": (pl.col("a") < 7) & pl.col("s"), "sh7p": (pl.col("a") >= 7) & pl.col("s"), "ct7": (pl.col("a") < 7) & pl.col("eid").is_in(CONTACT)}.items()}
        w = pl.when(pl.col("eid").is_in(CONTACT)).then(W_CONTACT).when(pl.col("eid") == SHOW).then(W_SHOW).otherwise(1.0).cast(pl.Float32)
        win, agg = ev.with_columns(w=w, b=(c - pl.col("timestamp")) // DAY), dict(w=pl.col("w").max(), ts=pl.col("timestamp").max(), contact=pl.col("eid").is_in(CONTACT).any())
        pairs, sess = win.group_by("user_id", "item_id", "ci").agg(**agg), win.group_by("user_id", "item_id", "ci", "b").agg(**agg)
        cnt, blocks = cnt | {"deg": np.bincount(pairs["ci"].to_numpy(), minlength=n), "nc": np.bincount(pairs.filter("contact")["ci"].to_numpy(), minlength=n),
                             "nu14": np.bincount(pairs.filter(pl.col("ts") >= c - 14 * DAY)["ci"].to_numpy(), minlength=n)}, {}
        for wn in walks:
            src = sess if "b" in WALKS[wn] else pairs
            k = code[wn][src["ci"].to_numpy()] + (src["b"].to_numpy().astype(np.uint64) if "b" in WALKS[wn] else 0)
            e = src.with_columns(e=pl.col("user_id").cast(pl.UInt64) * 65536 + k).with_columns(el=pl.col("e").rank("dense").cast(pl.Int32) - 1).sort("ts", descending=True)
            h2, h1 = e.group_by("el", maintain_order=True).head(CAP_ENT), e.group_by("ci", maintain_order=True).head(cap_part)
            blocks[wn] = (ssp.csr_matrix((h2["w"].to_numpy(), (h2["el"].to_numpy(), h2["ci"].to_numpy())), shape=(int(e["el"].max()) + 1, n)), h1)
        return cnt, pairs.join(users, on="user_id", how="semi"), sn.drop("item_id"), blocks
    st = {k: np.zeros(n, np.float32) for k in ("ev1", "ev7", "ev7p", "sh7", "sh7p", "ct7", "deg", "nc", "nu14")} | {"age": (c - T0) / DAY - first[uni], "last": (c - T0) / DAY - last[uni]}
    g, mine, seen, t0 = {w: {"bl": [], "h2": [], "h1": [], "ne": 0, "nnz": 0} for w in walks}, [], [], time.time()
    def flush(G):  # накопленные партиции -> блок (H1: item × сущности блока, H2: сущности блока × item); сущность целиком в одной партиции
        if G["h2"]: r, cc, v = (xp.concatenate(x) for x in zip(*G["h1"])); G["bl"].append((xsp.coo_matrix((v, (r, cc.astype(xp.int32))), shape=(n, G["ne"])).tocsr(), xsp.vstack(G["h2"]).tocsr()))
        G.update(h2=[], h1=[], ne=0, nnz=0)
    for k, (cnt, m, sn, blocks) in enumerate(map(part, parts)):
        for name, v in cnt.items(): st[name] += v
        mine.append(m); seen.append(sn)
        for wn, (h2, h1) in blocks.items():  # блоки графа сразу на устройство: на GPU копятся в VRAM, а не в RAM
            G = g[wn]; flush(G) if G["nnz"] + max(h2.nnz, h1.height) > NNZ_MAX else None
            G["h2"].append(xsp.csr_matrix(h2)); G["h1"].append(tuple(xp.asarray(h1[col].to_numpy()) + (G["ne"] if col == "el" else 0) for col in ("ci", "el", "w"))); G["ne"] += h2.shape[0]; G["nnz"] += max(h2.nnz, h1.height)
        eta(k, len(parts), t0, f"окно {day(c)} {'+'.join(walks)}: партиция")
    for wn, G in g.items():
        flush(G); g[wn] = G["bl"]; nnz = sum(a.nnz + b.nnz for a, b in g[wn])
        log(f"граф {wn}: {n:,} item, {sum(b.shape[0] for _, b in g[wn]):,} сущностей, {len(g[wn])} блок(а), nnz {nnz:,}, ~{nnz * 8 / 2**30 / len(parts):.3f} GB/партиция")
    return uni, st, g, pl.concat(mine), pl.concat(seen)

# %% Кандидаты: score_uj = Σ_блоков Σ_i x_ui (H1 H2)_ij (x — свежесть затравок), top-NOUT невиденных на юзера
def norm_rows(m, s=None):  # строки — в вероятности перехода; s — суммы строк по всем блокам. Без diag @ m (cuSPARSE spGEMM падает на 4М строк)
    s = xp.maximum(xp.asarray(m.sum(axis=1)).ravel() if s is None else s, 1e-9)  # сумма по CSR: без сортировки (lexsort на 2 млрд nnz — bad_alloc)
    m.data = (m.data / s[m.tocoo().row]).astype(xp.float32); return m

def bm25(H1, H2, rs):  # веса BM25 на месте (эксп. 6е): item — документ, юзер — терм; rs — вес item по всем блокам
    idf, ln = math.log(len(rs)) - xp.log1p(xp.diff(H2.indptr).astype(xp.float32)), (1 - B) + B * rs / rs.mean(); f = lambda w, it, us: (w * (K1 + 1) / (K1 * ln[it] + w) * idf[us]).astype(xp.float32)
    H1.data, H2.data = f(H1.data, H1.tocoo().row, H1.indices), f(H2.data, H2.indices, H2.tocoo().row)

def topn(rows, cols, vals, n):  # одна сортировка по ключу «строка + (0.5 − доля скора)»
    if not len(rows): return [cpu(rows)] * 4
    o = xp.argsort(rows.astype(xp.float64) + 0.5 - vals / (2 * (vals.max() + 1e-30))); r, cc, v = rows[o], cols[o], vals[o]
    rank = xp.arange(len(r)) - xp.searchsorted(r, xp.arange(int(r.max()) + 1))[r]
    return [cpu(a[rank < n]) for a in (r, cc, v, rank)]

def score(X, bl, name, uid, sk, n, batch):  # top-NOUT невиденных item на юзера по сумме блоков
    res, t0, lo, j = [], time.time(), 0, 0
    while lo < X.shape[0]:
        try: S = functools.reduce(lambda a, b: a + b, [(X[lo : lo + batch] @ H1) @ H2 for H1, H2 in bl]).tocoo()
        except Exception as ex:  # нехватка памяти GPU -> батч вдвое меньше
            if batch == 1 or not any(k in str(ex) for k in OOM): raise
            batch //= 2; log(f"нехватка памяти -> батч {batch}"); continue
        r, cc = S.row.astype(xp.int64) + lo, S.col.astype(xp.int64); keep = sk[xp.minimum(xp.searchsorted(sk, r * n + cc), len(sk) - 1)] != r * n + cc  # минус виденное
        rr, ii, vv, rk = topn(r[keep], cc[keep], S.data[keep], NOUT); res.append(pl.DataFrame({"user_id": uid[rr], "ci": ii.astype(np.int32), f"{name}_s": vv, f"{name}_r": rk.astype(np.uint16)}))
        lo, j = lo + batch, j + 1; eta(j - 1, j + math.ceil((X.shape[0] - lo) / batch), t0, f"{name}: батч")
    return pl.concat(res)

def candidates(c, g, st, mine, seen, batch):  # на графе s24 ещё co-visitation (сырые веса), на u0 — BM25, затем блуждание (нормированные строки)
    um = mine.select("user_id").unique().sort("user_id").with_row_index("ui")
    sd = (mine.join(um, on="user_id").with_columns(x=pl.col("w") * (-(c - pl.col("ts")) / (TAU * DAY)).exp())
          .sort("x", descending=True).group_by("user_id", maintain_order=True).head(SEEDS))
    n, uid, out = len(st["deg"]), um["user_id"].to_numpy(), []
    X = norm_rows(xsp.csr_matrix(ssp.csr_matrix((sd["x"].to_numpy(), (sd["ui"].to_numpy(), sd["ci"].to_numpy())), shape=(um.height, n))))
    sk = xp.asarray(np.append(seen.join(um, on="user_id").select(pl.col("ui").cast(pl.Int64) * n + pl.col("ci")).to_series().sort().to_numpy(), -1))
    sc = lambda bl, name: out.append(score(X, bl, name, uid, sk, n, batch))
    for wn, bl in g.items():
        if wn == "s24": sc(bl, "covis")
        if wn == "u0":  # BM25 на месте, затем исходные веса обратно (копия только данных, не графа)
            raw, rs = [(a.data.copy(), b.data.copy()) for a, b in bl], functools.reduce(xp.add, [xp.bincount(b.indices, weights=b.data, minlength=n) for _, b in bl])
            [bm25(a, b, rs) for a, b in bl]; sc(bl, "bm25")
            for (a, b), (da, db) in zip(bl, raw): a.data, b.data = da, db
        s1, raw = functools.reduce(xp.add, [xp.asarray(a.sum(axis=1)).ravel() for a, _ in bl]), None  # строки H1 — по всем блокам
        for a, b in bl: norm_rows(a, s1); norm_rows(b)
        sc(bl, wn); cp.get_default_memory_pool().free_all_blocks() if GPU else None
    return merge(out)

def seg_pop(mine, st, uni, A, seen):  # популярное по nu14 в главных сегментах s0 × loc юзера (класс промахов B, эксп. 6д); скор = доля юзера × log1p(nu14)
    key = lambda ci: A["s0"][uni[ci]].astype(np.int64) * 65536 + A["loc"][uni[ci]]
    aff = (mine.with_columns(g=key(mine["ci"].to_numpy())).group_by("user_id", "g").agg(pl.col("w").sum()).with_columns(sh=pl.col("w") / pl.col("w").sum().over("user_id"))
           .sort(["user_id", "sh"], descending=[False, True]).group_by("user_id", maintain_order=True).head(SP[0])); ci = np.flatnonzero(st["nu14"] > 0)
    top = pl.DataFrame({"ci": ci.astype(np.int32), "g": key(ci), "p": st["nu14"][ci]}).sort("p", descending=True).group_by("g", maintain_order=True).head(SP[1])
    df = aff.join(top, on="g").join(seen, on=["user_id", "ci"], how="anti").group_by("user_id", "ci").agg(sp_s=(pl.col("sh") * pl.col("p").log1p()).max().cast(pl.Float32))
    return df.sort(["user_id", "sp_s"], descending=[False, True]).group_by("user_id", maintain_order=True).head(SP[2]).with_columns(sp_r=pl.int_range(pl.len()).over("user_id").cast(pl.UInt16))

def stage(c, users, un, A, parts, batch):  # все графы за один проход, если помещаются в память; при нехватке — по проходу на граф
    est = len(parts) * GB_PER_PART * len(WALKS)
    for groups in [[list(WALKS)], [[w] for w in WALKS]][0 if est < 0.7 * MEM_GB else 1:]:
        log(f"окно {day(c)}: графы ~{est:.0f} GB при {'VRAM' if GPU else 'RAM'} {MEM_GB:.0f} GB -> проходов по данным: {len(groups)}"); cands = []
        try:
            for grp in groups:
                uni, st, g, mine, seen = window(c, users, un[str(c)], un["first"], un[f"last{c}"], A, parts, grp); cands.append(candidates(c, g, st, mine, seen, batch)); del g
            return uni, st, mine, merge(cands + [seg_pop(mine, st, uni, A, seen)])
        except Exception as ex:  # нехватка памяти при одном проходе -> по проходу на граф
            if len(groups) > 1 or not any(k in str(ex) for k in OOM): raise
            g = None; cp.get_default_memory_pool().free_all_blocks() if GPU else None

# %% Признаки, ранкер
def featurize(cand, c, uni, st, mine, A):
    it, um = cand["ci"].to_numpy(), (u := mine.join(cand.select("user_id").unique(), on="user_id")).with_columns(**{a: A[a][uni[u["ci"].to_numpy()]] for a in ATTRS.values()})
    df = cand.with_columns(item_id=uni[it], **{k: v[it] for k, v in st.items()}, **{a: A[a][uni[it]] for a in ATTRS.values()}).with_columns(
        i_trend=(pl.col("ev7") + 1) / (pl.col("ev7p") + 1), i_shtrend=(pl.col("sh7") + 1) / (pl.col("sh7p") + 1))
    df = df.join(um.group_by("user_id").agg(u_n=pl.len(), u_w=pl.col("w").sum(), u_nc=pl.col("contact").sum(),
                                            u_last=(c - pl.col("ts").max()) / DAY, u_ncat=pl.col("cat").n_unique()), on="user_id", how="left")
    for name, keys in AFF.items():  # аффинити юзера к атрибутам кандидата: доля веса, давность, контакты
        a = um.group_by("user_id", *keys).agg(_w=pl.col("w").sum(), _ts=pl.col("ts").max(), **{f"a_{name}_nc": pl.col("contact").sum()})
        df = df.join(a, on=["user_id", *keys], how="left").with_columns(
            **{f"a_{name}_sh": pl.col("_w") / pl.col("u_w"), f"a_{name}_ago": (c - pl.col("_ts")) / DAY}).drop("_w", "_ts")
    df = df.with_columns(pl.col("^.*_r$").fill_null(999), pl.col("^.*_ago$").fill_null(999), pl.col("u_last").fill_null(999))  # null — «нет сигнала»: иначе NaN < всех (ранг null ≈ ранг 0)
    return df.with_columns(pl.col("^.*(_s|_sh|_nc)$").fill_null(0), pl.col("^u_(n|w|ncat)$").fill_null(0))

def to_pool(df, feats, label=True):
    X = df.select(feats).to_pandas(); X[CATS] = X[CATS].fillna(-1).astype("int64")
    return Pool(X, df["target"].to_numpy() if label else None, group_id=df["user_id"].to_numpy(), cat_features=CATS)

def fit(tr, iters):  # early stopping по 10% обучающих юзеров (тест в подборе не участвует)
    feats, hold, srt = [f for f in tr.columns if f not in DROP], tr.select("user_id").unique().sort("user_id").sample(fraction=0.1, seed=SEED), lambda d: d.sort(["user_id", "item_id"])
    lr = 0.08 * (max(tr.height, 7e5) / 7e5) ** -0.15  # на 1% (~0.7М строк) подобрано 0.08; больше данных — меньше шаг
    m = CatBoostRanker(loss_function="YetiRank", iterations=iters, learning_rate=lr, depth=6, random_seed=SEED, task_type="GPU" if GPU else "CPU",
                       od_type="Iter", od_wait=150, use_best_model=True, verbose=max(1, iters // 10))
    m.fit(to_pool(srt(tr.join(hold, on="user_id", how="anti")), feats), eval_set=to_pool(srt(tr.join(hold, on="user_id")), feats))
    log(f"ранкер: {tr.height:,} строк, {len(feats)} признаков, lr {lr:.3f}, лучшая итерация {m.get_best_iteration()}"); return m

# %% Пайплайн с чекпоинтами: повторный запуск продолжает с последнего готового этапа
def pipeline(parts, out, frac=1.0, iters=3000, batch=512 if GPU else 256):
    ck, A = lambda name: f"{out}/{name}", item_arrays()
    if not os.path.exists(ck("univ.npz")):
        os.makedirs(out, exist_ok=True); T = targets(parts, A)
        for i, name in enumerate(("targets", "users")): pl.concat([T[c][i].with_columns(c=pl.lit(c)) for c in T]).write_parquet(ck(f"{name}.parquet"))
        np.savez(ck("univ.npz"), first=T[T0][3], **{str(c): T[c][2] for c in T}, **{f"last{c}": T[c][4] for c in T}); del T
    un = np.load(ck("univ.npz"))
    pick = lambda name, c: pl.read_parquet(ck(name)).filter((pl.col("c") == c) & (pl.col("user_id").hash(SEED) % 1000 < frac * 1000)).drop("c")
    if not os.path.exists(ck("model.cbm")):
        tgt = pick("targets.parquet", T1)
        uni, st, mine, cand = stage(T1, tgt.select("user_id").unique(), un, A, parts, batch)
        cand = cand.with_columns(item_id=uni[cand["ci"].to_numpy()]).join(tgt.with_columns(target=pl.lit(1, pl.Int8)), on=["user_id", "item_id"], how="left").with_columns(pl.col("target").fill_null(0))
        log(f"W1: {cand.height:,} кандидатов, потолок пула {cand['target'].sum() / tgt.height:.4f}")
        cand = cand.join(cand.filter(pl.col("target") == 1).select("user_id").unique(), on="user_id")  # группы без позитивов не учат; негативы — по хэшу пары, не по позиции строки (эксп. 6г)
        cand = cand.filter((pl.col("target") == 1) | (pl.struct("user_id", "ci").hash(SEED).rank("ordinal").over("user_id") <= NEG)).drop("item_id")
        fit(featurize(cand, T1, uni, st, mine, A), iters).save_model(ck("model.cbm")); del mine, cand
    m = CatBoostRanker().load_model(ck("model.cbm"))
    if not os.path.exists(ck("top160.parquet")):
        uni, st, mine, cand = stage(T0, pick("users.parquet", T0).select("user_id"), un, A, parts, batch)
        ub, preds, t0 = cand["user_id"].unique().sort().to_numpy(), [], time.time(); step = max(1, 4_000_000 * ub.size // cand.height)  # ~4М строк за раз
        for i in range(0, ub.size, step):
            d = featurize(cand.filter(pl.col("user_id").is_in(ub[i : i + step])), T0, uni, st, mine, A).sort("user_id")  # CatBoost: группы подряд
            preds.append(d.select("user_id", "item_id", score=m.predict(to_pool(d, m.feature_names_, label=False)))
                         .sort(["user_id", "score"], descending=[False, True]).group_by("user_id", maintain_order=True).head(K))
            eta(i // step, math.ceil(ub.size / step), t0, "W0: скоринг, батч")
        pl.concat(preds).write_parquet(ck("top160.parquet"))
    tgt = pick("targets.parquet", T0); hits = pl.read_parquet(ck("top160.parquet")).join(tgt, on=["user_id", "item_id"], how="semi").group_by("user_id").len()
    per = tgt.group_by("user_id").agg(n=pl.len()).join(hits, on="user_id", how="left").select("user_id", rec=pl.col("len").fill_null(0) / pl.col("n")).join(pick("users.parquet", T0), on="user_id")  # Recall@160 по юзерам W0
    boot = np.random.default_rng(SEED).choice(per["rec"].to_numpy(), (1000, per.height)).mean(axis=1)
    res = {"recall@160": round(per["rec"].mean(), 5), "ci95": np.quantile(boot, [0.025, 0.975]).round(5).tolist(), "users": per.height,
           "buckets": dict(per.group_by("bucket").agg(pl.col("rec").mean().round(5)).sort("bucket").rows()), "minutes": round((time.time() - T_START) / 60, 1)}
    json.dump(res, open(ck("metrics.json"), "w"), indent=1)
    log(f"Recall@160 = {res['recall@160']:.4f} {res['ci95']} по {res['users']} юзерам; бакеты {res['buckets']}")
    return res

# %% Smoke-test: весь путь на 1 партиции и ~3% юзеров за 1–2 минуты — ловит ошибки до долгого прогона
PARTS = [f"{RAW}/train_data/part_{p:03d}.parquet" for p in range(max(1, round(100 * SAMPLE)))]
shutil.rmtree(f"{OUT}/smoke", ignore_errors=True)
smoke = pipeline(PARTS[:1], f"{OUT}/smoke", frac=0.03, iters=30)
assert 0 < smoke["recall@160"] < 1 and smoke["users"] > 50, smoke

# %% Полный прогон (замер: SAMPLE=0.75 — 89 мин на A100 80 GB до добавления covis/BM25/sp и NOUT 1000; теперь ожидать ~×1.5)
res = pipeline(PARTS, f"{OUT}/sample_{SAMPLE}")
