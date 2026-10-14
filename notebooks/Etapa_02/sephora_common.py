"""Protocolo común y cuatro recomendadores de skincare para usuarios sin historial.

Los notebooks explican y muestran el código de cada clase. Este módulo centraliza
la preparación y evaluación para que no cambien entre experimentos.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import heapq
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

PROFILE_COLS = ["skin_type", "skin_tone", "eye_color", "hair_color"]
PROTOCOL_VERSION = "cold-start-users-v2.0"
TOP_N = 10


def find_project_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data" / "processed").is_dir():
            return p
    raise FileNotFoundError("No se encontró data/processed. Abrí el notebook dentro del proyecto.")


def read_table(directory, name):
    parquet, csv = directory / f"{name}.parquet", directory / f"{name}.csv"
    review_columns = {"author_id", "product_id", "rating", "is_recommended", "submission_time", *PROFILE_COLS}
    if parquet.exists():
        if name == "reviews_clean":
            import pyarrow.parquet as pq
            columns = [c for c in pq.read_schema(parquet).names if c in review_columns]
            return pd.read_parquet(parquet, columns=columns)
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, dtype={"author_id": "string", "product_id": "string"},
                           usecols=(lambda c: c in review_columns) if name == "reviews_clean" else None)
    raise FileNotFoundError(f"Falta {parquet.name} o {csv.name} en {directory}")


def text_value(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join(str(x) for x in v)
    if isinstance(v, dict):
        return " ".join(str(x) for x in v.values())
    if pd.isna(v):
        return ""
    return str(v)


def profile_mode(s):
    values = s.dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin(["", "unknown", "nan", "none", "<na>"])]
    return str(values.mode().iloc[0]) if len(values) else "Unknown"


@dataclass
class Context:
    train: pd.DataFrame
    catalog: pd.DataFrame
    profiles: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    manifest: dict
    protocol_id: str
    output_dir: Path | None = None


def prepare_context(reviews, products, n_validation=500, n_test=500, seed=42):
    """Parte por usuario: ningún usuario de validación/test aporta reviews a train."""
    required = {"author_id", "product_id", "rating", "skin_type", "skin_tone"}
    missing = required - set(reviews.columns)
    if missing:
        raise ValueError(f"Faltan columnas de reviews: {sorted(missing)}")
    if not {"product_id", "primary_category"}.issubset(products.columns):
        raise ValueError("El catálogo necesita product_id y primary_category")
    if n_validation < 1 or n_test < 1:
        raise ValueError("Validación y test necesitan al menos un usuario")
    catalog = products.dropna(subset=["product_id"]).copy()
    catalog["product_id"] = catalog.product_id.astype(str).str.strip()
    if not catalog.product_id.is_unique:
        raise ValueError("IDs de catálogo duplicados: resolverlos en procesamiento")
    catalog = catalog[catalog.primary_category.astype(str).str.lower().eq("skincare")]
    catalog = catalog.sort_values("product_id").reset_index(drop=True)
    if len(catalog) == 0:
        raise ValueError("No hay productos con primary_category=Skincare")
    consumed_cols = ["author_id", "product_id", "rating", "is_recommended", "submission_time", *PROFILE_COLS]
    r = reviews.copy()
    for col in consumed_cols:
        if col not in r:
            r[col] = pd.NA
    r = r[consumed_cols].copy()
    r = r.dropna(subset=["author_id", "product_id"])
    for col in ["author_id", "product_id"]:
        r[col] = r[col].astype(str).str.strip()
    r = r[r.author_id.ne("") & r.product_id.isin(catalog.product_id)]
    r["rating"] = pd.to_numeric(r.rating, errors="coerce")
    r = r[r.rating.between(1, 5)].copy()
    for col in PROFILE_COLS:
        r[col] = r[col].map(lambda v: text_value(v).strip() or "Unknown")
        r.loc[r[col].str.lower().isin(["unknown", "none", "nan", "<na>"]), col] = "Unknown"
    r["_order"] = np.arange(len(r))
    r["submission_time"] = pd.to_datetime(r.submission_time, errors="coerce", utc=True)
    before_pairs = len(r)
    # Última review con fecha disponible; empate o ausencia de fecha: orden del archivo.
    r = r.sort_values(["submission_time", "_order"], na_position="first", kind="stable")
    r = r.drop_duplicates(["author_id", "product_id"], keep="last")
    r["positive"] = r.rating.ge(4) | r.is_recommended.astype(str).str.lower().isin(["true", "1", "1.0"])
    r = r.sort_values(["author_id", "product_id"]).reset_index(drop=True)
    # Supuesto explícito: estos atributos personales están disponibles al consultar.
    # No se utilizan ratings, identidad de productos o resultados para construirlos.
    # Moda vectorizada: evita ejecutar una función Python dos millones de veces.
    profiles = pd.DataFrame(index=pd.Index(sorted(r.author_id.unique()), name="author_id"))
    for col in PROFILE_COLS:
        known = r[r[col].ne("Unknown")]
        frequency = known.groupby(["author_id", col]).size().reset_index(name="frequency")
        modes = frequency.sort_values(["author_id", "frequency", col], ascending=[True, False, True])
        modes = modes.drop_duplicates("author_id").set_index("author_id")[col]
        profiles[col] = modes.reindex(profiles.index).fillna("Unknown")
    positive = r[r.positive]
    counts = positive.groupby("author_id").size()
    complete = profiles.skin_type.ne("Unknown") & profiles.skin_tone.ne("Unknown")
    eligible = sorted(set(counts[counts >= 2].index) & set(profiles[complete].index))
    if len(eligible) < n_validation + n_test:
        raise ValueError(f"Hay {len(eligible)} usuarios elegibles; se requieren {n_validation+n_test}. Reducí N_VALIDACION y N_TEST para una prueba pequeña.")
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.array(eligible, dtype=object), n_validation + n_test, replace=False)
    heldout = set(selected)
    positives_by_user = positive.groupby("author_id").product_id.apply(list).to_dict()
    def make_cases(ids):
        return pd.DataFrame([
            {"author_id": uid, "target": str(rng.choice(positives_by_user[uid])),
             "known": positives_by_user[uid]} for uid in ids
        ])
    validation = make_cases(selected[:n_validation])
    test = make_cases(selected[n_validation:])
    train = r[~r.author_id.isin(heldout)].copy()
    if train.empty:
        raise ValueError("El split dejó train vacío")
    # Huella de todos los insumos que usan los modelos; bloquea comparaciones mezcladas.
    consumed = r[consumed_cols + ["positive"]].astype(str)
    catalog_cols = [c for c in ["product_id", "brand_name", "primary_category", "secondary_category", "tertiary_category", "highlights", "ingredients", "price_usd"] if c in catalog]
    catalog_digest = catalog[catalog_cols].copy()
    for col in catalog_digest:
        catalog_digest[col] = catalog_digest[col].map(text_value)
    digest = sha256(pd.util.hash_pandas_object(consumed, index=False).values.tobytes())
    digest.update(pd.util.hash_pandas_object(catalog_digest, index=False).values.tobytes())
    digest.update(f"{PROTOCOL_VERSION}|{seed}|{n_validation}|{n_test}".encode())
    protocol_id = digest.hexdigest()
    manifest = {
        "version": PROTOCOL_VERSION, "seed": seed, "protocol_id": protocol_id,
        "reviews_input": len(reviews), "reviews_valid_before_pair_dedup": before_pairs,
        "reviews_unique_pairs": len(r), "removed_duplicate_pairs": before_pairs-len(r),
        "catalog_skincare": len(catalog), "users_total": len(profiles),
        "eligible_users": len(eligible), "training_users": train.author_id.nunique(),
        "training_reviews": len(train), "validation_users": len(validation),
        "test_users": len(test), "positive_rule": "rating >= 4 OR is_recommended == true",
        "holdout": "all reviews of validation/test users excluded from training",
        "query_inputs": "profile only; other positives of held-out user are NOT used or filtered",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "validation_cases": validation.to_dict("records"), "test_cases": test.to_dict("records"),
    }
    return Context(train, catalog, profiles, validation, test, manifest, protocol_id)


def load_context(n_validation=500, n_test=500, seed=42):
    root = find_project_root()
    ctx = prepare_context(read_table(root / "data" / "processed", "reviews_clean"),
                          read_table(root / "data" / "processed", "products_clean"),
                          n_validation, n_test, seed)
    ctx.output_dir = root / "artifacts" / "Etapa_02"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(ctx.output_dir / "protocolo.json", ctx.manifest)
    return ctx


def rank_scores(frame, score_col, top_n=10, exclude=()):
    if frame.empty:
        return []
    f = frame[~frame.product_id.isin(set(exclude))].copy()
    f = f[np.isfinite(f[score_col])]
    return f.sort_values([score_col, "product_id"], ascending=[False, True], kind="stable").product_id.head(top_n).tolist()


class PopularityBaseline:
    def __init__(self, ctx):
        self.ranking = ctx.train[ctx.train.positive].groupby("product_id").size().reset_index(name="score")

    def recommend(self, profile, config=None, top_n=10, exclude=()):
        return rank_scores(self.ranking, "score", top_n, exclude)


class ProfileKNN:
    """K vecinos exactos por coseno; agrupa perfiles repetidos para acelerar."""
    def __init__(self, ctx):
        self.profiles = ctx.profiles.loc[sorted(ctx.train.author_id.unique())]
        self.encoder = DictVectorizer(sparse=True)
        records = self.profiles.to_dict("records")
        self.encoder.fit(records + [{c: "Unknown" for c in PROFILE_COLS}])
        self.allowed = {c: set(self.profiles[c]) | {"Unknown"} for c in PROFILE_COLS}
        # Una firma representa a todas las personas con exactamente ese perfil.
        self.groups = {}
        for uid, row in self.profiles.iterrows():
            self.groups.setdefault(tuple(row[c] for c in PROFILE_COLS), []).append(uid)
        self.signatures = list(self.groups)
        self.X = self.encoder.transform([dict(zip(PROFILE_COLS, s)) for s in self.signatures])
        self.positives = ctx.train[ctx.train.positive].groupby("author_id").product_id.apply(list).to_dict()
        self.cache = {}
        self.n_train_users = len(self.profiles)

    def neighbors(self, profile, k):
        if not 1 <= k <= self.n_train_users:
            raise ValueError(f"K={k} inválido para {self.n_train_users} usuarios de entrenamiento")
        key = tuple(profile.get(c, "Unknown") if profile.get(c, "Unknown") in self.allowed[c] else "Unknown" for c in PROFILE_COLS)
        if key not in self.cache or len(self.cache[key]) < k:
            query = self.encoder.transform([dict(zip(PROFILE_COLS, key))])
            distances = np.round(1-cosine_similarity(query, self.X).ravel(), 12)
            neighbors = []
            for d in sorted(set(distances)):
                group_indices = np.flatnonzero(distances == d)
                # Cada lista ya está ordenada por author_id. El merge resuelve empates.
                streams = [self.groups[self.signatures[i]] for i in group_indices]
                remaining = k-len(neighbors)
                neighbors.extend(islice(heapq.merge(*streams), remaining))
                if len(neighbors) == k:
                    break
            self.cache[key] = neighbors
        return self.cache[key][:k]

    def recommend(self, profile, config, top_n=10, exclude=()):
        votes = Counter()
        for uid in self.neighbors(profile, int(config["k"])):
            votes.update(self.positives.get(uid, []))
        frame = pd.DataFrame(list(votes.items()), columns=["product_id", "score"])
        return rank_scores(frame, "score", top_n, exclude)


class SegmentSatisfaction:
    """Segmentación estricta por tipo de piel + media de ratings suavizada."""
    def __init__(self, ctx):
        r = ctx.train[ctx.train.skin_type.ne("Unknown")]
        self.stats = r.groupby(["skin_type", "product_id"]).rating.agg(["sum", "count"]).reset_index()
        self.priors = r.groupby("skin_type").rating.mean().to_dict()

    def scores(self, profile, config):
        skin = profile.get("skin_type", "Unknown")
        stats = self.stats[self.stats.skin_type.eq(skin)].copy()
        if not stats.empty:
            m = float(config["m"])
            stats["score"] = (stats["sum"] + m*self.priors[skin]) / (stats["count"]+m)
        return stats

    def recommend(self, profile, config, top_n=10, exclude=()):
        return rank_scores(self.scores(profile, config), "score", top_n, exclude)


class WeightedProfileSatisfaction:
    """Cada experiencia pesa según coincidencia de tipo y tono de piel."""
    def __init__(self, ctx):
        r = ctx.train[ctx.train.skin_type.ne("Unknown") & ctx.train.skin_tone.ne("Unknown")]
        self.stats = r.groupby(["skin_type", "skin_tone", "product_id"]).rating.agg(["sum", "count"]).reset_index()
        self.cache = {}

    def scores(self, profile, config):
        wt, wn, m = float(config["weight_type"]), float(config["weight_tone"]), float(config["m"])
        if wt < 0 or wn < 0 or not np.isclose(wt+wn, 1):
            raise ValueError("Los pesos deben ser no negativos y sumar 1")
        key = (profile.get("skin_type"), profile.get("skin_tone"), wt, wn, m)
        if key in self.cache:
            return self.cache[key]
        d = self.stats.copy()
        weight = wt*d.skin_type.eq(key[0]).astype(float) + wn*d.skin_tone.eq(key[1]).astype(float)
        d["weighted_sum"] = d["sum"]*weight
        d["weighted_count"] = d["count"]*weight
        d = d[weight > 0]
        grouped = d.groupby("product_id")[["weighted_sum", "weighted_count"]].sum().reset_index()
        if not grouped.empty:
            prior = grouped.weighted_sum.sum()/grouped.weighted_count.sum()
            grouped["score"] = (grouped.weighted_sum + m*prior)/(grouped.weighted_count+m)
        self.cache[key] = grouped
        return grouped

    def recommend(self, profile, config, top_n=10, exclude=()):
        return rank_scores(self.scores(profile, config), "score", top_n, exclude)


class SupervisedAffinity:
    """Regresión logística: perfil + atributos del producto + interacciones.

    No utiliza author_id, product_id, rating del catálogo ni volumen de reviews
    como features. Negativos = reviews no positivas realmente observadas.
    """
    def __init__(self, ctx, max_train_rows=150000, max_text_features=500, seed=42):
        self.ctx = ctx
        self.catalog = ctx.catalog.set_index("product_id", drop=False)
        self.product_order = list(self.catalog.index)
        r = ctx.train.copy()
        self.training_rows_available = len(r)
        if len(r) > max_train_rows:
            # Muestra estratificada aproximadamente proporcional a la clase original.
            portions = []
            for _, group in r.groupby("positive"):
                count = max(1, int(round(max_train_rows*len(group)/len(r))))
                portions.append(group.sample(n=min(count, len(group)), random_state=seed))
            r = pd.concat(portions).sort_values(["author_id", "product_id"])
        self.training_rows_used = len(r)
        self.y = r.positive.astype(int).to_numpy()
        if len(np.unique(self.y)) < 2:
            raise ValueError("La regresión logística necesita reviews positivas y negativas en train")
        self.train_user_ids = set(r.author_id)
        self.types = sorted(set(ctx.profiles.loc[sorted(ctx.train.author_id.unique()), "skin_type"]))
        self.numeric_price = pd.to_numeric(self.catalog.get("price_usd", pd.Series(0, index=self.catalog.index)), errors="coerce")
        self.numeric_price = self.numeric_price.clip(lower=0).fillna(self.numeric_price.median()).fillna(0)
        logprice = np.log1p(self.numeric_price)
        self.price_mean, self.price_std = float(logprice.mean()), float(logprice.std(ddof=0)) or 1.0
        self.price = ((logprice-self.price_mean)/self.price_std).to_dict()
        text_cols = [c for c in ["primary_category", "secondary_category", "tertiary_category", "highlights", "ingredients"] if c in self.catalog]
        texts = self.catalog[text_cols].apply(lambda row: " ".join(text_value(v) for v in row), axis=1)
        self.tfidf = TfidfVectorizer(max_features=max_text_features, stop_words="english", dtype=np.float32)
        try:
            self.X_product_text = self.tfidf.fit_transform(texts)
        except ValueError as e:
            if "empty vocabulary" not in str(e):
                raise
            self.X_product_text = self.tfidf.fit_transform(["skincare"]*len(texts))
        self.product_positions = {pid: i for i, pid in enumerate(self.product_order)}
        profiles = [ctx.profiles.loc[uid].to_dict() for uid in r.author_id]
        pids = r.product_id.tolist()
        self.dict_encoder = DictVectorizer(sparse=True, dtype=np.float32)
        dictionaries = [self.feature_dict(profile, pid) for profile, pid in zip(profiles, pids)]
        self.dict_encoder.fit(dictionaries)
        self.X_train = self.design(profiles, pids)
        self.models = {}
        self.score_cache = {}

    def feature_dict(self, profile, pid):
        product = self.catalog.loc[pid]
        d = {c: str(profile.get(c, "Unknown")) for c in PROFILE_COLS}
        for c in ["brand_name", "secondary_category", "tertiary_category"]:
            d[c] = text_value(product.get(c)) or "Unknown"
        # Cruce explícito: permite aprender afinidades del tipo de piel con categorías.
        d["skin_x_category"] = d["skin_type"]+"|"+d["secondary_category"]
        d["log_price_scaled"] = float(self.price[pid])
        return d

    def design(self, profiles, pids):
        dictionaries = [self.feature_dict(p, pid) for p, pid in zip(profiles, pids)]
        X_cat = self.dict_encoder.transform(dictionaries)
        positions = [self.product_positions[pid] for pid in pids]
        X_text = self.X_product_text[positions]
        blocks = [X_cat, X_text]
        # Interacción tipo de piel × texto: términos pueden tener efectos diferentes.
        skin = np.array([p.get("skin_type", "Unknown") for p in profiles], dtype=object)
        for value in self.types:
            blocks.append(X_text.multiply((skin == value).astype(np.float32)[:, None]))
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    def fit_config(self, config):
        C = float(config["C"])
        if C not in self.models:
            model = LogisticRegression(C=C, solver="liblinear", max_iter=500, random_state=42)
            model.fit(self.X_train, self.y)
            self.models[C] = model
        return self.models[C]

    def scores(self, profile, config):
        C = float(config["C"])
        key = (tuple(profile.get(c, "Unknown") for c in PROFILE_COLS), C)
        if key not in self.score_cache:
            model = self.fit_config(config)
            X = self.design([profile]*len(self.product_order), self.product_order)
            probabilities = model.predict_proba(X)[:, 1]
            self.score_cache[key] = pd.DataFrame({"product_id": self.product_order, "score": probabilities})
        return self.score_cache[key]

    def recommend(self, profile, config, top_n=10, exclude=()):
        return rank_scores(self.scores(profile, config), "score", top_n, exclude)


def evaluate(model, ctx, cases, config, verbose=True):
    records = []
    for num, row in enumerate(cases.itertuples(index=False), 1):
        profile = ctx.profiles.loc[row.author_id].to_dict()
        # Cold start estricto: la consulta no dispone de historial.
        # Ni siquiera filtramos los otros positivos ocultos del usuario.
        exclude = set()
        started = time.perf_counter()
        ids = model.recommend(profile, config, TOP_N, exclude)
        elapsed = time.perf_counter()-started
        assert len(ids) == len(set(ids)) and len(ids) <= TOP_N
        assert not (set(ids) & exclude)
        assert set(ids).issubset(set(ctx.catalog.product_id))
        position = ids.index(row.target)+1 if row.target in ids else None
        records.append({"author_id": row.author_id, "target": row.target,
                        "hit": int(position is not None), "ndcg_at_10": 1/np.log2(position+1) if position else 0.0,
                        "n_recommendations": len(ids), "seconds": elapsed, "ids": ids})
        if verbose and (num % 50 == 0 or num == len(cases)):
            print(f"Evaluados {num}/{len(cases)}", flush=True)
    detail = pd.DataFrame(records)
    union = {pid for ids in detail.ids for pid in ids}
    metrics = {
        "users": len(detail), "hits": int(detail.hit.sum()),
        "hit_rate_at_10": float(detail.hit.mean()), "ndcg_at_10": float(detail.ndcg_at_10.mean()),
        "user_coverage": float(detail.n_recommendations.gt(0).mean()),
        "full_top10_rate": float(detail.n_recommendations.eq(TOP_N).mean()),
        "catalog_coverage": len(union)/len(ctx.catalog),
        "mean_recommendations": float(detail.n_recommendations.mean()),
        "mean_query_seconds": float(detail.seconds.mean()),
    }
    return metrics, detail


def run_experiment(model, ctx, configs, verbose=True):
    """Ajusta SOLO en validación; evalúa test una sola vez con el elegido."""
    validation_results = []
    for config in configs:
        if verbose:
            print("Validación:", config, flush=True)
        if hasattr(model, "fit_config"):
            model.fit_config(config)
        metrics, _ = evaluate(model, ctx, ctx.validation, config, verbose)
        validation_results.append({"config": dict(config), **metrics})
    # Prioridad HR, luego posición del hit; empate total: primera configuración.
    best_index = max(range(len(configs)), key=lambda i: (
        validation_results[i]["hit_rate_at_10"], validation_results[i]["ndcg_at_10"]))
    best = dict(configs[best_index])
    if verbose:
        print("Configuración seleccionada:", best, "\nEvaluación test", flush=True)
    test_metrics, test_detail = evaluate(model, ctx, ctx.test, best, verbose)
    baseline_metrics, _ = evaluate(PopularityBaseline(ctx), ctx, ctx.test, {}, False)
    return best, pd.DataFrame(validation_results), test_metrics, test_detail, baseline_metrics


def example_top10(model, ctx, config, profile=None):
    if profile is None:
        profile = {"skin_type": "oily", "skin_tone": "dark", "eye_color": "blue", "hair_color": "brown"}
    ids = model.recommend(profile, config, TOP_N, ())
    table = pd.DataFrame({"position": np.arange(1, len(ids)+1), "product_id": ids})
    table = table.merge(ctx.catalog, on="product_id", how="left", validate="one_to_one")
    cols = [c for c in ["position", "product_id", "product_name", "brand_name", "secondary_category", "price_usd"] if c in table]
    return profile, table[cols]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")


def save_result(ctx, name, best, validation_table, metrics, detail, baseline, example, extra=None):
    if ctx.output_dir is None:
        raise ValueError("Contexto sin carpeta de resultados")
    payload = {"model": name, "protocol_id": ctx.protocol_id, "protocol_version": PROTOCOL_VERSION,
               "selected_config": best, "validation": validation_table.to_dict("records"),
               "test": metrics, "baseline_test": baseline, "extra": extra or {}}
    write_json(ctx.output_dir / f"{name}.json", payload)
    detail.assign(ids=detail.ids.map(json.dumps)).to_csv(ctx.output_dir / f"{name}_test_detail.csv", index=False)
    example.to_csv(ctx.output_dir / f"{name}_example_top10.csv", index=False)
    return ctx.output_dir / f"{name}.json"


def conclusion(metrics, baseline):
    delta = metrics["hit_rate_at_10"]-baseline["hit_rate_at_10"]
    print(f"HR@10 test: {metrics['hit_rate_at_10']:.2%}; baseline: {baseline['hit_rate_at_10']:.2%}; diferencia: {delta*100:+.2f} puntos porcentuales.")
    print(f"Cobertura de usuarios: {metrics['user_coverage']:.2%}; catálogo: {metrics['catalog_coverage']:.2%}.")
    if delta > 0:
        print("En esta muestra supera al baseline en HR; considerar incertidumbre, cobertura y objetivo antes de elegir.")
    else:
        print("En esta muestra no supera al baseline en HR. No declararlo ganador por personalización solamente.")
    print("Un hit recupera una preferencia observada; no prueba eficacia ni compatibilidad dermatológica.")
